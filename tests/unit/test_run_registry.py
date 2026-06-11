"""Tests for process/run_registry.py — schema-v2 run registry + durable ballots."""
import os
import shutil
import sys
import tempfile
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from process import db
from process import run_registry as rr


class TestComposeRaterLabel(unittest.TestCase):
    def test_plain_model(self):
        self.assertEqual(rr.compose_rater_label('qwen-3-70b'), 'qwen-3-70b')

    def test_alias_wins(self):
        self.assertEqual(
            rr.compose_rater_label('qwen-3-70b', quantization='Q4', alias='myrun'),
            'myrun',
        )

    def test_quant_only(self):
        self.assertEqual(rr.compose_rater_label('m', quantization='Q4'), 'm[Q4]')

    def test_thinking_only(self):
        self.assertEqual(rr.compose_rater_label('m', thinking='off'), 'm[think:off]')

    def test_quant_and_thinking(self):
        self.assertEqual(
            rr.compose_rater_label('m', quantization='Q4', thinking='off'),
            'm[Q4,think:off]',
        )

    def test_none_parts_omitted(self):
        self.assertEqual(
            rr.compose_rater_label('m', quantization=None, thinking=None), 'm'
        )

    def test_dedupe(self):
        existing = {'m', 'm#2'}
        self.assertEqual(rr.compose_rater_label('m', existing=existing), 'm#3')

    def test_dedupe_alias(self):
        existing = {'a'}
        self.assertEqual(rr.compose_rater_label('m', alias='a', existing=existing), 'a#2')


class TestRunRegistry(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # ensure a fresh v2 schema
        with db.open_db(self.tmpdir) as conn:
            self.assertEqual(db.get_meta(conn, 'schema_version'), '2')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_and_get_run(self):
        rid = rr.create_run(self.tmpdir, overlay='theme', model='modelA',
                            backend='ollama', note='hi', temperature=0.7,
                            params={'k': 1})
        self.assertIsInstance(rid, int)
        run = rr.get_run(self.tmpdir, rid)
        self.assertEqual(run['overlay'], 'theme')
        self.assertEqual(run['model'], 'modelA')
        self.assertEqual(run['rater_label'], 'modelA')  # auto-composed
        self.assertEqual(run['backend'], 'ollama')
        self.assertEqual(run['status'], 'queued')
        self.assertFalse(run['selected'])
        self.assertEqual(run['params'], {'k': 1})

    def test_get_run_missing(self):
        self.assertIsNone(rr.get_run(self.tmpdir, 999))

    def test_auto_label_dedupes_within_overlay(self):
        r1 = rr.create_run(self.tmpdir, overlay='theme', model='m')
        r2 = rr.create_run(self.tmpdir, overlay='theme', model='m')
        l1 = rr.get_run(self.tmpdir, r1)['rater_label']
        l2 = rr.get_run(self.tmpdir, r2)['rater_label']
        self.assertEqual(l1, 'm')
        self.assertEqual(l2, 'm#2')

    def test_explicit_rater_label(self):
        rid = rr.create_run(self.tmpdir, overlay='purer', model='m',
                            rater_label='custom')
        self.assertEqual(rr.get_run(self.tmpdir, rid)['rater_label'], 'custom')

    def test_list_runs_filters(self):
        rr.create_run(self.tmpdir, overlay='theme', model='a')
        rr.create_run(self.tmpdir, overlay='theme', model='b')
        rr.create_run(self.tmpdir, overlay='purer', model='c')
        self.assertEqual(len(rr.list_runs(self.tmpdir)), 3)
        self.assertEqual(len(rr.list_runs(self.tmpdir, overlay='theme')), 2)
        self.assertEqual(len(rr.list_runs(self.tmpdir, overlay='purer')), 1)
        self.assertEqual(
            len(rr.list_runs(self.tmpdir, overlay='theme', statuses=['queued'])), 2)
        self.assertEqual(
            len(rr.list_runs(self.tmpdir, overlay='theme', statuses=['completed'])), 0)

    def test_list_runs_ordered_by_run_id(self):
        ids = [rr.create_run(self.tmpdir, overlay='theme', model=f'm{i}')
               for i in range(4)]
        got = [r['run_id'] for r in rr.list_runs(self.tmpdir, overlay='theme')]
        self.assertEqual(got, sorted(ids))

    def test_update_run_whitelist(self):
        rid = rr.create_run(self.tmpdir, overlay='theme', model='m')
        rr.update_run(self.tmpdir, rid, status='completed', selected=True,
                      n_total=10, completed_at='2026-01-01T00:00:00')
        run = rr.get_run(self.tmpdir, rid)
        self.assertEqual(run['status'], 'completed')
        self.assertTrue(run['selected'])
        self.assertEqual(run['n_total'], 10)
        self.assertEqual(run['completed_at'], '2026-01-01T00:00:00')

    def test_update_run_rejects_unknown(self):
        rid = rr.create_run(self.tmpdir, overlay='theme', model='m')
        with self.assertRaises(ValueError):
            rr.update_run(self.tmpdir, rid, overlay='purer')

    def test_update_run_params_alias(self):
        rid = rr.create_run(self.tmpdir, overlay='theme', model='m')
        rr.update_run(self.tmpdir, rid, params={'x': 2})
        self.assertEqual(rr.get_run(self.tmpdir, rid)['params'], {'x': 2})

    def test_set_selected_and_selected_runs(self):
        ids = [rr.create_run(self.tmpdir, overlay='theme', model=f'm{i}')
               for i in range(3)]
        rr.set_selected(self.tmpdir, 'theme', [ids[2], ids[0]])
        # ordered by run_id, regardless of input order
        self.assertEqual(rr.selected_runs(self.tmpdir, 'theme'), sorted([ids[0], ids[2]]))
        # re-select clears the previous set
        rr.set_selected(self.tmpdir, 'theme', [ids[1]])
        self.assertEqual(rr.selected_runs(self.tmpdir, 'theme'), [ids[1]])
        # empty clears all
        rr.set_selected(self.tmpdir, 'theme', [])
        self.assertEqual(rr.selected_runs(self.tmpdir, 'theme'), [])

    def test_selected_runs_scoped_to_overlay(self):
        t = rr.create_run(self.tmpdir, overlay='theme', model='m')
        p = rr.create_run(self.tmpdir, overlay='purer', model='m')
        rr.set_selected(self.tmpdir, 'theme', [t])
        self.assertEqual(rr.selected_runs(self.tmpdir, 'theme'), [t])
        self.assertEqual(rr.selected_runs(self.tmpdir, 'purer'), [])
        self.assertNotIn(p, rr.selected_runs(self.tmpdir, 'theme'))


class TestBallots(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.run_id = rr.create_run(self.tmpdir, overlay='theme', model='m')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_none_cell_is_error_row(self):
        n = rr.upsert_ballots(self.tmpdir, 'theme', self.run_id, {'s1': None})
        self.assertEqual(n, 1)
        got = rr.ballots_for_runs(self.tmpdir, 'theme', [self.run_id])
        self.assertEqual(got['s1'][self.run_id], None)  # NULL raw_json -> None
        # row carries vote='ERROR', NULL stage
        with db.open_db(self.tmpdir) as conn:
            row = conn.execute(
                "SELECT vote, stage, raw_json FROM label_ballots "
                "WHERE segment_id='s1' AND run_id=?", (self.run_id,)
            ).fetchone()
        self.assertEqual(row['vote'], 'ERROR')
        self.assertIsNone(row['stage'])
        self.assertIsNone(row['raw_json'])

    def test_coded_cell_decomposed(self):
        cell = {
            'vote': 'CODED', 'primary_stage': 2, 'primary_confidence': 0.9,
            'secondary_stage': 3, 'secondary_confidence': 0.4,
            'justification': 'because',
        }
        rr.upsert_ballots(self.tmpdir, 'theme', self.run_id, {'s1': cell})
        with db.open_db(self.tmpdir) as conn:
            row = conn.execute(
                "SELECT * FROM label_ballots WHERE segment_id='s1' AND run_id=?",
                (self.run_id,),
            ).fetchone()
        self.assertEqual(row['vote'], 'CODED')
        self.assertEqual(row['stage'], 2)
        self.assertAlmostEqual(row['confidence'], 0.9)
        self.assertEqual(row['secondary_stage'], 3)
        self.assertEqual(row['justification'], 'because')

    def test_vote_inferred_from_primary_stage(self):
        # No explicit 'vote'; primary_stage present -> CODED.
        rr.upsert_ballots(self.tmpdir, 'theme', self.run_id,
                          {'s1': {'primary_stage': 1}})
        # primary_stage None -> ABSTAIN.
        rr.upsert_ballots(self.tmpdir, 'theme', self.run_id,
                          {'s2': {'primary_stage': None}})
        with db.open_db(self.tmpdir) as conn:
            votes = dict(conn.execute(
                "SELECT segment_id, vote FROM label_ballots WHERE run_id=?",
                (self.run_id,),
            ).fetchall())
        self.assertEqual(votes['s1'], 'CODED')
        self.assertEqual(votes['s2'], 'ABSTAIN')

    def test_raw_json_byte_fidelity_roundtrip(self):
        cell = {
            'vote': 'CODED', 'primary_stage': 4, 'primary_confidence': 0.55,
            'secondary_stage': None, 'secondary_confidence': None,
            'justification': 'résumé — quote "q" \\ slash',
            'extra_nested': {'a': [1, 2, {'b': True}]},
        }
        rr.upsert_ballots(self.tmpdir, 'theme', self.run_id, {'s1': cell})
        got = rr.ballots_for_runs(self.tmpdir, 'theme', [self.run_id])
        self.assertEqual(got['s1'][self.run_id], cell)  # exact dict round-trip

    def test_upsert_replaces_by_pk(self):
        rr.upsert_ballots(self.tmpdir, 'theme', self.run_id,
                          {'s1': {'vote': 'CODED', 'primary_stage': 1}})
        rr.upsert_ballots(self.tmpdir, 'theme', self.run_id,
                          {'s1': {'vote': 'CODED', 'primary_stage': 2}})
        got = rr.ballots_for_runs(self.tmpdir, 'theme', [self.run_id])
        self.assertEqual(got['s1'][self.run_id]['primary_stage'], 2)
        with db.open_db(self.tmpdir) as conn:
            cnt = conn.execute(
                "SELECT COUNT(*) FROM label_ballots WHERE segment_id='s1' AND run_id=?",
                (self.run_id,),
            ).fetchone()[0]
        self.assertEqual(cnt, 1)

    def test_applies_to_recorded(self):
        rr.upsert_ballots(self.tmpdir, 'theme', self.run_id,
                          {'s1': {'vote': 'CODED', 'primary_stage': 1}},
                          applies_to={'s1': ['s1', 's2', 's3']})
        with db.open_db(self.tmpdir) as conn:
            row = conn.execute(
                "SELECT applies_to_json FROM label_ballots "
                "WHERE segment_id='s1' AND run_id=?", (self.run_id,)
            ).fetchone()
        self.assertEqual(db.loads(row['applies_to_json']), ['s1', 's2', 's3'])

    def test_ballots_for_runs_empty(self):
        self.assertEqual(rr.ballots_for_runs(self.tmpdir, 'theme', []), {})

    def test_refresh_counters(self):
        rr.upsert_ballots(self.tmpdir, 'theme', self.run_id, {
            's1': {'vote': 'CODED', 'primary_stage': 1},
            's2': {'vote': 'CODED', 'primary_stage': 2},
            's3': {'vote': 'ABSTAIN', 'primary_stage': None},
            's4': None,  # ERROR
        })
        rr.refresh_counters(self.tmpdir, self.run_id)
        run = rr.get_run(self.tmpdir, self.run_id)
        self.assertEqual(run['n_coded'], 2)
        self.assertEqual(run['n_abstain'], 1)
        self.assertEqual(run['n_error'], 1)
        self.assertEqual(run['n_total'], 4)

    def test_ballots_for_runs_multi_run(self):
        r2 = rr.create_run(self.tmpdir, overlay='theme', model='m2')
        rr.upsert_ballots(self.tmpdir, 'theme', self.run_id,
                          {'s1': {'vote': 'CODED', 'primary_stage': 1}})
        rr.upsert_ballots(self.tmpdir, 'theme', r2,
                          {'s1': {'vote': 'CODED', 'primary_stage': 2}})
        got = rr.ballots_for_runs(self.tmpdir, 'theme', [self.run_id, r2])
        self.assertEqual(got['s1'][self.run_id]['primary_stage'], 1)
        self.assertEqual(got['s1'][r2]['primary_stage'], 2)


class TestRemapBallotSegmentIds(unittest.TestCase):
    """remap_ballot_segment_ids — anonymization-key cascade for durable ballots."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.run_id = rr.create_run(self.tmpdir, overlay='theme', model='m')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_empty_map_is_noop(self):
        rr.upsert_ballots(self.tmpdir, 'theme', self.run_id,
                          {'a': {'vote': 'CODED', 'primary_stage': 1}})
        self.assertEqual(rr.remap_ballot_segment_ids(self.tmpdir, {}), 0)

    def test_id_swap_a_to_b_b_to_a(self):
        # A swap is the collision-stress case the overlay remap also supports.
        rr.upsert_ballots(self.tmpdir, 'theme', self.run_id, {
            'a': {'vote': 'CODED', 'primary_stage': 1},
            'b': {'vote': 'CODED', 'primary_stage': 2},
        })
        n = rr.remap_ballot_segment_ids(self.tmpdir, {'a': 'b', 'b': 'a'})
        self.assertEqual(n, 2)
        got = rr.ballots_for_runs(self.tmpdir, 'theme', [self.run_id])
        # The ballot that WAS 'a' (stage 1) is now keyed 'b', and vice-versa.
        self.assertEqual(got['b'][self.run_id]['primary_stage'], 1)
        self.assertEqual(got['a'][self.run_id]['primary_stage'], 2)
        # No rows lost or duplicated.
        with db.open_db(self.tmpdir) as conn:
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM label_ballots").fetchone()[0], 2)

    def test_applies_to_json_rewritten_even_when_unit_id_unchanged(self):
        # A PURER cue unit 'u1' whose CONSTITUENTS (t_old1, t_old2) are remapped,
        # but the unit's own segment_id is not — applies_to must still be rewritten.
        rr.upsert_ballots(
            self.tmpdir, 'theme', self.run_id,
            {'u1': {'vote': 'CODED', 'primary_stage': 0}},
            applies_to={'u1': ['t_old1', 't_old2']},
        )
        n = rr.remap_ballot_segment_ids(self.tmpdir, {'t_old1': 't_new1', 't_old2': 't_new2'})
        self.assertEqual(n, 1)  # the u1 row (applies_to-only rewrite)
        with db.open_db(self.tmpdir) as conn:
            row = conn.execute(
                "SELECT segment_id, applies_to_json FROM label_ballots "
                "WHERE run_id = ?", (self.run_id,)).fetchone()
        self.assertEqual(row['segment_id'], 'u1')  # unit id unchanged
        self.assertEqual(db.loads(row['applies_to_json']), ['t_new1', 't_new2'])

    def test_segment_id_and_applies_to_both_remapped(self):
        rr.upsert_ballots(
            self.tmpdir, 'theme', self.run_id,
            {'old': {'vote': 'CODED', 'primary_stage': 3}},
            applies_to={'old': ['old', 'other']},
        )
        rr.remap_ballot_segment_ids(self.tmpdir, {'old': 'new', 'other': 'other2'})
        got = rr.ballots_for_runs(self.tmpdir, 'theme', [self.run_id])
        self.assertIn('new', got)
        self.assertNotIn('old', got)
        with db.open_db(self.tmpdir) as conn:
            row = conn.execute(
                "SELECT applies_to_json FROM label_ballots WHERE segment_id='new'"
            ).fetchone()
        self.assertEqual(db.loads(row['applies_to_json']), ['new', 'other2'])


if __name__ == '__main__':
    unittest.main()
