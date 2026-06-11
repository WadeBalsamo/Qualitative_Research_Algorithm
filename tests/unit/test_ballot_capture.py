"""Hermetic tests for orchestrator._persist_ballots_from_results (no LLM).

Exercises the inline ballot-capture hook with synthetic segments / cue-unit
records carrying rater_votes, asserting runs + label_ballots land correctly.
"""
import os
import shutil
import sys
import tempfile
import types
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from process import db
from process import run_registry as rr
from process.orchestrator import _persist_ballots_from_results


def _theme_seg(seg_id, rater_ids, rater_votes):
    """A minimal stand-in carrying the attributes the hook reads (theme)."""
    return types.SimpleNamespace(
        segment_id=seg_id, rater_ids=rater_ids, rater_votes=rater_votes)


class TestThemeCapture(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_segments_produce_runs_and_ballots(self):
        segs = [
            _theme_seg('s1', ['mA', 'mB'], [
                {'rater': 'mA', 'vote': 'CODED', 'stage': 2, 'confidence': 0.9,
                 'secondary_stage': None, 'secondary_confidence': None,
                 'justification': 'j'},
                {'rater': 'mB', 'vote': 'ABSTAIN', 'stage': None, 'confidence': None,
                 'secondary_stage': None, 'secondary_confidence': None,
                 'justification': ''},
            ]),
            _theme_seg('s2', ['mA', 'mB'], [
                {'rater': 'mA', 'vote': 'ERROR', 'stage': None, 'confidence': None,
                 'secondary_stage': None, 'secondary_confidence': None,
                 'justification': ''},
                {'rater': 'mB', 'vote': 'CODED', 'stage': 1, 'confidence': 0.7,
                 'secondary_stage': None, 'secondary_confidence': None,
                 'justification': 'j2'},
            ]),
        ]
        _persist_ballots_from_results(self.tmpdir, 'theme', segs)

        runs = rr.list_runs(self.tmpdir, overlay='theme')
        self.assertEqual([r['rater_label'] for r in runs], ['mA', 'mB'])
        for r in runs:
            self.assertEqual(r['status'], 'completed')
            self.assertTrue(r['selected'])
            self.assertEqual(r['n_total'], 2)
        run_by = {r['rater_label']: r for r in runs}
        self.assertEqual(run_by['mA']['n_coded'], 1)
        self.assertEqual(run_by['mA']['n_error'], 1)
        self.assertEqual(run_by['mB']['n_coded'], 1)
        self.assertEqual(run_by['mB']['n_abstain'], 1)

        got = rr.ballots_for_runs(
            self.tmpdir, 'theme', [run_by['mA']['run_id'], run_by['mB']['run_id']])
        # ERROR cell -> None.  A CODED cell is stored in the canonical parsed-run
        # shape (primary_stage/primary_confidence) so it re-votes byte-identically
        # via build_merge_result (the M1 capture-gap fix); the DB stage column is
        # decomposed from it for counters/queries.
        self.assertIsNone(got['s2'][run_by['mA']['run_id']])
        self.assertEqual(got['s1'][run_by['mA']['run_id']]['primary_stage'], 2)
        self.assertEqual(got['s1'][run_by['mA']['run_id']]['vote'], 'CODED')

    def test_segments_without_votes_skipped(self):
        segs = [_theme_seg('s1', None, None)]
        _persist_ballots_from_results(self.tmpdir, 'theme', segs)
        self.assertEqual(rr.list_runs(self.tmpdir, overlay='theme'), [])

    def test_idempotent_recapture(self):
        segs = [_theme_seg('s1', ['mA'], [
            {'rater': 'mA', 'vote': 'CODED', 'stage': 2, 'confidence': 0.9,
             'secondary_stage': None, 'secondary_confidence': None,
             'justification': 'j'},
        ])]
        _persist_ballots_from_results(self.tmpdir, 'theme', segs)
        _persist_ballots_from_results(self.tmpdir, 'theme', segs)  # re-run
        runs = rr.list_runs(self.tmpdir, overlay='theme')
        self.assertEqual(len(runs), 1)  # get-or-create, no dup
        with db.open_db(self.tmpdir) as conn:
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM label_ballots").fetchone()[0], 1)

    def test_recapture_preserves_deselection(self):
        """A re-capture must NOT re-select a run the user deselected via
        `qra runs select` (the existing-row branch updates status only)."""
        segs = [
            _theme_seg('s1', ['mA', 'mB'], [
                {'rater': 'mA', 'vote': 'CODED', 'stage': 2, 'confidence': 0.9,
                 'secondary_stage': None, 'secondary_confidence': None,
                 'justification': 'j'},
                {'rater': 'mB', 'vote': 'CODED', 'stage': 1, 'confidence': 0.7,
                 'secondary_stage': None, 'secondary_confidence': None,
                 'justification': 'j2'},
            ]),
        ]
        _persist_ballots_from_results(self.tmpdir, 'theme', segs)
        runs = {r['rater_label']: r['run_id'] for r in rr.list_runs(self.tmpdir, overlay='theme')}
        # Curate: keep only mA selected (deselect mB).
        rr.set_selected(self.tmpdir, 'theme', [runs['mA']])
        self.assertEqual(rr.selected_runs(self.tmpdir, 'theme'), [runs['mA']])

        # Re-run capture (e.g. a re-classify) — mB must STAY deselected.
        _persist_ballots_from_results(self.tmpdir, 'theme', segs)
        self.assertEqual(rr.selected_runs(self.tmpdir, 'theme'), [runs['mA']])
        self.assertFalse(rr.get_run(self.tmpdir, runs['mB'])['selected'])
        # Both runs still completed.
        self.assertEqual(rr.get_run(self.tmpdir, runs['mB'])['status'], 'completed')


class TestPurerCueCapture(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cue_unit_records_applies_to_and_errors(self):
        records = [
            # A successful 2-constituent cue unit (applies_to = both ids).
            {'segment_ids': ['t1', 't2'], 'rater_ids': ['pM'],
             'rater_votes': [
                 {'rater': 'pM', 'vote': 'CODED', 'stage': 0, 'confidence': 0.8,
                  'secondary_stage': None, 'secondary_confidence': None,
                  'justification': 'p'}]},
            # A failed unit (no rater_votes) -> ERROR ballot for the roster rater.
            {'segment_ids': ['t3'], 'rater_ids': ['pM'], 'rater_votes': None},
        ]
        _persist_ballots_from_results(self.tmpdir, 'purer', records)

        runs = rr.list_runs(self.tmpdir, overlay='purer')
        self.assertEqual(len(runs), 1)
        run_id = runs[0]['run_id']
        self.assertEqual(runs[0]['n_coded'], 2)   # t1 + t2 propagated
        self.assertEqual(runs[0]['n_error'], 1)   # t3 failed

        with db.open_db(self.tmpdir) as conn:
            row = conn.execute(
                "SELECT applies_to_json FROM label_ballots "
                "WHERE run_id=? AND segment_id='t1'", (run_id,)
            ).fetchone()
            self.assertEqual(db.loads(row['applies_to_json']), ['t1', 't2'])
            err = conn.execute(
                "SELECT vote, raw_json FROM label_ballots "
                "WHERE run_id=? AND segment_id='t3'", (run_id,)
            ).fetchone()
        self.assertEqual(err['vote'], 'ERROR')
        self.assertIsNone(err['raw_json'])


if __name__ == '__main__':
    unittest.main()
