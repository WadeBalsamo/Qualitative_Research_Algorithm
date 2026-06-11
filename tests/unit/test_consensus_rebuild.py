"""M2 gate tests — process/consensus_rebuild.rebuild_overlay.

Equivalence keystone: inline classification (FakeLLMClient, multi-rater) writes
both the overlay AND durable ballots (M1 capture, born selected).  Wiping the
overlay and rebuilding from those ballots must reproduce it record-for-record,
because both paths vote through ``llm_classifier.build_merge_result``.

Also exercised: idempotency, the selection effect (selected-ballots cache),
``only_segment_ids`` subset rebuild, PURER turn-mode equivalence, and the
zero-selected-runs skip (overlay untouched).

Hermetic: no network, no model downloads — FakeLLMClient is patched at the seam
``classification_tools.theme_llm.llm_classifier.LLMClient``.
"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from classification_tools.data_structures import Segment
from process import segments_io, classifications_io as cio, run_registry as rr
from process import consensus_rebuild as crb
from process.config import PipelineConfig
from process.orchestrator import stage_classify_theme, stage_classify_purer
from tests.testhelpers import FakeLLMClient, tiny_vaamr_framework, tiny_purer_framework


_LLM_SEAM = 'classification_tools.theme_llm.llm_classifier.LLMClient'

# A participant utterance carrying both VAAMR construct cues so the FakeLLM
# responder has names to echo; the actual stage is chosen per-rater below.
_P_TEXT = ('I keep noticing the pain and try to avoid moving, but I am '
           'attending to my breath with metacognition and reappraisal.')


def _participant(seg_id, idx):
    return Segment(
        segment_id=seg_id, trial_id='t', participant_id='P1', session_id='c1s1',
        session_number=1, cohort_id=1, segment_index=idx, speaker='participant',
        text=_P_TEXT, word_count=len(_P_TEXT.split()),
        start_time_ms=idx * 1000, end_time_ms=idx * 1000 + 800,
    )


def _therapist(seg_id, idx, text='What did you notice in your body just now?'):
    return Segment(
        segment_id=seg_id, trial_id='t', participant_id='P1', session_id='c1s1',
        session_number=1, cohort_id=1, segment_index=idx, speaker='therapist',
        text=text, word_count=len(text.split()),
        start_time_ms=idx * 1000, end_time_ms=idx * 1000 + 800,
    )


def _theme_config(tmp, raters, vote_mode='majority'):
    cfg = PipelineConfig()
    cfg.output_dir = tmp
    tc = cfg.theme_classification
    tc.per_run_models = list(raters)
    tc.n_runs = len(raters)
    tc.backend = 'fake'
    tc.temperature = 0.0
    tc.vote_mode = vote_mode
    cfg.speaker_filter.mode = 'exclude'
    cfg.speaker_filter.speakers = ['therapist']
    return cfg


def _stage_responder(fake, mapping, default='Vigilance'):
    """A responder that echoes a per-model construct NAME (so distinct raters
    disagree).  Reads the model the model-first sweep set on ``fake.config``."""
    def responder(prompt):
        name = mapping.get(fake.config.model, default)
        return {'primary_stage': name, 'primary_confidence': 0.8,
                'secondary_stage': None, 'secondary_confidence': None,
                'justification': f'j-{fake.config.model}', 'evidence_phrase': 'e'}
    return responder


def _snapshot(tmp, key):
    return {r['segment_id']: r for r in cio.read_overlay(tmp, key)}


def _assert_records_equal(test, a_by_id, b_by_id, msg=''):
    test.assertEqual(set(a_by_id), set(b_by_id), f'{msg}: segment_id sets differ')
    for sid in a_by_id:
        test.assertEqual(a_by_id[sid], b_by_id[sid],
                         f'{msg}: record for {sid!r} differs')


# ---------------------------------------------------------------------------
# Theme (VAAMR) equivalence
# ---------------------------------------------------------------------------

class TestThemeRebuildEquivalence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.fw = tiny_vaamr_framework()
        # 4 participants + 1 therapist (therapist excluded from VAAMR).
        raw = [_participant(f'p{i}', i) for i in range(4)] + [_therapist('th0', 4)]
        segments_io.write_session_segments(self.tmp, 'c1s1', raw, 'hash1')

        # 3 raters; mA/mB agree on Avoidance, mC dissents to Metacognition →
        # a genuine majority (not unanimous) so the rebuild exercises voting.
        self.raters = ['mA', 'mB', 'mC']
        self.cfg = _theme_config(self.tmp, self.raters)
        fake = FakeLLMClient()
        fake._responder = _stage_responder(
            fake, {'mA': 'Avoidance', 'mB': 'Avoidance', 'mC': 'Metacognition'})
        segs = segments_io.load_segments_for_stage(self.tmp, apply=())
        with mock.patch(_LLM_SEAM, return_value=fake):
            stage_classify_theme(self.cfg, self.fw, segments=segs, output_dir=self.tmp)
        self.snapshot = _snapshot(self.tmp, 'theme')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_inline_populated_overlay_and_ballots(self):
        # Sanity: the inline path must have written the overlay AND captured
        # born-selected runs/ballots (the rebuild's input).
        self.assertEqual(len(self.snapshot), 5)  # 4 participants + 1 therapist row
        runs = rr.list_runs(self.tmp, overlay='theme')
        self.assertEqual([r['rater_label'] for r in runs], self.raters)
        self.assertEqual(rr.selected_runs(self.tmp, 'theme'),
                         [r['run_id'] for r in runs])
        # majority of mA/mB(Avoidance=1) vs mC(Metacognition) → stage 1.
        self.assertEqual(self.snapshot['p0']['primary_stage'], 1)
        self.assertEqual(self.snapshot['p0']['agreement_level'], 'majority')

    def test_rebuild_reproduces_overlay_record_for_record(self):
        cio.clear_overlay(self.tmp, 'theme')
        self.assertEqual(cio.read_overlay(self.tmp, 'theme'), [])
        stats = crb.rebuild_overlay(self.tmp, 'theme', self.cfg)
        self.assertFalse(stats.get('skipped'))
        self.assertEqual(stats['n_units'], 4)         # 4 participant units voted
        self.assertEqual(stats['n_labeled'], 4)
        self.assertEqual(stats['models_used'], self.raters)
        rebuilt = _snapshot(self.tmp, 'theme')
        _assert_records_equal(self, self.snapshot, rebuilt, 'theme rebuild')

    def test_idempotent(self):
        crb.rebuild_overlay(self.tmp, 'theme', self.cfg)
        once = _snapshot(self.tmp, 'theme')
        crb.rebuild_overlay(self.tmp, 'theme', self.cfg)
        twice = _snapshot(self.tmp, 'theme')
        _assert_records_equal(self, once, twice, 'theme idempotency')
        # and still equal to the original inline overlay.
        _assert_records_equal(self, self.snapshot, twice, 'theme idempotency vs inline')

    def test_n_changed_zero_when_unchanged(self):
        # Rebuilding over an identical overlay reports no primary changes.
        stats = crb.rebuild_overlay(self.tmp, 'theme', self.cfg)
        self.assertEqual(stats['n_changed'], 0)

    def test_manifest_records_rebuild_provenance(self):
        crb.rebuild_overlay(self.tmp, 'theme', self.cfg)
        manifest = cio.read_classification_manifest(self.tmp)
        entry = manifest['theme']
        self.assertTrue(entry['rebuilt_from_ballots'])
        self.assertEqual(entry['n_runs'], 3)
        self.assertEqual(entry['vote_mode'], 'majority')
        # 'model' is a SINGLE real model string (the first selected run's model),
        # NOT the joined rater_labels — pinning the joined string as an LLM model
        # id poisoned `qra add-data` incremental classification.
        self.assertEqual(entry['model'], 'mA')
        self.assertNotIn(',', entry['model'])
        # The full roster is preserved for incremental re-classification.
        self.assertEqual(entry['per_run_models'], ['mA', 'mB', 'mC'])
        self.assertEqual(entry['rater_labels'], ['mA', 'mB', 'mC'])
        self.assertEqual(entry['run_ids'], rr.selected_runs(self.tmp, 'theme'))

    def test_selection_effect_recomputes_consensus(self):
        # Deselect mC (the dissenter). With only mA+mB (both Avoidance), the
        # rebuilt cache must contain exactly those two raters, and the consensus
        # is now UNANIMOUS Avoidance (hand-computed).
        runs = rr.list_runs(self.tmp, overlay='theme')
        by_label = {r['rater_label']: r['run_id'] for r in runs}
        rr.set_selected(self.tmp, 'theme', [by_label['mA'], by_label['mB']])
        crb.rebuild_overlay(self.tmp, 'theme', self.cfg)

        rec = _snapshot(self.tmp, 'theme')['p0']
        self.assertEqual([v['rater'] for v in rec['rater_votes']], ['mA', 'mB'])
        self.assertEqual(rec['rater_ids'], ['mA', 'mB'])
        self.assertEqual(rec['primary_stage'], 1)               # Avoidance
        self.assertEqual(rec['agreement_level'], 'unanimous')   # 2/2 now
        self.assertEqual(rec['llm_run_consistency'], 2)

    def test_only_segment_ids_leaves_others_untouched(self):
        # Change the overlay first so a subset rebuild is observable: flip mC to
        # also dissent differently is unnecessary — instead, corrupt one row and
        # rebuild only it; the other rows must remain byte-identical.
        # Mutate p0's stored row to a wrong value via a merge write.
        seg = _participant('p0', 0)
        seg.primary_stage = 99  # sentinel wrong value
        cio.merge_theme_overlay(self.tmp, [seg])
        corrupted = _snapshot(self.tmp, 'theme')
        self.assertEqual(corrupted['p0']['primary_stage'], 99)

        stats = crb.rebuild_overlay(self.tmp, 'theme', self.cfg,
                                    only_segment_ids={'p0'})
        self.assertEqual(stats['n_units'], 1)
        after = _snapshot(self.tmp, 'theme')
        # p0 is repaired back to the correct consensus.
        self.assertEqual(after['p0'], self.snapshot['p0'])
        # every other row is exactly as it was before the subset rebuild.
        for sid in after:
            if sid == 'p0':
                continue
            self.assertEqual(after[sid], corrupted[sid],
                             f'subset rebuild touched {sid!r}')

    def test_zero_selected_runs_skips_and_leaves_overlay(self):
        rr.set_selected(self.tmp, 'theme', [])   # deselect all
        before = _snapshot(self.tmp, 'theme')
        stats = crb.rebuild_overlay(self.tmp, 'theme', self.cfg)
        self.assertTrue(stats['skipped'])
        self.assertIn('reason', stats)
        after = _snapshot(self.tmp, 'theme')
        _assert_records_equal(self, before, after, 'zero-selected overlay untouched')

    def test_bad_overlay_raises(self):
        with self.assertRaises(ValueError):
            crb.rebuild_overlay(self.tmp, 'codebook', self.cfg)


# ---------------------------------------------------------------------------
# PURER turn-mode equivalence
# ---------------------------------------------------------------------------

class TestPurerTurnModeRebuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.fw = tiny_purer_framework()
        # Interleave participant/therapist so PURER turn-mode builds cue units;
        # every therapist turn becomes its own unit (applies_to = self).
        raw = [
            _participant('p0', 0),
            _therapist('t0', 1, 'What did you notice in the body just now?'),
            _participant('p1', 2),
            _therapist('t1', 3, 'Try carrying that noticing into your week.'),
            _participant('p2', 4),
        ]
        segments_io.write_session_segments(self.tmp, 'c1s1', raw, 'hash1')

        # PURER inherits per_run_models from theme in the orchestrator; set the
        # theme roster and enable the PURER labeler. Two distinct raters give
        # clean model-string rater_labels (PURER needs >=2 for per-run raters).
        self.raters = ['pA', 'pB']
        self.cfg = PipelineConfig()
        self.cfg.output_dir = self.tmp
        self.cfg.run_purer_labeler = True
        tc = self.cfg.theme_classification
        tc.per_run_models = list(self.raters)
        tc.n_runs = len(self.raters)
        tc.backend = 'fake'
        tc.temperature = 0.0
        # turn mode is the registry-supported PURER unit (and the default).
        self.cfg.purer_cue.classification_unit = 'turn'

        fake = FakeLLMClient(default_name='Phenomenological')
        segs = segments_io.load_segments_for_stage(self.tmp, apply=())
        with mock.patch(_LLM_SEAM, return_value=fake):
            stage_classify_purer(self.cfg, segments=segs, output_dir=self.tmp)
        self.snapshot = _snapshot(self.tmp, 'purer')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_inline_labeled_therapist_segments(self):
        # Therapist turns are labeled; participant rows are present but null.
        runs = rr.list_runs(self.tmp, overlay='purer')
        self.assertEqual([r['rater_label'] for r in runs], self.raters)
        self.assertEqual(rr.selected_runs(self.tmp, 'purer'),
                         [r['run_id'] for r in runs])
        self.assertIsNotNone(self.snapshot['t0']['purer_primary'])
        self.assertIsNotNone(self.snapshot['t1']['purer_primary'])

    def test_rebuild_reproduces_purer_overlay(self):
        cio.clear_overlay(self.tmp, 'purer')
        stats = crb.rebuild_overlay(self.tmp, 'purer', self.cfg)
        self.assertFalse(stats.get('skipped'))
        # 2 therapist turn-units voted.
        self.assertEqual(stats['n_units'], 2)
        rebuilt = _snapshot(self.tmp, 'purer')
        _assert_records_equal(self, self.snapshot, rebuilt, 'purer rebuild')

    def test_purer_idempotent(self):
        crb.rebuild_overlay(self.tmp, 'purer', self.cfg)
        once = _snapshot(self.tmp, 'purer')
        crb.rebuild_overlay(self.tmp, 'purer', self.cfg)
        twice = _snapshot(self.tmp, 'purer')
        _assert_records_equal(self, once, twice, 'purer idempotency')


# ---------------------------------------------------------------------------
# Backfill (v1 -> v2) rebuild equivalence — the M1 "rebuild reproduces today's
# overlays" guarantee for migrated projects (the gemma-incident repair path).
# ---------------------------------------------------------------------------

class TestBackfillRebuild(unittest.TestCase):
    """A v1 DB's legacy ``rater_votes`` cache backfills into born-selected runs;
    rebuilding from those backfilled ballots must re-vote correctly (CODED
    ballots keep their primary), not collapse to NULL — exercises the
    ``ballots_for_runs`` shape normalization."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        from process import db
        # Stand up v2 tables, force version back to 1, seed legacy caches.
        dbf = db.db_path(self.tmp)
        conn = db.connect(dbf)
        db.ensure_schema(conn)
        db.set_meta(conn, 'schema_version', '1')
        conn.execute("DELETE FROM classification_runs")
        conn.execute("DELETE FROM label_ballots")
        conn.execute(
            "INSERT INTO segments (segment_id, session_id, speaker, params_hash) "
            "VALUES ('p0','c1s1','participant','h1')")
        rv = [
            {'rater': 'mA', 'vote': 'CODED', 'stage': 1, 'confidence': 0.8,
             'secondary_stage': None, 'secondary_confidence': None, 'justification': 'jA'},
            {'rater': 'mB', 'vote': 'CODED', 'stage': 1, 'confidence': 0.8,
             'secondary_stage': None, 'secondary_confidence': None, 'justification': 'jB'},
            {'rater': 'mC', 'vote': 'CODED', 'stage': 3, 'confidence': 0.8,
             'secondary_stage': None, 'secondary_confidence': None, 'justification': 'jC'},
        ]
        import json as _json
        conn.execute(
            "INSERT INTO theme_labels (segment_id, primary_stage, consensus_vote, "
            "rater_ids, rater_votes) VALUES (?,?,?,?,?)",
            ('p0', 1, _json.dumps(1), _json.dumps(['mA', 'mB', 'mC']), _json.dumps(rv)))
        conn.commit()
        conn.close()
        # Open via open_db -> triggers the v1->v2 backfill.
        with db.open_db(self.tmp) as c:
            assert db.get_meta(c, 'schema_version') == '2'

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_rebuild_from_backfilled_ballots_keeps_majority(self):
        cfg = _theme_config(self.tmp, ['mA', 'mB', 'mC'])  # vote_mode majority
        self.assertEqual(rr.selected_runs(self.tmp, 'theme'), [1, 2, 3])
        stats = crb.rebuild_overlay(self.tmp, 'theme', cfg)
        self.assertEqual(stats['n_labeled'], 1)   # NOT collapsed to unlabeled
        rec = _snapshot(self.tmp, 'theme')['p0']
        # mA/mB(stage1) majority over mC(stage3) → stage 1 reproduced.
        self.assertEqual(rec['primary_stage'], 1)
        self.assertEqual(rec['agreement_level'], 'majority')
        self.assertEqual([v['stage'] for v in rec['rater_votes']], [1, 1, 3])


if __name__ == '__main__':
    unittest.main()
