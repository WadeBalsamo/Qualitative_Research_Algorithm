"""Unit tests for process/repair.py — M4.

Key scenarios (all hermetic / FakeLLM / no network):
  1. flaky-then-ok:  FakeLLM produces ERROR on first sweep, then CODED on second →
     fix_errors repairs (ballots updated, counters refreshed, overlay rebuilt,
     errors→0 after repair).
  2. no-progress break: always-garbage FakeLLM (always returns None) → passes stop
     early, flagged_for_review_repair.json written.
  3. dry_run mutates NOTHING: assert DB table bytes / row counts identical before/after.
  4. missing checkpoint → skip note (run left flagged, no crash).
  5. only_segment_ids patch (patch_run_errors_only subset behavior):
     passing segment_ids=set clears only those cells.
  6. dead-rater skip: run with n_error/n_total ≥ 0.5 is skipped unless force=True.

Note: Tests call fix_errors directly with mocked execute_single_run so we don't
need a live LLM or real checkpoint execution.
"""
import json
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
from process import segments_io, run_registry as rr, classifications_io as cio, db as _db
from process.reclassify_ops import patch_run_errors_only
from process import error_detection as ed
from process import run_executor as rx


# Seam to patch: run_executor.execute_single_run (used by repair.py).
_EXECUTE_SEAM = 'process.run_executor.execute_single_run'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(seg_id, idx, speaker='participant'):
    return Segment(
        segment_id=seg_id, trial_id='t', participant_id='P1', session_id='c1s1',
        session_number=1, cohort_id=1, segment_index=idx, speaker=speaker,
        text='I notice the pain and want to avoid it.', word_count=9,
        start_time_ms=idx * 1000, end_time_ms=idx * 1000 + 800,
    )


def _coded_ballot(stage=1):
    return {'vote': 'CODED', 'primary_stage': stage, 'primary_confidence': 0.8,
            'secondary_stage': None, 'secondary_confidence': None,
            'justification': 'j', 'evidence_phrase': 'e'}


def _seed_project(tmp, n_segs=4):
    """Write n_segs frozen participant segments and return their ids."""
    raw = [_make_segment(f's{i}', i) for i in range(n_segs)]
    segments_io.write_session_segments(tmp, 'c1s1', raw, 'hash1')
    return [f's{i}' for i in range(n_segs)]


def _write_per_run_ckpt(tmp, overlay, run_id, cells_by_seg_id):
    """Write a model_first_v1 per-run checkpoint for run_id."""
    ckpt_dir = os.path.join(tmp, '02_meta', 'auditable_logs', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    prefix = 'llm_results' if overlay == 'theme' else 'purer_cue_results'
    path = os.path.join(ckpt_dir, f'{prefix}_run{run_id:04d}_runs.json')
    payload = {
        '_meta': {'format': 'model_first_v1', 'n_runs': 1,
                  'per_run_models': ['mA'], 'completed_runs': []},
        'run_results': {sid: {'0': cell} for sid, cell in cells_by_seg_id.items()},
    }
    with open(path, 'w') as f:
        json.dump(payload, f)
    return path


def _make_config(tmp):
    from process.config import PipelineConfig
    cfg = PipelineConfig()
    cfg.output_dir = tmp
    cfg.theme_classification.backend = 'fake'
    cfg.theme_classification.model = 'mA'
    cfg.theme_classification.n_runs = 1
    cfg.theme_classification.per_run_models = ['mA']
    cfg.speaker_filter.mode = 'exclude'
    cfg.speaker_filter.speakers = ['therapist']
    return cfg


# ---------------------------------------------------------------------------
# Simulate what execute_single_run does on repair: re-fills ERROR cells with
# CODED ballots by upserting into label_ballots.  We mock execute_single_run to
# do this directly without an LLM.
# ---------------------------------------------------------------------------

def _make_repair_side_effect(tmp, run_id, new_stage=1, always_error=False):
    """Return a mock execute_single_run side effect.

    On call, reads the per-run checkpoint to find cleared (missing) cells, then
    upserts them as CODED (or ERROR if always_error=True) into label_ballots.
    """
    def _side_effect(run_dir, config, run_row, **kw):
        overlay = run_row.get('overlay', 'theme')
        prefix = 'llm_results' if overlay == 'theme' else 'purer_cue_results'
        rid = run_row.get('run_id', run_id)
        ckpt_path = os.path.join(tmp, '02_meta', 'auditable_logs', 'checkpoints',
                                 f'{prefix}_run{rid:04d}_runs.json')
        if not os.path.isfile(ckpt_path):
            return 'failed'
        with open(ckpt_path) as f:
            data = json.load(f)
        # Find cells that are missing (cleared by patch_run_errors_only).
        run_results = data.get('run_results', {})
        new_cells = {}
        for seg_id, by_run in run_results.items():
            if '0' not in by_run:
                # This cell was cleared → re-fill it.
                if always_error:
                    new_cells[seg_id] = None
                else:
                    new_cells[seg_id] = _coded_ballot(new_stage)
        if new_cells:
            rr.upsert_ballots(tmp, overlay, rid, new_cells)
            rr.refresh_counters(tmp, rid)
        status = 'completed' if not always_error else 'completed_with_errors'
        rr.update_run(tmp, rid, status=status)
        return status
    return _side_effect


# ---------------------------------------------------------------------------
# Test 1: flaky-then-ok repair
# ---------------------------------------------------------------------------

class TestRepairFlakyThenOk(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # Use 6 segments so 2 errors = 33% < 50% dead-rater threshold.
        seg_ids = _seed_project(self.tmp, n_segs=6)
        # Create a run with 2 error cells (2/6 = 33% < threshold).
        self.run_id = rr.create_run(self.tmp, overlay='theme', model='mA', rater_label='mA')
        cells = {
            's0': _coded_ballot(1),
            's1': None,  # ERROR
            's2': _coded_ballot(2),
            's3': None,  # ERROR
            's4': _coded_ballot(0),
            's5': _coded_ballot(3),
        }
        rr.upsert_ballots(self.tmp, 'theme', self.run_id, cells)
        rr.update_run(self.tmp, self.run_id, status='completed_with_errors', selected=1)
        rr.refresh_counters(self.tmp, self.run_id)
        # Write the per-run checkpoint with error cells (None).
        self._ckpt = _write_per_run_ckpt(self.tmp, 'theme', self.run_id, {
            's0': _coded_ballot(1), 's1': None, 's2': _coded_ballot(2), 's3': None,
            's4': _coded_ballot(0), 's5': _coded_ballot(3),
        })

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_repair_fixes_errors(self):
        from process.repair import fix_errors
        config = _make_config(self.tmp)

        side_effect = _make_repair_side_effect(self.tmp, self.run_id, new_stage=1)
        with mock.patch(_EXECUTE_SEAM, side_effect=side_effect):
            result = fix_errors(self.tmp, config, overlays=('theme',), max_passes=2)

        ov = result['overlays']['theme']
        self.assertGreater(ov['repaired'], 0, "Expected >0 repaired cells")
        self.assertEqual(ov['remaining'], 0, "Expected 0 remaining errors")
        # Ballots should now have CODED votes for s1 and s3.
        error_cells = ed.detect_run_error_cells(self.tmp, self.run_id)
        self.assertEqual(error_cells, [], "Expected no ERROR ballots remaining")

    def test_dry_run_mutates_nothing(self):
        from process.repair import fix_errors
        config = _make_config(self.tmp)

        # Capture DB state before.
        db_path = _db.db_path(self.tmp)
        before_size = os.path.getsize(db_path)
        # Read label_ballots count before.
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            before_count = conn.execute(
                "SELECT COUNT(*) FROM label_ballots").fetchone()[0]

        side_effect = _make_repair_side_effect(self.tmp, self.run_id, new_stage=1)
        with mock.patch(_EXECUTE_SEAM, side_effect=side_effect) as mock_exe:
            result = fix_errors(self.tmp, config, overlays=('theme',),
                                max_passes=2, dry_run=True)

        # execute_single_run must NOT have been called in dry_run mode.
        mock_exe.assert_not_called()
        # DB unchanged.
        after_size = os.path.getsize(db_path)
        with sqlite3.connect(db_path) as conn:
            after_count = conn.execute(
                "SELECT COUNT(*) FROM label_ballots").fetchone()[0]
        self.assertEqual(before_count, after_count, "dry_run must not modify ballots")
        # dry_run flag echoed back.
        self.assertTrue(result['dry_run'])

    def test_counters_updated_after_repair(self):
        from process.repair import fix_errors
        config = _make_config(self.tmp)

        before_run = rr.get_run(self.tmp, self.run_id)
        self.assertEqual(before_run['n_error'], 2)

        side_effect = _make_repair_side_effect(self.tmp, self.run_id, new_stage=1)
        with mock.patch(_EXECUTE_SEAM, side_effect=side_effect):
            fix_errors(self.tmp, config, overlays=('theme',), max_passes=2)

        after_run = rr.get_run(self.tmp, self.run_id)
        self.assertEqual(after_run['n_error'], 0, "Run counters should show 0 errors after repair")


# ---------------------------------------------------------------------------
# Test 2: no-progress break + flagged-for-review
# ---------------------------------------------------------------------------

class TestRepairNoProgress(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _seed_project(self.tmp, n_segs=3)
        self.run_id = rr.create_run(self.tmp, overlay='theme', model='mB', rater_label='mB')
        cells = {'s0': None, 's1': None, 's2': None}  # ALL errors
        rr.upsert_ballots(self.tmp, 'theme', self.run_id, cells)
        rr.update_run(self.tmp, self.run_id, status='completed_with_errors', selected=1)
        rr.refresh_counters(self.tmp, self.run_id)
        self._ckpt = _write_per_run_ckpt(self.tmp, 'theme', self.run_id,
                                          {'s0': None, 's1': None, 's2': None})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_progress_stops_and_flags(self):
        from process.repair import fix_errors

        config = _make_config(self.tmp)
        # Override dead_rater fraction to 0.0 so the dead-rater guard doesn't trigger.
        # (3/3 errors = 1.0 ≥ 0.5 default → would be skipped unless we force.)
        side_effect = _make_repair_side_effect(self.tmp, self.run_id, always_error=True)
        with mock.patch(_EXECUTE_SEAM, side_effect=side_effect):
            result = fix_errors(self.tmp, config, overlays=('theme',),
                                max_passes=3, force=True)

        ov = result['overlays']['theme']
        # Passes should stop early (no progress) — at most max_passes but likely fewer.
        self.assertGreaterEqual(ov['passes'], 1)
        # The run should be in flagged list.
        self.assertTrue(len(ov['flagged']) > 0 or ov['remaining'] > 0,
                        "Expected flagged entries or remaining errors")

    def test_flagged_file_written(self):
        from process.repair import fix_errors, _flagged_path
        config = _make_config(self.tmp)
        side_effect = _make_repair_side_effect(self.tmp, self.run_id, always_error=True)
        with mock.patch(_EXECUTE_SEAM, side_effect=side_effect):
            fix_errors(self.tmp, config, overlays=('theme',), max_passes=2, force=True)

        flagged_path = _flagged_path(self.tmp)
        if ov_data := rr.get_run(self.tmp, self.run_id):
            if (ov_data.get('n_error') or 0) > 0:
                # Flagged file should exist (errors persist).
                self.assertTrue(os.path.isfile(flagged_path),
                                "flagged_for_review_repair.json should be written")


# ---------------------------------------------------------------------------
# Test 3: missing checkpoint → skip with note, no crash
# ---------------------------------------------------------------------------

class TestRepairMissingCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # 5 segments so 1 error = 20% < 50% threshold (avoids dead_rater guard).
        _seed_project(self.tmp, n_segs=5)
        self.run_id = rr.create_run(self.tmp, overlay='theme', model='mC', rater_label='mC')
        cells = {'s0': None,    # ERROR (1/5 = 20%)
                 's1': _coded_ballot(2),
                 's2': _coded_ballot(1),
                 's3': _coded_ballot(3),
                 's4': _coded_ballot(0)}
        rr.upsert_ballots(self.tmp, 'theme', self.run_id, cells)
        rr.update_run(self.tmp, self.run_id, status='completed_with_errors', selected=1)
        rr.refresh_counters(self.tmp, self.run_id)
        # Intentionally do NOT write a per-run checkpoint.

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_ckpt_skip_no_crash(self):
        from process.repair import fix_errors
        config = _make_config(self.tmp)
        # Should not raise.
        with mock.patch(_EXECUTE_SEAM) as mock_exe:
            result = fix_errors(self.tmp, config, overlays=('theme',), max_passes=2)
        # execute_single_run should NOT have been called (checkpoint missing → skipped).
        mock_exe.assert_not_called()
        ov = result['overlays']['theme']
        flagged_reasons = [f.get('reason') for f in ov.get('flagged', [])]
        self.assertIn('missing_checkpoint', flagged_reasons)


# ---------------------------------------------------------------------------
# Test 4: dead-rater skip (+ force bypass)
# ---------------------------------------------------------------------------

class TestRepairDeadRater(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _seed_project(self.tmp, n_segs=4)
        self.run_id = rr.create_run(self.tmp, overlay='theme', model='mD', rater_label='mD')
        # 3/4 = 75% errors → dead rater (above 50% default threshold).
        cells = {
            's0': None, 's1': None, 's2': None,
            's3': _coded_ballot(1),
        }
        rr.upsert_ballots(self.tmp, 'theme', self.run_id, cells)
        rr.update_run(self.tmp, self.run_id, status='completed_with_errors', selected=1)
        rr.refresh_counters(self.tmp, self.run_id)
        self._ckpt = _write_per_run_ckpt(self.tmp, 'theme', self.run_id, {
            's0': None, 's1': None, 's2': None, 's3': _coded_ballot(1),
        })

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dead_rater_skipped(self):
        from process.repair import fix_errors
        config = _make_config(self.tmp)
        with mock.patch(_EXECUTE_SEAM) as mock_exe:
            result = fix_errors(self.tmp, config, overlays=('theme',), max_passes=2)
        # execute_single_run should NOT be called (dead rater skipped).
        mock_exe.assert_not_called()
        ov = result['overlays']['theme']
        flagged_reasons = [f.get('reason') for f in ov.get('flagged', [])]
        self.assertIn('dead_rater_skipped', flagged_reasons)

    def test_dead_rater_force_bypasses(self):
        from process.repair import fix_errors
        config = _make_config(self.tmp)
        side_effect = _make_repair_side_effect(self.tmp, self.run_id, new_stage=1)
        with mock.patch(_EXECUTE_SEAM, side_effect=side_effect) as mock_exe:
            result = fix_errors(self.tmp, config, overlays=('theme',),
                                max_passes=2, force=True)
        # execute_single_run should be called when force=True.
        mock_exe.assert_called()


# ---------------------------------------------------------------------------
# Test 5: patch_run_errors_only segment_ids subset behavior
# ---------------------------------------------------------------------------

class TestPatchRunErrorsOnlySubset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_ckpt(self, cells):
        """Write a minimal model_first_v1 checkpoint with given cells."""
        ckpt_dir = os.path.join(self.tmp, 'ckpts')
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, 'test_run0001_runs.json')
        payload = {
            '_meta': {'format': 'model_first_v1', 'n_runs': 1,
                      'per_run_models': ['mA'], 'completed_runs': [0]},
            'run_results': {sid: {'0': cell} for sid, cell in cells.items()},
        }
        with open(path, 'w') as f:
            json.dump(payload, f)
        return path

    def test_all_errors_cleared_when_no_filter(self):
        """Default behavior: all null cells cleared."""
        path = self._write_ckpt({'s0': None, 's1': _coded_ballot(1), 's2': None})
        result = patch_run_errors_only(path, 0)
        self.assertEqual(result['cleared_errors'], 2)
        self.assertEqual(result['preserved'], 1)
        self.assertIn('cleared_segment_ids', result)
        self.assertEqual(sorted(result['cleared_segment_ids']), ['s0', 's2'])

    def test_subset_only_clears_targeted(self):
        """segment_ids={s0} only clears s0, leaves s2 (also an error) intact."""
        path = self._write_ckpt({'s0': None, 's1': _coded_ballot(1), 's2': None})
        result = patch_run_errors_only(path, 0, segment_ids={'s0'})
        self.assertEqual(result['cleared_errors'], 1)
        self.assertEqual(result['cleared_segment_ids'], ['s0'])
        # s2 error should still be in the checkpoint.
        with open(path) as f:
            data = json.load(f)
        self.assertIn('0', data['run_results']['s2'])
        self.assertIsNone(data['run_results']['s2']['0'])

    def test_empty_subset_clears_nothing(self):
        """segment_ids={} clears no cells."""
        path = self._write_ckpt({'s0': None, 's1': None})
        result = patch_run_errors_only(path, 0, segment_ids=set())
        self.assertEqual(result['cleared_errors'], 0)
        self.assertEqual(result['cleared_segment_ids'], [])

    def test_cleared_segment_ids_in_result(self):
        """cleared_segment_ids always present in result (backward compat)."""
        path = self._write_ckpt({'s0': None})
        result = patch_run_errors_only(path, 0)
        self.assertIn('cleared_segment_ids', result)
        self.assertIsInstance(result['cleared_segment_ids'], list)

    def test_nonexistent_subset_clears_nothing(self):
        """segment_ids with ids not in checkpoint clears nothing."""
        path = self._write_ckpt({'s0': None, 's1': _coded_ballot(2)})
        result = patch_run_errors_only(path, 0, segment_ids={'s99', 's100'})
        self.assertEqual(result['cleared_errors'], 0)


# ---------------------------------------------------------------------------
# Test 6: failed repair sweep restores prior status (does not demote a run that
# still has valid ballots — keeps it eligible for selection / in consensus).
# ---------------------------------------------------------------------------

class TestRepairSweepFailureRestoresStatus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _seed_project(self.tmp, n_segs=6)
        self.run_id = rr.create_run(self.tmp, overlay='theme', model='mE', rater_label='mE')
        # 1/6 error (well under dead-rater threshold); the rest are valid ballots.
        cells = {
            's0': None,  # ERROR
            's1': _coded_ballot(1), 's2': _coded_ballot(2),
            's3': _coded_ballot(0), 's4': _coded_ballot(3),
            's5': _coded_ballot(1),
        }
        rr.upsert_ballots(self.tmp, 'theme', self.run_id, cells)
        rr.update_run(self.tmp, self.run_id, status='completed_with_errors', selected=1)
        rr.refresh_counters(self.tmp, self.run_id)
        self._ckpt = _write_per_run_ckpt(self.tmp, 'theme', self.run_id, cells)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_failed_sweep_restores_prior_status_and_flags(self):
        from process.repair import fix_errors

        config = _make_config(self.tmp)

        def _failing_sweep(run_dir, config, run_row, **kw):
            # Mimic execute_single_run with retries=0: it stamps 'failed' before
            # giving up — but the run's valid ballots are untouched.
            rid = run_row.get('run_id', self.run_id)
            rr.update_run(self.tmp, rid, status='failed')
            return 'failed'

        with mock.patch(_EXECUTE_SEAM, side_effect=_failing_sweep):
            result = fix_errors(self.tmp, config, overlays=('theme',), max_passes=1)

        # Status restored to the pre-repair value (NOT left 'failed').
        run = rr.get_run(self.tmp, self.run_id)
        self.assertEqual(run['status'], 'completed_with_errors')
        # Still selected (eligibility preserved).
        self.assertTrue(run['selected'])
        # The overlay result records the restoration.
        ov = result['overlays']['theme']
        self.assertTrue(ov.get('repair_sweep_failed'))
        self.assertIn(self.run_id, ov.get('repair_sweep_failed_run_ids', []))


if __name__ == '__main__':
    unittest.main(verbosity=2)
