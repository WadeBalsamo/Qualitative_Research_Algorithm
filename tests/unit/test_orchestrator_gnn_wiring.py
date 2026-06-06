"""
tests/unit/test_orchestrator_gnn_wiring.py
------------------------------------------
Structural / wiring tests for GNN integration in process/orchestrator.py.

Tests (without running a real pipeline):

1. stage_assemble forwards gnn_authoritative into assemble_master_dataset
   — driven by config.gnn_layer.gnn_authoritative (True / False)

2. stage_assemble loads frozen segments with apply=(..., 'gnn', ...)
   — the 'gnn' key must be in the apply tuple so GNN predictions are merged
     before assembly

All heavy callees (assemble_master_dataset, segments_io.load_segments_for_stage)
are monkeypatched so the test is instant and requires no on-disk data.
"""
import os
import sys
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock, call
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from classification_tools.data_structures import Segment


# ---------------------------------------------------------------------------
# Minimal helpers
# ---------------------------------------------------------------------------

def _make_segment(segment_id='seg_001', session_id='c1s1', speaker='participant'):
    return Segment(
        segment_id=segment_id,
        trial_id='trial_A',
        participant_id='participant_1',
        session_id=session_id,
        session_number=1,
        cohort_id=1,
        session_variant='',
        segment_index=0,
        start_time_ms=0,
        end_time_ms=5000,
        total_segments_in_session=1,
        speaker=speaker,
        text='Test segment text.',
        word_count=3,
    )


def _build_config(gnn_authoritative: bool = False, gnn_enabled: bool = True):
    """Build a minimal PipelineConfig with gnn_layer set as requested."""
    from process.config import PipelineConfig
    config = PipelineConfig()
    config.gnn_layer.enabled = gnn_enabled
    config.gnn_layer.gnn_authoritative = gnn_authoritative
    return config


# ---------------------------------------------------------------------------
# Test 1 — stage_assemble forwards gnn_authoritative
# ---------------------------------------------------------------------------

class TestStageAssembleGnnAuthoritative(unittest.TestCase):
    """stage_assemble must pass gnn_authoritative to assemble_master_dataset."""

    def setUp(self):
        self.run_dir = tempfile.mkdtemp()
        # Create the minimal structure that stage_assemble needs to not fail
        # on os.makedirs calls.
        ms_dir = os.path.join(self.run_dir, '02_meta', 'training_data')
        os.makedirs(ms_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def _call_stage_assemble(self, gnn_authoritative: bool, segments=None):
        """
        Invoke stage_assemble with mocked I/O; return the kwargs captured by
        the assemble_master_dataset mock.
        """
        from process.orchestrator import stage_assemble

        if segments is None:
            segments = [_make_segment()]

        fake_df = pd.DataFrame({'segment_id': ['seg_001']})

        captured = {}

        def fake_assemble(segs, output_path, confidence_tiers=None,
                          gnn_authoritative=False, gate_passed=False):
            captured['gnn_authoritative'] = gnn_authoritative
            captured['gate_passed'] = gate_passed
            captured['output_path'] = output_path
            return fake_df

        config = _build_config(gnn_authoritative=gnn_authoritative)
        config.output_dir = self.run_dir

        with patch('process.assembly.master_dataset.assemble_master_dataset',
                   side_effect=fake_assemble), \
             patch('process.orchestrator.assemble_master_dataset',
                   side_effect=fake_assemble):
            stage_assemble(config, segments=segments, output_dir=self.run_dir)

        return captured

    def test_gnn_authoritative_true_forwarded(self):
        captured = self._call_stage_assemble(gnn_authoritative=True)
        self.assertIn('gnn_authoritative', captured)
        self.assertTrue(captured['gnn_authoritative'])

    def test_gate_passed_false_when_no_gate_verdict(self):
        """Track 0.2: with gnn_authoritative=True but no persisted gate verdict on
        disk, stage_assemble must forward gate_passed=False (no un-gated promotion)."""
        captured = self._call_stage_assemble(gnn_authoritative=True)
        self.assertIn('gate_passed', captured)
        self.assertFalse(captured['gate_passed'])

    def test_gnn_authoritative_false_forwarded(self):
        captured = self._call_stage_assemble(gnn_authoritative=False)
        self.assertIn('gnn_authoritative', captured)
        self.assertFalse(captured['gnn_authoritative'])

    def test_gnn_authoritative_default_is_false(self):
        """When gnn_layer is absent from config, gnn_authoritative defaults to False."""
        from process.orchestrator import stage_assemble
        from process.config import PipelineConfig

        config = PipelineConfig()
        # Do not set gnn_layer at all — rely on the getattr default chain
        config.output_dir = self.run_dir

        segments = [_make_segment()]
        fake_df = pd.DataFrame({'segment_id': ['seg_001']})
        captured = {}

        def fake_assemble(segs, output_path, confidence_tiers=None,
                          gnn_authoritative=False, gate_passed=False):
            captured['gnn_authoritative'] = gnn_authoritative
            return fake_df

        with patch('process.orchestrator.assemble_master_dataset',
                   side_effect=fake_assemble):
            stage_assemble(config, segments=segments, output_dir=self.run_dir)

        self.assertFalse(captured.get('gnn_authoritative', True))


# ---------------------------------------------------------------------------
# Test 2 — stage_assemble loads segments with 'gnn' in the apply tuple
# ---------------------------------------------------------------------------

class TestStageAssembleApplyIncludesGnn(unittest.TestCase):
    """
    When stage_assemble loads segments from disk (segments=None), it must
    request the 'gnn' overlay via the apply= parameter of
    load_segments_for_stage.
    """

    def setUp(self):
        self.run_dir = tempfile.mkdtemp()
        ms_dir = os.path.join(self.run_dir, '02_meta', 'training_data')
        os.makedirs(ms_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def test_gnn_key_in_apply_tuple(self):
        from process.orchestrator import stage_assemble

        config = _build_config()
        config.output_dir = self.run_dir

        segments = [_make_segment()]
        fake_df = pd.DataFrame({'segment_id': ['seg_001']})
        captured_apply = {}

        def fake_load(run_dir, apply=()):
            captured_apply['apply'] = apply
            return segments

        def fake_assemble(segs, output_path, confidence_tiers=None,
                          gnn_authoritative=False, gate_passed=False):
            return fake_df

        with patch('process.orchestrator.segments_io.load_segments_for_stage',
                   side_effect=fake_load), \
             patch('process.orchestrator.assemble_master_dataset',
                   side_effect=fake_assemble):
            # segments=None triggers the disk-load path
            stage_assemble(config, segments=None, output_dir=self.run_dir)

        self.assertIn('apply', captured_apply,
                      "load_segments_for_stage was not called with an apply kwarg")
        self.assertIn('gnn', captured_apply['apply'],
                      f"'gnn' not in apply tuple: {captured_apply['apply']!r}")

    def test_apply_tuple_also_contains_standard_overlays(self):
        """The apply tuple must include 'theme', 'purer', 'codebook', 'cv' as well."""
        from process.orchestrator import stage_assemble

        config = _build_config()
        config.output_dir = self.run_dir

        segments = [_make_segment()]
        fake_df = pd.DataFrame({'segment_id': ['seg_001']})
        captured_apply = {}

        def fake_load(run_dir, apply=()):
            captured_apply['apply'] = apply
            return segments

        def fake_assemble(segs, output_path, confidence_tiers=None,
                          gnn_authoritative=False, gate_passed=False):
            return fake_df

        with patch('process.orchestrator.segments_io.load_segments_for_stage',
                   side_effect=fake_load), \
             patch('process.orchestrator.assemble_master_dataset',
                   side_effect=fake_assemble):
            stage_assemble(config, segments=None, output_dir=self.run_dir)

        apply = captured_apply.get('apply', ())
        for expected_key in ('theme', 'purer', 'codebook', 'cv', 'gnn'):
            self.assertIn(expected_key, apply,
                          f"Expected overlay key '{expected_key}' missing from apply={apply!r}")


# ---------------------------------------------------------------------------
# Test 3 — stage_assemble signature contract
# ---------------------------------------------------------------------------

class TestStageAssembleImportable(unittest.TestCase):
    """stage_assemble must be importable and have the right signature."""

    def test_importable(self):
        from process.orchestrator import stage_assemble  # noqa

    def test_accepts_output_dir_kwarg(self):
        """stage_assemble must accept an output_dir keyword argument."""
        import inspect
        from process.orchestrator import stage_assemble
        sig = inspect.signature(stage_assemble)
        self.assertIn('output_dir', sig.parameters)

    def test_accepts_segments_kwarg(self):
        import inspect
        from process.orchestrator import stage_assemble
        sig = inspect.signature(stage_assemble)
        self.assertIn('segments', sig.parameters)


# ---------------------------------------------------------------------------
# Test 4 — assemble_master_dataset itself accepts gnn_authoritative param
# ---------------------------------------------------------------------------

class TestAssembleMasterDatasetGnnParam(unittest.TestCase):
    """assemble_master_dataset must accept a gnn_authoritative kwarg."""

    def test_gnn_authoritative_param_exists(self):
        import inspect
        from process.assembly.master_dataset import assemble_master_dataset
        sig = inspect.signature(assemble_master_dataset)
        self.assertIn('gnn_authoritative', sig.parameters)

    def test_gnn_authoritative_default_is_false(self):
        import inspect
        from process.assembly.master_dataset import assemble_master_dataset
        sig = inspect.signature(assemble_master_dataset)
        param = sig.parameters['gnn_authoritative']
        self.assertFalse(param.default)

    def test_gate_passed_param_exists_and_defaults_false(self):
        """Track 0.2: gate_passed must exist and default False (safe — no un-gated
        promotion even if a caller forgets to thread the gate verdict through)."""
        import inspect
        from process.assembly.master_dataset import assemble_master_dataset
        sig = inspect.signature(assemble_master_dataset)
        self.assertIn('gate_passed', sig.parameters)
        self.assertFalse(sig.parameters['gate_passed'].default)


if __name__ == '__main__':
    unittest.main()
