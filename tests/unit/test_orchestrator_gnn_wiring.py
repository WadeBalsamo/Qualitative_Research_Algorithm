"""
tests/unit/test_orchestrator_gnn_wiring.py
------------------------------------------
Structural / wiring tests for the cheap-scaler (probe + demoted GNN) integration in
process/orchestrator.py.

Tests (without running a real pipeline):

1. stage_assemble forwards probe_ready / gnn_ready into assemble_master_dataset
   — resolved from the persisted gate verdicts (probe_gate.json / gnn_gate.json), NOT a
     'gnn_authoritative' config flag (that override is retired).
2. With no gate verdicts on disk, both flags are False (no un-gated fill).
3. stage_assemble loads frozen segments with apply including 'gnn' AND 'probe'.
4. assemble_master_dataset exposes probe_ready / gnn_ready params (default False) and NO
   gnn_authoritative param.

Heavy callees are monkeypatched so the test is instant and needs no on-disk data.
"""
import os
import sys
import shutil
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from classification_tools.data_structures import Segment


def _make_segment(segment_id='seg_001', session_id='c1s1', speaker='participant'):
    return Segment(segment_id=segment_id, trial_id='trial_A', participant_id='participant_1',
                   session_id=session_id, session_number=1, cohort_id=1, segment_index=0,
                   start_time_ms=0, end_time_ms=5000, total_segments_in_session=1,
                   speaker=speaker, text='Test segment text.', word_count=3)


def _config(run_dir):
    from process.config import PipelineConfig
    config = PipelineConfig()
    config.output_dir = run_dir
    return config


def _fake_assemble_factory(captured):
    def fake_assemble(segs, output_path, confidence_tiers=None,
                      probe_ready=False, gnn_ready=False):
        captured['probe_ready'] = probe_ready
        captured['gnn_ready'] = gnn_ready
        captured['output_path'] = output_path
        return pd.DataFrame({'segment_id': ['seg_001']})
    return fake_assemble


class TestStageAssembleForwardsGateFlags(unittest.TestCase):
    """stage_assemble must forward probe_ready / gnn_ready, resolved from the gates."""

    def setUp(self):
        self.run_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.run_dir, '02_meta', 'training_data'), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def _call(self, probe_ready=False, gnn_ready=False):
        from process.orchestrator import stage_assemble
        captured = {}
        with patch('process.orchestrator.assemble_master_dataset',
                   side_effect=_fake_assemble_factory(captured)), \
             patch('process.orchestrator._probe_promotion_flag', return_value=probe_ready), \
             patch('process.orchestrator._gnn_promotion_flag', return_value=gnn_ready):
            stage_assemble(_config(self.run_dir), segments=[_make_segment()],
                           output_dir=self.run_dir)
        return captured

    def test_no_gates_means_both_false(self):
        # Real flag resolvers, no gate files on disk → both False (no un-gated fill).
        from process.orchestrator import stage_assemble
        captured = {}
        with patch('process.orchestrator.assemble_master_dataset',
                   side_effect=_fake_assemble_factory(captured)):
            stage_assemble(_config(self.run_dir), segments=[_make_segment()],
                           output_dir=self.run_dir)
        self.assertFalse(captured['probe_ready'])
        self.assertFalse(captured['gnn_ready'])

    def test_probe_ready_forwarded(self):
        captured = self._call(probe_ready=True)
        self.assertTrue(captured['probe_ready'])
        self.assertFalse(captured['gnn_ready'])

    def test_gnn_ready_forwarded(self):
        captured = self._call(gnn_ready=True)
        self.assertTrue(captured['gnn_ready'])
        self.assertFalse(captured['probe_ready'])

    def test_explicit_probe_ready_override_bypasses_gate(self):
        # An upstream caller (e.g. `qra probe classify` / --force) may force promotion of
        # the batch's fills even when the standing gate resolver says no.
        from process.orchestrator import stage_assemble
        captured = {}
        with patch('process.orchestrator.assemble_master_dataset',
                   side_effect=_fake_assemble_factory(captured)), \
             patch('process.orchestrator._probe_promotion_flag', return_value=False):
            stage_assemble(_config(self.run_dir), segments=[_make_segment()],
                           output_dir=self.run_dir, probe_ready=True)
        self.assertTrue(captured['probe_ready'])


class TestProbePromotionFlag(unittest.TestCase):
    """_probe_promotion_flag is gate-based (symmetric with _gnn_promotion_flag): the persisted
    reliability gate authorizes promotion, NOT config.probe.enabled (which only governs the
    full-pipeline auto-run)."""

    def setUp(self):
        self.run_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def test_promotes_when_gate_ready_even_if_disabled(self):
        from process import orchestrator as orch
        cfg = _config(self.run_dir)
        cfg.probe.enabled = False
        with patch('classification_tools.probe_classifier.probe_gate_ready', return_value=True):
            self.assertTrue(orch._probe_promotion_flag(cfg, self.run_dir))

    def test_no_promotion_without_gate_even_if_enabled(self):
        from process import orchestrator as orch
        cfg = _config(self.run_dir)
        cfg.probe.enabled = True
        with patch('classification_tools.probe_classifier.probe_gate_ready', return_value=False):
            self.assertFalse(orch._probe_promotion_flag(cfg, self.run_dir))


class TestStageAssembleApplyIncludesCheapTiers(unittest.TestCase):
    """The disk-load apply tuple must include the standard overlays + 'gnn' + 'probe'."""

    def setUp(self):
        self.run_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.run_dir, '02_meta', 'training_data'), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def _captured_apply(self):
        from process.orchestrator import stage_assemble
        captured = {}

        def fake_load(run_dir, apply=()):
            captured['apply'] = apply
            return [_make_segment()]

        with patch('process.orchestrator.segments_io.load_segments_for_stage',
                   side_effect=fake_load), \
             patch('process.orchestrator.assemble_master_dataset',
                   side_effect=_fake_assemble_factory({})):
            stage_assemble(_config(self.run_dir), segments=None, output_dir=self.run_dir)
        return captured.get('apply', ())

    def test_apply_tuple_contains_all_overlays(self):
        apply = self._captured_apply()
        for key in ('theme', 'purer', 'codebook', 'cv', 'gnn', 'probe'):
            self.assertIn(key, apply, f"overlay key {key!r} missing from apply={apply!r}")


class TestAssembleSignature(unittest.TestCase):
    """assemble_master_dataset exposes the new gate-flag params and not the retired one."""

    def test_probe_and_gnn_ready_params(self):
        import inspect
        from process.assembly.master_dataset import assemble_master_dataset
        sig = inspect.signature(assemble_master_dataset)
        self.assertIn('probe_ready', sig.parameters)
        self.assertIn('gnn_ready', sig.parameters)
        self.assertFalse(sig.parameters['probe_ready'].default)
        self.assertFalse(sig.parameters['gnn_ready'].default)

    def test_gnn_authoritative_param_removed(self):
        import inspect
        from process.assembly.master_dataset import assemble_master_dataset
        sig = inspect.signature(assemble_master_dataset)
        self.assertNotIn('gnn_authoritative', sig.parameters)
        self.assertNotIn('gate_passed', sig.parameters)


class TestStageAssembleImportable(unittest.TestCase):
    def test_importable_and_signature(self):
        import inspect
        from process.orchestrator import stage_assemble
        sig = inspect.signature(stage_assemble)
        self.assertIn('output_dir', sig.parameters)
        self.assertIn('segments', sig.parameters)


if __name__ == '__main__':
    unittest.main()
