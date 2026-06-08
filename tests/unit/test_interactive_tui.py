"""
tests/unit/test_interactive_tui.py
-----------------------------------
Pure-state helper tests for process/interactive_tui.py.

Covers:
  - _gnn_status:  'ready', 'not_ready', 'trained', 'absent' branches
  - _gnn_tag:     format of each status label
  - _detect_state: returns a dict that includes the 'gnn_status' key, driven
                   by the same files that _gnn_status probes

Does NOT launch any interactive loop or require a real terminal.
"""
import os
import sys
import shutil
import tempfile
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from process.interactive_tui import _gnn_status, _gnn_tag, _probe_status, _probe_tag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reports_gnn_dir(run_dir: str) -> str:
    p = os.path.join(run_dir, '06_reports', '06_gnn')
    os.makedirs(p, exist_ok=True)
    return p


def _make_gnn_model_dir(run_dir: str) -> str:
    p = os.path.join(run_dir, '02_meta', 'gnn')
    os.makedirs(p, exist_ok=True)
    return p


def _write_validation_report(gnn_dir: str, has_yes: bool) -> str:
    path = os.path.join(gnn_dir, 'validation.txt')
    line = 'LLM-FREE SCALING? YES' if has_yes else 'LLM-FREE SCALING? NO'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'{line}\nSome other content.\n')
    return path


def _write_weights(gnn_model_dir: str) -> str:
    path = os.path.join(gnn_model_dir, 'weights.pt')
    with open(path, 'wb') as f:
        f.write(b'\x00' * 4)
    return path


# ---------------------------------------------------------------------------
# _gnn_status tests
# ---------------------------------------------------------------------------

class TestGnnStatus(unittest.TestCase):

    def setUp(self):
        self.run_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    # --- absent ---

    def test_absent_when_no_files(self):
        """No weights, no validation report → 'absent'."""
        self.assertEqual(_gnn_status(self.run_dir), 'absent')

    # --- trained ---

    def test_trained_when_weights_exist_no_report(self):
        """Weights checkpoint present but no validation.txt → 'trained'."""
        gnn_dir = _make_gnn_model_dir(self.run_dir)
        _write_weights(gnn_dir)
        self.assertEqual(_gnn_status(self.run_dir), 'trained')

    def test_trained_when_validation_report_present_but_no_scaling_line(self):
        """validation.txt exists but no LLM-FREE SCALING line → 'trained'."""
        gnn_dir = _make_reports_gnn_dir(self.run_dir)
        path = os.path.join(gnn_dir, 'validation.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('This report has no scaling decision yet.\n')
        self.assertEqual(_gnn_status(self.run_dir), 'trained')

    # --- ready ---

    def test_ready_when_validation_report_says_yes(self):
        """validation.txt contains 'LLM-FREE SCALING? YES' → 'ready'."""
        gnn_dir = _make_reports_gnn_dir(self.run_dir)
        _write_validation_report(gnn_dir, has_yes=True)
        self.assertEqual(_gnn_status(self.run_dir), 'ready')

    # --- not_ready ---

    def test_not_ready_when_validation_report_says_no(self):
        """validation.txt contains 'LLM-FREE SCALING? NO' → 'not_ready'."""
        gnn_dir = _make_reports_gnn_dir(self.run_dir)
        _write_validation_report(gnn_dir, has_yes=False)
        self.assertEqual(_gnn_status(self.run_dir), 'not_ready')

    # --- report takes precedence over weights ---

    def test_ready_even_when_weights_missing(self):
        """If validation.txt says YES, weights presence is irrelevant."""
        gnn_dir = _make_reports_gnn_dir(self.run_dir)
        _write_validation_report(gnn_dir, has_yes=True)
        # No weights file
        self.assertEqual(_gnn_status(self.run_dir), 'ready')

    def test_not_ready_overrides_weights(self):
        """If validation.txt says NO, return not_ready even if weights exist."""
        gnn_dir = _make_reports_gnn_dir(self.run_dir)
        _write_validation_report(gnn_dir, has_yes=False)
        model_dir = _make_gnn_model_dir(self.run_dir)
        _write_weights(model_dir)
        self.assertEqual(_gnn_status(self.run_dir), 'not_ready')


# ---------------------------------------------------------------------------
# _gnn_tag tests
# ---------------------------------------------------------------------------

class TestGnnTag(unittest.TestCase):

    def test_ready_tag_contains_recommended(self):
        tag = _gnn_tag('ready')
        self.assertIn('ready', tag.lower())

    def test_not_ready_tag_not_empty(self):
        tag = _gnn_tag('not_ready')
        self.assertTrue(tag.strip())

    def test_trained_tag_not_empty(self):
        tag = _gnn_tag('trained')
        self.assertTrue(tag.strip())

    def test_absent_tag_not_empty(self):
        tag = _gnn_tag('absent')
        self.assertTrue(tag.strip())

    def test_unknown_status_returns_empty_string(self):
        tag = _gnn_tag('completely_unknown_status_xyz')
        self.assertEqual(tag, '')

    def test_all_four_statuses_are_distinct(self):
        tags = [_gnn_tag(s) for s in ('ready', 'not_ready', 'trained', 'absent')]
        self.assertEqual(len(set(tags)), 4, "Each GNN status should produce a unique tag")


# ---------------------------------------------------------------------------
# _detect_state (gnn_status key)
# ---------------------------------------------------------------------------

class TestDetectStateGnnKey(unittest.TestCase):
    """
    _detect_state imports several process/ modules and scans the filesystem.
    We test only the 'gnn_status' key to avoid needing a complete project tree.
    The key must be present regardless of what other state keys contain.
    """

    def setUp(self):
        self.run_dir = tempfile.mkdtemp()
        # Create the minimum directory skeleton that _detect_state touches:
        # it calls output_paths.* functions which all do os.path.* checks.
        # Many dirs just need to not raise; they need not be populated.
        for sub in (
            '01_transcripts/segmented',
            '02_meta/classifications',
            '02_meta/gnn',
            '03_analysis_data',
            '04_validation/testsets',
            '04_validation/content_validity',
            '05_figures',
            '06_reports/06_gnn',
        ):
            os.makedirs(os.path.join(self.run_dir, sub), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def test_detect_state_has_gnn_status_key(self):
        from process.interactive_tui import _detect_state
        state = _detect_state(self.run_dir)
        self.assertIn('gnn_status', state)

    def test_detect_state_gnn_status_absent_by_default(self):
        from process.interactive_tui import _detect_state
        state = _detect_state(self.run_dir)
        self.assertEqual(state['gnn_status'], 'absent')

    def test_detect_state_gnn_status_ready_when_report_yes(self):
        gnn_dir = os.path.join(self.run_dir, '06_reports', '06_gnn')
        path = os.path.join(gnn_dir, 'validation.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('LLM-FREE SCALING? YES\n')

        from process.interactive_tui import _detect_state
        state = _detect_state(self.run_dir)
        self.assertEqual(state['gnn_status'], 'ready')

    def test_detect_state_gnn_status_trained_when_weights_only(self):
        weights_dir = os.path.join(self.run_dir, '02_meta', 'gnn')
        with open(os.path.join(weights_dir, 'weights.pt'), 'wb') as f:
            f.write(b'\x00' * 4)

        from process.interactive_tui import _detect_state
        state = _detect_state(self.run_dir)
        self.assertEqual(state['gnn_status'], 'trained')

    def test_detect_state_returns_required_keys(self):
        """Smoke-test that _detect_state returns all documented keys."""
        from process.interactive_tui import _detect_state
        state = _detect_state(self.run_dir)
        required = {
            'is_legacy', 'has_segments', 'has_theme', 'has_purer',
            'has_codebook', 'has_master', 'has_testsets', 'has_cv_testsets',
            'has_analysis', 'gnn_status', 'config_path',
        }
        for key in required:
            self.assertIn(key, state, f"Missing key: {key}")

    def test_detect_state_has_probe_status_key(self):
        from process.interactive_tui import _detect_state
        state = _detect_state(self.run_dir)
        self.assertIn('probe_status', state)
        self.assertEqual(state['probe_status'], 'absent')


# ---------------------------------------------------------------------------
# _probe_status / _probe_tag tests
# ---------------------------------------------------------------------------

def _write_probe_gate(run_dir: str, ready: bool) -> str:
    import json
    d = os.path.join(run_dir, '03_analysis_data', 'probe')
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, 'probe_gate.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'ready_for_scaling': ready, 'probe_human_kappa': 0.45}, f)
    return path


class TestProbeStatus(unittest.TestCase):
    def setUp(self):
        self.run_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def test_absent_when_no_gate(self):
        self.assertEqual(_probe_status(self.run_dir), 'absent')

    def test_ready_when_gate_says_yes(self):
        _write_probe_gate(self.run_dir, True)
        self.assertEqual(_probe_status(self.run_dir), 'ready')

    def test_not_ready_when_gate_says_no(self):
        _write_probe_gate(self.run_dir, False)
        self.assertEqual(_probe_status(self.run_dir), 'not_ready')


class TestProbeTag(unittest.TestCase):
    def test_ready_tag_recommends(self):
        self.assertIn('recommended', _probe_tag('ready'))

    def test_statuses_distinct_and_unknown_empty(self):
        tags = [_probe_tag(s) for s in ('ready', 'not_ready', 'absent')]
        self.assertEqual(len(set(tags)), 3)
        self.assertEqual(_probe_tag('xyz'), '')


# ---------------------------------------------------------------------------
# Add-data mode-picker recommendation (scale-aware)
# ---------------------------------------------------------------------------

class TestAddDataRecommendation(unittest.TestCase):
    def _rec(self, probe, gnn, n_new):
        from process.interactive_tui import _recommend_add_data_mode
        return _recommend_add_data_mode(probe, gnn, n_new)[0]

    def test_llm_when_no_cheap_gate_ready(self):
        self.assertEqual(self._rec({'ready': False}, {'ready': False}, 99), 'llm')

    def test_llm_for_small_addition_even_if_probe_ready(self):
        self.assertEqual(self._rec({'ready': True, 'human_kappa': 0.45},
                                   {'ready': False}, 2), 'llm')

    def test_probe_for_large_addition_when_ready(self):
        self.assertEqual(self._rec({'ready': True, 'human_kappa': 0.45},
                                   {'ready': False}, 20), 'probe')

    def test_probe_preferred_over_gnn_when_both_ready(self):
        self.assertEqual(self._rec({'ready': True, 'human_kappa': 0.45},
                                   {'ready': True}, 20), 'probe')

    def test_gnn_when_only_gnn_ready_and_large(self):
        self.assertEqual(self._rec({'ready': False}, {'ready': True}, 20), 'gnn')

    def test_unknown_scale_with_ready_probe_recommends_probe(self):
        # n_new is None (couldn't estimate) → don't force small; a ready probe is recommended
        self.assertEqual(self._rec({'ready': True, 'human_kappa': 0.45},
                                   {'ready': False}, None), 'probe')


if __name__ == '__main__':
    unittest.main()
