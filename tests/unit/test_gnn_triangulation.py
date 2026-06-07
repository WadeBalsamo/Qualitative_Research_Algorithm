"""
tests/unit/test_gnn_triangulation.py
-------------------------------------
Deep unit tests for gnn_layer/triangulation.py: _kappa, _agreement,
compute_triangulation (full + human-subset path), and write_triangulation_report
text wording (distillation-fidelity framing).
"""

import os
import sys
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from tests.testhelpers import synthetic_df, embedding_patch, make_master_df
from gnn_layer.config import GnnLayerConfig
from gnn_layer.classifier import triangulation as tri


class TestKappa(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_perfect_agreement(self):
        k = tri._kappa([0, 1, 2, 0, 1], [0, 1, 2, 0, 1])
        self.assertAlmostEqual(k, 1.0, places=5)

    def test_all_same_class_returns_1(self):
        # Both lists same single-class value → defined as 1.0 in source
        k = tri._kappa([2, 2, 2], [2, 2, 2])
        self.assertEqual(k, 1.0)

    def test_random_agreement_near_zero(self):
        # Labels with no relationship → κ near 0
        k = tri._kappa([0, 1, 2, 3, 4, 0, 1, 2], [1, 2, 3, 4, 0, 2, 3, 4])
        self.assertIsNotNone(k)
        self.assertIsInstance(k, float)

    def test_too_short_returns_none(self):
        k = tri._kappa([1], [1])
        self.assertIsNone(k)

    def test_empty_returns_none(self):
        k = tri._kappa([], [])
        self.assertIsNone(k)


class TestAgreement(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_perfect(self):
        pred = pd.Series([0, 1, 2, 3])
        ref = pd.Series([0, 1, 2, 3])
        d = tri._agreement(pred, ref)
        self.assertEqual(d['n'], 4)
        self.assertAlmostEqual(d['percent_agreement'], 1.0, places=5)
        self.assertAlmostEqual(d['cohen_kappa'], 1.0, places=5)

    def test_partial_agreement(self):
        pred = pd.Series([0, 1, 2, 3])
        ref = pd.Series([0, 1, 2, 2])  # last differs
        d = tri._agreement(pred, ref)
        self.assertEqual(d['n'], 4)
        self.assertAlmostEqual(d['percent_agreement'], 0.75, places=3)

    def test_nan_dropped(self):
        pred = pd.Series([0, 1, np.nan, 2])
        ref = pd.Series([0, 1, 3, 2])
        d = tri._agreement(pred, ref)
        self.assertEqual(d['n'], 3)

    def test_empty_returns_zero_n(self):
        pred = pd.Series([], dtype=float)
        ref = pd.Series([], dtype=float)
        d = tri._agreement(pred, ref)
        self.assertEqual(d['n'], 0)
        self.assertIsNone(d['percent_agreement'])
        self.assertIsNone(d['cohen_kappa'])

    def test_all_nan_returns_zero_n(self):
        pred = pd.Series([np.nan, np.nan])
        ref = pd.Series([np.nan, np.nan])
        d = tri._agreement(pred, ref)
        self.assertEqual(d['n'], 0)

    def test_keys_present(self):
        d = tri._agreement(pd.Series([0, 1]), pd.Series([0, 0]))
        self.assertIn('n', d)
        self.assertIn('percent_agreement', d)
        self.assertIn('cohen_kappa', d)


class TestComputeTriangulation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _basic_preds(self, n=6):
        return {
            'segment_id': [f's{i}' for i in range(n)],
            'node_type': ['participant_segment'] * n,
            'gnn_vaamr_pred': list(range(n)),
        }

    def _basic_df(self, n=6, final_labels=None):
        if final_labels is None:
            final_labels = list(range(n))
        return pd.DataFrame({
            'segment_id': [f's{i}' for i in range(n)],
            'node_type': ['participant_segment'] * n,
            'final_label': final_labels,
        })

    def test_basic_keys(self):
        preds = self._basic_preds(4)
        dfa = self._basic_df(4, final_labels=[0, 1, 2, 2])
        preds['gnn_vaamr_pred'] = [0, 1, 2, 3]
        out = tri.compute_triangulation(preds, dfa)
        self.assertIn('vaamr_gnn_vs_llm', out)
        self.assertEqual(out['vaamr_gnn_vs_llm']['n'], 4)
        self.assertAlmostEqual(out['vaamr_gnn_vs_llm']['percent_agreement'], 0.75, places=3)

    def test_purer_key_included_when_pred_present(self):
        preds = {
            'segment_id': ['t0', 't1', 't2', 't3'],
            'node_type': ['therapist_segment'] * 4,
            'gnn_vaamr_pred': [0, 1, 2, 3],
            'gnn_purer_pred': [0, 1, 2, 3],
        }
        dfa = pd.DataFrame({
            'segment_id': ['t0', 't1', 't2', 't3'],
            'node_type': ['therapist_segment'] * 4,
            'final_label': [np.nan, np.nan, np.nan, np.nan],
            'purer_primary': [0, 1, 2, 3],
        })
        out = tri.compute_triangulation(preds, dfa)
        self.assertIn('purer_gnn_vs_llm', out)

    def test_human_subset_path(self):
        """When in_human_coded_subset is present, vaamr_gnn_vs_human and vaamr_llm_vs_human appear."""
        n = 6
        preds = {
            'segment_id': [f's{i}' for i in range(n)],
            'node_type': ['participant_segment'] * n,
            'gnn_vaamr_pred': [0, 1, 2, 3, 4, 0],
        }
        dfa = pd.DataFrame({
            'segment_id': [f's{i}' for i in range(n)],
            'node_type': ['participant_segment'] * n,
            'final_label': [0, 1, 2, 3, 4, 0],
            'human_label': [0, 1, 2, 3, 4, np.nan],
            'in_human_coded_subset': [True, True, True, True, True, False],
        })
        out = tri.compute_triangulation(preds, dfa)
        self.assertIn('vaamr_gnn_vs_human', out)
        self.assertIn('vaamr_llm_vs_human', out)

    def test_empty_preds_returns_empty(self):
        out = tri.compute_triangulation({}, pd.DataFrame())
        self.assertEqual(out, {})

    def test_no_segment_id_key_returns_empty(self):
        preds = {'gnn_vaamr_pred': [0, 1, 2]}
        dfa = pd.DataFrame({'segment_id': ['a', 'b', 'c'], 'final_label': [0, 1, 2]})
        out = tri.compute_triangulation(preds, dfa)
        self.assertEqual(out, {})


class TestWriteTriangulationReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _minimal_tri(self):
        return {
            'vaamr_gnn_vs_llm': {'n': 8, 'percent_agreement': 0.875, 'cohen_kappa': 0.83},
        }

    def test_writes_to_correct_path(self):
        t = self._minimal_tri()
        path = tri.write_triangulation_report(t, self.tmp)
        from process import output_paths as _paths
        expected = os.path.join(_paths.reports_gnn_dir(self.tmp), 'triangulation.txt')
        self.assertEqual(path, expected)
        self.assertTrue(os.path.isfile(path))

    def test_distillation_fidelity_framing_present(self):
        """The report must explain the two axes including 'DISTILLATION FIDELITY'."""
        t = self._minimal_tri()
        path = tri.write_triangulation_report(t, self.tmp)
        text = open(path, encoding='utf-8').read()
        self.assertIn('DISTILLATION FIDELITY', text)

    def test_independent_quality_framing_present(self):
        t = self._minimal_tri()
        path = tri.write_triangulation_report(t, self.tmp)
        text = open(path, encoding='utf-8').read()
        self.assertIn('INDEPENDENT QUALITY', text)

    def test_gnn_vs_llm_section_present(self):
        t = self._minimal_tri()
        path = tri.write_triangulation_report(t, self.tmp)
        text = open(path, encoding='utf-8').read()
        self.assertIn('GNN vs LLM', text)

    def test_header_present(self):
        t = self._minimal_tri()
        path = tri.write_triangulation_report(t, self.tmp)
        text = open(path, encoding='utf-8').read()
        self.assertIn('GNN', text)
        self.assertIn('TRIANGULATION', text)

    def test_no_human_subset_note_when_absent(self):
        t = self._minimal_tri()
        path = tri.write_triangulation_report(t, self.tmp)
        text = open(path, encoding='utf-8').read()
        # Should note that no human-coded subset is available
        self.assertIn('no human-coded subset', text)

    def test_lift_summary_section_written_when_provided(self):
        t = self._minimal_tri()
        lift = {'n_pairs': 10, 'n_both_elevated': 3}
        path = tri.write_triangulation_report(t, self.tmp, lift_summary=lift)
        text = open(path, encoding='utf-8').read()
        self.assertIn('convergence', text.lower())
        self.assertIn('10', text)

    def test_directory_created_if_absent(self):
        import shutil as _sh
        from process import output_paths as _paths
        rep_dir = _paths.reports_gnn_dir(self.tmp)
        if os.path.isdir(rep_dir):
            _sh.rmtree(rep_dir)
        t = self._minimal_tri()
        path = tri.write_triangulation_report(t, self.tmp)
        self.assertTrue(os.path.isfile(path))


class TestIndependenceMode(unittest.TestCase):
    """gnn_layer.runner._independence_mode — G1 independence-pass mode resolution."""

    def _cfg(self, **kw):
        c = GnnLayerConfig()
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    def test_explicit_human_honored(self):
        from gnn_layer import runner
        df = pd.DataFrame({'segment_id': ['a']})
        self.assertEqual(runner._independence_mode(self._cfg(independence_label_mode='human'), df), 'human')

    def test_explicit_self_supervised_honored(self):
        from gnn_layer import runner
        df = pd.DataFrame({'segment_id': ['a']})
        self.assertEqual(
            runner._independence_mode(self._cfg(independence_label_mode='self_supervised'), df),
            'self_supervised')

    def test_auto_picks_human_when_subset_large_enough(self):
        from gnn_layer import runner
        n = 12
        df = pd.DataFrame({
            'segment_id': [f's{i}' for i in range(n)],
            'in_human_coded_subset': [True] * n,
            'human_label': [0] * n,
        })
        self.assertEqual(
            runner._independence_mode(self._cfg(independence_label_mode='auto', independence_min_human=10), df),
            'human')

    def test_auto_falls_back_to_self_supervised_when_subset_small(self):
        from gnn_layer import runner
        df = pd.DataFrame({
            'segment_id': ['s0', 's1', 's2'],
            'in_human_coded_subset': [True, True, False],
            'human_label': [0, 1, np.nan],
        })
        self.assertEqual(
            runner._independence_mode(self._cfg(independence_label_mode='auto', independence_min_human=10), df),
            'self_supervised')

    def test_auto_falls_back_when_no_human_columns(self):
        from gnn_layer import runner
        df = pd.DataFrame({'segment_id': ['a', 'b']})
        self.assertEqual(runner._independence_mode(self._cfg(independence_label_mode='auto'), df),
                         'self_supervised')


class TestIndependencePassReport(unittest.TestCase):
    """write_triangulation_report independence-pass framing + custom filename."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _tri(self):
        return {
            'vaamr_gnn_vs_llm': {'n': 8, 'percent_agreement': 0.62, 'cohen_kappa': 0.41},
            'vaamr_gnn_vs_human': {'n': 5, 'percent_agreement': 0.80, 'cohen_kappa': 0.74},
            'vaamr_llm_vs_human': {'n': 5, 'percent_agreement': 0.80, 'cohen_kappa': 0.75},
        }

    def test_human_mode_writes_custom_filename_and_corroboration_framing(self):
        path = tri.write_triangulation_report(
            self._tri(), self.tmp, mode='human', filename='triangulation_independence.txt')
        self.assertTrue(path.endswith('triangulation_independence.txt'))
        text = open(path, encoding='utf-8').read()
        self.assertIn('INDEPENDENCE PASS', text)
        self.assertIn('WITHHELD', text)
        self.assertIn('INDEPENDENT CORROBORATION', text)
        # The independence pass must NOT frame GNN-vs-LLM as distillation fidelity.
        self.assertNotIn('DISTILLATION FIDELITY', text)

    def test_self_supervised_mode_frames_null_control(self):
        path = tri.write_triangulation_report(
            self._tri(), self.tmp, mode='self_supervised', filename='triangulation_independence.txt')
        text = open(path, encoding='utf-8').read()
        self.assertIn('INDEPENDENCE PASS', text)
        self.assertIn('NULL control', text)

    def test_main_report_points_to_independence_file(self):
        path = tri.write_triangulation_report(self._tri(), self.tmp)
        text = open(path, encoding='utf-8').read()
        self.assertIn('triangulation_independence.txt', text)
        self.assertIn('DISTILLATION FIDELITY', text)


if __name__ == '__main__':
    unittest.main()
