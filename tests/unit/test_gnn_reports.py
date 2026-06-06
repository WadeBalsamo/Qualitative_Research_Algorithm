"""
tests/unit/test_gnn_reports.py
--------------------------------
Unit tests for gnn_layer/reports.py — every public writer function.

Asserts:
  - file written to exact path (gnn_data_dir → 03_analysis_data/gnn;
    reports_gnn_dir → 06_reports/06_gnn)
  - output has expected columns / sections
"""

import os
import sys
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.testhelpers import synthetic_df, embedding_patch, make_master_df
from gnn_layer.config import GnnLayerConfig
from gnn_layer import reports as _rep
from process import output_paths as _paths


class TestWriteSegmentPositions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _positions(self, n=8):
        rng = np.random.default_rng(0)
        mix = rng.dirichlet(np.ones(5), size=n).astype(np.float32)
        return {
            'segment_id': [f's{i}' for i in range(n)],
            'node_type': ['participant_segment' if i % 2 == 0 else 'therapist_segment'
                          for i in range(n)],
            'progression_coord': rng.standard_normal(n).tolist(),
            'vaamr_mixture': mix,
            'gnn_embedding': rng.standard_normal((n, 8)).astype(np.float32),
        }

    def test_file_written_to_gnn_data_dir(self):
        pos = self._positions()
        path = _rep.write_segment_positions(pos, self.tmp)
        expected = os.path.join(_paths.gnn_data_dir(self.tmp), 'segment_positions.csv')
        self.assertEqual(path, expected)
        self.assertTrue(os.path.isfile(path))

    def test_csv_has_required_columns(self):
        pos = self._positions()
        path = _rep.write_segment_positions(pos, self.tmp)
        df = pd.read_csv(path)
        for col in ('segment_id', 'node_type', 'progression_coord'):
            self.assertIn(col, df.columns)

    def test_csv_has_vaamr_mix_columns(self):
        pos = self._positions()
        path = _rep.write_segment_positions(pos, self.tmp)
        df = pd.read_csv(path)
        mix_cols = [c for c in df.columns if c.startswith('vaamr_mix_')]
        self.assertEqual(len(mix_cols), 5)

    def test_vaamr_mix_sums_to_1(self):
        pos = self._positions()
        path = _rep.write_segment_positions(pos, self.tmp)
        df = pd.read_csv(path)
        mix_cols = [f'vaamr_mix_{k}' for k in range(5)]
        self.assertTrue(np.allclose(df[mix_cols].sum(axis=1), 1.0, atol=1e-3))

    def test_row_count_matches_input(self):
        pos = self._positions(10)
        path = _rep.write_segment_positions(pos, self.tmp)
        df = pd.read_csv(path)
        self.assertEqual(len(df), 10)


class TestWriteCueMotifs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _inputs(self):
        stats = {
            0: {'influence': 1.5, 'mean_pred_forward': 0.7, 'n_blocks': 5},
            1: {'influence': 0.9, 'mean_pred_forward': 0.4, 'n_blocks': 3},
        }
        purity = {
            0: {'dominant_purer': 2, 'purer_purity': 0.8},
            1: {'dominant_purer': 0, 'purer_purity': 0.4},
        }
        exemplars = {
            0: [{'session_id': 'c1s1', 'from_stage': 0, 'to_stage': 1,
                 'from_seg_id': 'p0', 'to_seg_id': 'p1'}],
            1: [],
        }
        return stats, purity, exemplars

    def test_file_written_to_gnn_data_dir(self):
        stats, purity, exemplars = self._inputs()
        path = _rep.write_cue_motifs(stats, purity, exemplars, self.tmp)
        expected = os.path.join(_paths.gnn_data_dir(self.tmp), 'cue_motifs.csv')
        self.assertEqual(path, expected)
        self.assertTrue(os.path.isfile(path))

    def test_columns(self):
        stats, purity, exemplars = self._inputs()
        path = _rep.write_cue_motifs(stats, purity, exemplars, self.tmp)
        df = pd.read_csv(path)
        for col in ('motif_id', 'n_blocks', 'influence', 'mean_pred_forward',
                    'dominant_purer', 'purer_purity', 'n_exemplars'):
            self.assertIn(col, df.columns)

    def test_rows_match_motif_count(self):
        stats, purity, exemplars = self._inputs()
        path = _rep.write_cue_motifs(stats, purity, exemplars, self.tmp)
        df = pd.read_csv(path)
        self.assertEqual(len(df), 2)


class TestWriteGnnHeadPredictions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _preds(self, n=6):
        return {
            'segment_id': [f's{i}' for i in range(n)],
            'node_type': ['participant_segment'] * n,
            'gnn_vaamr_pred': list(range(n)),
            'gnn_vaamr_conf': [0.8] * n,
            'gnn_purer_pred': [i % 5 for i in range(n)],
            'gnn_purer_conf': [0.7] * n,
        }

    def test_file_written_to_gnn_data_dir(self):
        preds = self._preds()
        path = _rep.write_gnn_head_predictions(preds, self.tmp)
        expected = os.path.join(_paths.gnn_data_dir(self.tmp), 'gnn_head_predictions.csv')
        self.assertEqual(path, expected)
        self.assertTrue(os.path.isfile(path))

    def test_columns_present(self):
        preds = self._preds()
        path = _rep.write_gnn_head_predictions(preds, self.tmp)
        df = pd.read_csv(path)
        for col in ('segment_id', 'node_type', 'gnn_vaamr_pred', 'gnn_purer_pred'):
            self.assertIn(col, df.columns)

    def test_row_count(self):
        preds = self._preds(8)
        path = _rep.write_gnn_head_predictions(preds, self.tmp)
        df = pd.read_csv(path)
        self.assertEqual(len(df), 8)


class TestWriteGnnVsLlmLift(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _comparison(self):
        return pd.DataFrame([
            {'vaamr_stage': 0, 'vce_code': 'a', 'lift_gnn': 1.8, 'lift_llm': 1.6, 'both_elevated': True},
            {'vaamr_stage': 1, 'vce_code': 'b', 'lift_gnn': 0.9, 'lift_llm': 1.2, 'both_elevated': False},
        ])

    def test_file_written_to_gnn_data_dir(self):
        comp = self._comparison()
        path = _rep.write_gnn_vs_llm_lift(comp, self.tmp)
        expected = os.path.join(_paths.gnn_data_dir(self.tmp), 'gnn_vs_llm_lift.csv')
        self.assertEqual(path, expected)
        self.assertTrue(os.path.isfile(path))

    def test_roundtrip_columns(self):
        comp = self._comparison()
        path = _rep.write_gnn_vs_llm_lift(comp, self.tmp)
        df = pd.read_csv(path)
        for col in ('vaamr_stage', 'vce_code', 'lift_gnn', 'lift_llm', 'both_elevated'):
            self.assertIn(col, df.columns)


class TestWriteCouplingFactors(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _inputs(self, n=3):
        evr = [0.4, 0.3, 0.15][:n]
        corr = [0.6, -0.2, 0.1][:n]
        factors = {'explained_variance_ratio': evr, 'factor_forward_corr': corr}
        exemplars = {f: [{'from_stage': 0, 'to_stage': 1,
                           'session_id': 'c1s1', 'from_seg_id': 'p0'}]
                     for f in range(n)}
        interpretation = {f: {'nearest_cf_ic': 'bond', 'similarity': 0.7} for f in range(n)}
        return factors, exemplars, interpretation

    def test_file_written_to_gnn_data_dir(self):
        factors, exemplars, interp = self._inputs()
        path = _rep.write_coupling_factors(factors, exemplars, interp, self.tmp)
        expected = os.path.join(_paths.gnn_data_dir(self.tmp), 'coupling_factors.csv')
        self.assertEqual(path, expected)
        self.assertTrue(os.path.isfile(path))

    def test_columns(self):
        factors, exemplars, interp = self._inputs()
        path = _rep.write_coupling_factors(factors, exemplars, interp, self.tmp)
        df = pd.read_csv(path)
        for col in ('factor', 'explained_variance_ratio', 'forward_corr',
                    'nearest_cf_ic', 'cf_ic_similarity', 'n_exemplars'):
            self.assertIn(col, df.columns)

    def test_row_count(self):
        factors, exemplars, interp = self._inputs(3)
        path = _rep.write_coupling_factors(factors, exemplars, interp, self.tmp)
        df = pd.read_csv(path)
        self.assertEqual(len(df), 3)


class TestWriteEmergentMotifsReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _inputs(self):
        stats = {
            0: {'influence': 1.7, 'mean_pred_forward': 0.8, 'n_blocks': 6},
        }
        purity = {0: {'dominant_purer': 1, 'purer_purity': 0.3}}
        exemplars = {
            0: [{'session_id': 'c1s1', 'from_stage': 1, 'to_stage': 2,
                 'from_seg_id': 'p1', 'to_seg_id': 'p2'}]
        }
        return stats, purity, exemplars

    def test_file_written_to_reports_gnn_dir(self):
        stats, purity, exemplars = self._inputs()
        path = _rep.write_emergent_motifs_report([0], stats, purity, exemplars, self.tmp)
        expected = os.path.join(_paths.reports_gnn_dir(self.tmp), 'emergent_motifs.txt')
        self.assertEqual(path, expected)
        self.assertTrue(os.path.isfile(path))

    def test_header_text(self):
        stats, purity, exemplars = self._inputs()
        path = _rep.write_emergent_motifs_report([0], stats, purity, exemplars, self.tmp)
        text = open(path, encoding='utf-8').read()
        self.assertIn('EMERGENT', text)
        self.assertIn('HUMAN REVIEW', text)

    def test_no_flagged_motifs_message(self):
        stats, purity, exemplars = self._inputs()
        path = _rep.write_emergent_motifs_report([], stats, purity, exemplars, self.tmp)
        text = open(path, encoding='utf-8').read()
        self.assertIn('No emergent motifs', text)

    def test_flagged_motif_details_present(self):
        stats, purity, exemplars = self._inputs()
        path = _rep.write_emergent_motifs_report([0], stats, purity, exemplars, self.tmp)
        text = open(path, encoding='utf-8').read()
        self.assertIn('Motif 0', text)
        self.assertIn('influence', text)


class TestWriteCouplingReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_file_written_to_reports_gnn_dir(self):
        factors = {'explained_variance_ratio': [0.4, 0.3], 'factor_forward_corr': [0.5, -0.1]}
        exemplars = {0: [], 1: []}
        interp = {0: {'nearest_cf_ic': 'bond', 'similarity': 0.8}}
        path = _rep.write_coupling_report(factors, exemplars, interp, self.tmp)
        expected = os.path.join(_paths.reports_gnn_dir(self.tmp), 'coupling.txt')
        self.assertEqual(path, expected)
        self.assertTrue(os.path.isfile(path))

    def test_header_text(self):
        factors = {'explained_variance_ratio': [0.4], 'factor_forward_corr': [0.5]}
        path = _rep.write_coupling_report(factors, {}, {}, self.tmp)
        text = open(path, encoding='utf-8').read()
        self.assertIn('COUPLING', text)

    def test_cf_ic_label_in_report_when_provided(self):
        factors = {'explained_variance_ratio': [0.4], 'factor_forward_corr': [0.5]}
        interp = {0: {'nearest_cf_ic': 'goal_alignment', 'similarity': 0.75}}
        path = _rep.write_coupling_report(factors, {0: []}, interp, self.tmp)
        text = open(path, encoding='utf-8').read()
        self.assertIn('goal_alignment', text)


class TestWriteGnnConstructSignal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _rows(self):
        return [
            {'ablate': 'vce', 'objective_removed': 'vce_multilabel',
             'best_loss_full': 0.3, 'best_loss_ablated': 0.45, 'delta': 0.15},
            {'ablate': 'purer', 'objective_removed': 'purer',
             'best_loss_full': 0.3, 'best_loss_ablated': 0.35, 'delta': 0.05},
        ]

    def test_files_written(self):
        rows = self._rows()
        path = _rep.write_gnn_construct_signal(rows, self.tmp)
        expected = os.path.join(_paths.reports_gnn_dir(self.tmp), 'construct_signal.txt')
        self.assertEqual(path, expected)
        self.assertTrue(os.path.isfile(path))
        csv_path = os.path.join(_paths.gnn_data_dir(self.tmp), 'gnn_construct_signal.csv')
        self.assertTrue(os.path.isfile(csv_path))

    def test_report_contains_ablation_header(self):
        path = _rep.write_gnn_construct_signal(self._rows(), self.tmp)
        text = open(path, encoding='utf-8').read()
        self.assertIn('ABLATION', text)

    def test_report_contains_delta_values(self):
        path = _rep.write_gnn_construct_signal(self._rows(), self.tmp)
        text = open(path, encoding='utf-8').read()
        self.assertIn('δloss', text.lower())


class TestWriteCueBlockAssignments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_file_written_to_gnn_data_dir(self):
        rows = [
            {'session_id': 'c1s1', 'from_seg_id': 'p0', 'to_seg_id': 'p1',
             'from_stage': 0, 'to_stage': 1},
            {'session_id': 'c1s1', 'from_seg_id': 'p1', 'to_seg_id': 'p2',
             'from_stage': 1, 'to_stage': 2},
        ]
        motif_ids = np.array([0, 1])
        path = _rep.write_cue_block_assignments(rows, motif_ids, self.tmp)
        expected = os.path.join(_paths.gnn_data_dir(self.tmp), 'cue_block_assignments.csv')
        self.assertEqual(path, expected)
        self.assertTrue(os.path.isfile(path))

    def test_csv_columns(self):
        rows = [
            {'session_id': 'c1s1', 'from_seg_id': 'p0', 'to_seg_id': 'p1',
             'from_stage': 0, 'to_stage': 1},
        ]
        motif_ids = np.array([2])
        path = _rep.write_cue_block_assignments(rows, motif_ids, self.tmp)
        df = pd.read_csv(path)
        for col in ('session_id', 'from_seg_id', 'to_seg_id', 'from_stage', 'to_stage', 'motif_id'):
            self.assertIn(col, df.columns)


if __name__ == '__main__':
    unittest.main()
