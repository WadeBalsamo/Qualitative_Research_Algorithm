"""
tests/unit/test_gnn_figures.py
--------------------------------
Unit tests for gnn_layer/figures.py.
Sets matplotlib to Agg backend BEFORE any pyplot import.

Tests:
  - plot_validation_kappa:  writes gnn_validation_kappa_by_stage.png to 05_figures/
  - plot_motif_influence:   writes gnn_motif_influence.png
  - plot_coupling_factors:  writes gnn_coupling_factors.png
  - generate_gnn_figures:   returns paths for every figure that has data
  - graceful degradation:   returns None when data CSV is absent
"""

import matplotlib
matplotlib.use('Agg')  # must be before any pyplot import

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
from gnn_layer import figures as _figs
from process import output_paths as _paths


def _write_validation_csv(tmp, n_vaamr=5, n_purer=5):
    """Write a minimal gnn_validation.csv so plot_validation_kappa has data."""
    gnn_dir = _paths.gnn_data_dir(tmp)
    os.makedirs(gnn_dir, exist_ok=True)
    rows = []
    stages = ['Vigilance', 'Avoidance', 'AttReg', 'Metacog', 'Reappraisal']
    moves = ['P', 'U', 'R', 'E', 'R2']
    for i in range(n_vaamr):
        rows.append({'scope': 'per_class', 'framework': 'vaamr',
                     'class_id': i, 'class_name': stages[i % 5],
                     'kappa': 0.5 + (i * 0.05), 'support': 10})
    for j in range(n_purer):
        rows.append({'scope': 'per_class', 'framework': 'purer',
                     'class_id': j, 'class_name': moves[j % 5],
                     'kappa': 0.6 + (j * 0.03), 'support': 8})
    pd.DataFrame(rows).to_csv(os.path.join(gnn_dir, 'gnn_validation.csv'), index=False)


def _write_cue_motifs_csv(tmp, n=6):
    """Write a minimal cue_motifs.csv so plot_motif_influence has data."""
    gnn_dir = _paths.gnn_data_dir(tmp)
    os.makedirs(gnn_dir, exist_ok=True)
    rows = []
    for i in range(n):
        rows.append({
            'motif_id': i,
            'influence': 0.8 + i * 0.2,
            'purer_purity': 0.3 + i * 0.1,
            'dominant_purer': i % 5,
            'n_blocks': 4 + i,
            'n_exemplars': 2,
            'mean_pred_forward': 0.5,
        })
    pd.DataFrame(rows).to_csv(os.path.join(gnn_dir, 'cue_motifs.csv'), index=False)


def _write_coupling_factors_csv(tmp, n=4):
    """Write a minimal coupling_factors.csv so plot_coupling_factors has data."""
    gnn_dir = _paths.gnn_data_dir(tmp)
    os.makedirs(gnn_dir, exist_ok=True)
    rows = []
    cf_labels = ['bond', 'goal_alignment', 'task_agreement', 'empathy_acceptance_regard']
    for i in range(n):
        rows.append({
            'factor': i,
            'explained_variance_ratio': 0.3 - i * 0.05,
            'forward_corr': 0.4 - i * 0.15,
            'nearest_cf_ic': cf_labels[i % len(cf_labels)],
            'cf_ic_similarity': 0.7,
            'n_exemplars': 2,
        })
    pd.DataFrame(rows).to_csv(os.path.join(gnn_dir, 'coupling_factors.csv'), index=False)


class TestPlotValidationKappa(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writes_png_to_figures_dir(self):
        _write_validation_csv(self.tmp)
        path = _figs.plot_validation_kappa(self.tmp, irr_target=0.70)
        self.assertIsNotNone(path)
        expected = os.path.join(_paths.figures_dir(self.tmp), 'gnn_validation_kappa_by_stage.png')
        self.assertEqual(path, expected)
        self.assertTrue(os.path.isfile(path))

    def test_png_nonzero_size(self):
        _write_validation_csv(self.tmp)
        path = _figs.plot_validation_kappa(self.tmp)
        self.assertGreater(os.path.getsize(path), 100)

    def test_returns_none_when_no_csv(self):
        result = _figs.plot_validation_kappa(self.tmp)
        self.assertIsNone(result)

    def test_returns_none_when_no_per_class_rows(self):
        gnn_dir = _paths.gnn_data_dir(self.tmp)
        os.makedirs(gnn_dir, exist_ok=True)
        pd.DataFrame([{'scope': 'overall', 'framework': 'vaamr', 'class_id': 0,
                        'class_name': 'all', 'kappa': 0.7, 'support': 30}]
                     ).to_csv(os.path.join(gnn_dir, 'gnn_validation.csv'), index=False)
        result = _figs.plot_validation_kappa(self.tmp)
        self.assertIsNone(result)

    def test_custom_irr_target_accepted(self):
        _write_validation_csv(self.tmp)
        path = _figs.plot_validation_kappa(self.tmp, irr_target=0.60)
        self.assertIsNotNone(path)
        self.assertTrue(os.path.isfile(path))


class TestPlotMotifInfluence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writes_png_to_figures_dir(self):
        _write_cue_motifs_csv(self.tmp)
        path = _figs.plot_motif_influence(self.tmp)
        self.assertIsNotNone(path)
        expected = os.path.join(_paths.figures_dir(self.tmp), 'gnn_motif_influence.png')
        self.assertEqual(path, expected)
        self.assertTrue(os.path.isfile(path))

    def test_png_nonzero_size(self):
        _write_cue_motifs_csv(self.tmp)
        path = _figs.plot_motif_influence(self.tmp)
        self.assertGreater(os.path.getsize(path), 100)

    def test_returns_none_when_no_csv(self):
        result = _figs.plot_motif_influence(self.tmp)
        self.assertIsNone(result)

    def test_returns_none_on_empty_csv(self):
        gnn_dir = _paths.gnn_data_dir(self.tmp)
        os.makedirs(gnn_dir, exist_ok=True)
        pd.DataFrame(columns=['motif_id', 'influence', 'purer_purity']).to_csv(
            os.path.join(gnn_dir, 'cue_motifs.csv'), index=False)
        result = _figs.plot_motif_influence(self.tmp)
        self.assertIsNone(result)


class TestPlotCouplingFactors(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writes_png_to_figures_dir(self):
        _write_coupling_factors_csv(self.tmp)
        path = _figs.plot_coupling_factors(self.tmp)
        self.assertIsNotNone(path)
        expected = os.path.join(_paths.figures_dir(self.tmp), 'gnn_coupling_factors.png')
        self.assertEqual(path, expected)
        self.assertTrue(os.path.isfile(path))

    def test_png_nonzero_size(self):
        _write_coupling_factors_csv(self.tmp)
        path = _figs.plot_coupling_factors(self.tmp)
        self.assertGreater(os.path.getsize(path), 100)

    def test_returns_none_when_no_csv(self):
        result = _figs.plot_coupling_factors(self.tmp)
        self.assertIsNone(result)

    def test_returns_none_when_no_forward_corr_column(self):
        gnn_dir = _paths.gnn_data_dir(self.tmp)
        os.makedirs(gnn_dir, exist_ok=True)
        pd.DataFrame([{'factor': 0, 'explained_variance_ratio': 0.4}]).to_csv(
            os.path.join(gnn_dir, 'coupling_factors.csv'), index=False)
        result = _figs.plot_coupling_factors(self.tmp)
        self.assertIsNone(result)

    def test_all_nan_forward_corr_returns_none(self):
        gnn_dir = _paths.gnn_data_dir(self.tmp)
        os.makedirs(gnn_dir, exist_ok=True)
        pd.DataFrame([{'factor': 0, 'forward_corr': float('nan')}]).to_csv(
            os.path.join(gnn_dir, 'coupling_factors.csv'), index=False)
        result = _figs.plot_coupling_factors(self.tmp)
        self.assertIsNone(result)


class TestGenerateGnnFigures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_data_returns_empty_list(self):
        paths = _figs.generate_gnn_figures(self.tmp)
        self.assertIsInstance(paths, list)
        self.assertEqual(paths, [])

    def test_returns_paths_for_available_data(self):
        _write_validation_csv(self.tmp)
        _write_cue_motifs_csv(self.tmp)
        _write_coupling_factors_csv(self.tmp)
        paths = _figs.generate_gnn_figures(self.tmp)
        self.assertIsInstance(paths, list)
        self.assertGreater(len(paths), 0)
        for p in paths:
            self.assertTrue(os.path.isfile(p))

    def test_returns_up_to_three_figures(self):
        _write_validation_csv(self.tmp)
        _write_cue_motifs_csv(self.tmp)
        _write_coupling_factors_csv(self.tmp)
        paths = _figs.generate_gnn_figures(self.tmp)
        # There are 3 plotters: validation, motif, coupling
        self.assertLessEqual(len(paths), 3)

    def test_partial_data_produces_partial_figures(self):
        """Only validation data → only validation figure."""
        _write_validation_csv(self.tmp)
        paths = _figs.generate_gnn_figures(self.tmp)
        basenames = [os.path.basename(p) for p in paths]
        self.assertIn('gnn_validation_kappa_by_stage.png', basenames)
        # motif and coupling should be absent
        self.assertNotIn('gnn_motif_influence.png', basenames)
        self.assertNotIn('gnn_coupling_factors.png', basenames)

    def test_all_pngs_written_to_figures_dir(self):
        _write_validation_csv(self.tmp)
        _write_cue_motifs_csv(self.tmp)
        _write_coupling_factors_csv(self.tmp)
        paths = _figs.generate_gnn_figures(self.tmp)
        figs_dir = _paths.figures_dir(self.tmp)
        for p in paths:
            self.assertTrue(p.startswith(figs_dir))


if __name__ == '__main__':
    unittest.main()
