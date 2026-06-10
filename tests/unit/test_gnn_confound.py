"""
tests/unit/test_gnn_confound.py
-------------------------------
Unit tests for WS3 — confound localization (gnn_layer/confound.py).

Hermetic: a synthetic dataset + counterfactual with an engineered sign-inverting cell (observed
Δ positive, learned counterfactual negative) so the divergence + sign-disagreement flag are
verifiable. Also exercises the clustered-divergence CI, the writers, and the figure guard.
numpy only; no model, no network, no qra.db.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from gnn_layer import confound as CF
from gnn_layer import figures as FIG


def _ds_cf():
    """One cell (from_stage 2, move 1): observed Δ = +2 (FROM peak@1 → TO peak@3); learned = −0.5."""
    S = 5
    F = np.tile(np.eye(S)[1], (4, 1))
    Y = np.tile(np.eye(S)[3], (4, 1))
    ds = {'F': F, 'Y': Y, 'from_stage': np.array([2, 2, 2, 2]),
          'move': np.array([1, 1, 1, 1]), 'parts': ['P0', 'P0', 'P1', 'P1'], 'n_stages': S}
    cf = {'status': 'ok',
          'rows': [{'from_stage': 2, 'move': 1, 'influence': -0.5, 'participant_id': p}
                   for p in ['P0', 'P0', 'P1', 'P1']]}
    return ds, cf


class TestDivergenceCI(unittest.TestCase):

    def test_point_and_ci(self):
        ci = CF._divergence_ci([2.0, 2.0, 2.0, 2.0], ['P0', 'P0', 'P1', 'P1'],
                               [-0.5, -0.5, -0.5, -0.5], ['P0', 'P0', 'P1', 'P1'],
                               n_boot=200, seed=1)
        self.assertAlmostEqual(ci['point'], 2.5, places=3)
        self.assertIsNotNone(ci['lo'])

    def test_single_participant_no_ci(self):
        ci = CF._divergence_ci([1.0, 1.0], ['P0', 'P0'], [0.0, 0.0], ['P0', 'P0'], n_boot=50)
        self.assertIsNone(ci['lo'])


class TestCellValues(unittest.TestCase):

    def test_keys_and_values(self):
        ds, cf = _ds_cf()
        obs, lrn = CF._cell_values(ds, cf)
        self.assertIn((2, 1), obs)
        self.assertIn((2, 1), lrn)
        # observed Δ = E[stage 3] − E[stage 1] = 2 per block
        self.assertTrue(all(abs(v - 2.0) < 1e-6 for v in obs[(2, 1)][0]))


class TestRunConfound(unittest.TestCase):

    def test_divergence_and_sign_inversion(self):
        ds, cf = _ds_cf()
        with tempfile.TemporaryDirectory() as tmp:
            res = CF.run_confound_localization(None, tmp, config=None,
                                               transition_result={'counterfactual': cf, 'dataset': ds})
            self.assertEqual(res['status'], 'ok')
            self.assertTrue(res['cells'])
            c = res['cells'][0]
            self.assertEqual((c['from_stage'], c['move']), (2, 1))
            self.assertTrue(c['sign_disagree'])              # observed +, counterfactual −
            self.assertAlmostEqual(c['divergence'], 2.5, places=3)
            from process import output_paths as _paths
            self.assertTrue(os.path.isfile(
                os.path.join(_paths.reports_gnn_dir(tmp), 'confound_localization.txt')))
            self.assertTrue(os.path.isfile(os.path.join(tmp, '03_analysis_data', 'gnn',
                                                        'confound_localization.csv')))


class TestReportWriterAndFigure(unittest.TestCase):

    def test_report_substrings(self):
        cells = [{'from_stage': 2, 'from_stage_name': 'AttentionReg', 'move': 1,
                  'move_name': 'Utilization', 'observed_delta': 0.2, 'counterfactual': -0.1,
                  'divergence': 0.3, 'div_lo': 0.05, 'div_hi': 0.55, 'sign_disagree': True,
                  'n_observed': 8, 'fdr_significant': False}]
        with tempfile.TemporaryDirectory() as tmp:
            rep = CF.write_confound_report(cells, tmp)
            txt = open(rep, encoding='utf-8').read()
            self.assertIn('CONFOUND LOCALIZATION', txt)
            self.assertIn('INVERT', txt.upper())

    def test_figure_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(FIG.plot_confound_localization(tmp))


if __name__ == '__main__':
    unittest.main()
