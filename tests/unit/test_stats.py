"""
tests/test_stats.py
-------------------
Unit tests for analysis/stats.py — the inference toolkit (Wilson CIs, cluster
bootstrap, within-stratum permutation, effect sizes, BH-FDR, mixed-effects trend).
No model backends; pure numpy/scipy/statsmodels.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from analysis import stats as S


class TestWilson(unittest.TestCase):
    def test_bounds_within_unit(self):
        lo, hi = S.wilson_ci(5, 10)
        self.assertTrue(0 <= lo < 0.5 < hi <= 1)

    def test_zero_n(self):
        self.assertEqual(S.wilson_ci(0, 0), (0.0, 0.0))

    def test_extreme_proportion_clipped(self):
        lo, hi = S.wilson_ci(10, 10)
        self.assertLessEqual(hi, 1.0)
        self.assertGreater(lo, 0.0)


class TestClusterBootstrap(unittest.TestCase):
    def test_point_and_ci_brackets(self):
        rng = np.random.default_rng(0)
        clusters = np.repeat(np.arange(10), 5)
        values = rng.normal(2.0, 1.0, size=50)
        res = S.cluster_bootstrap_ci(values, clusters, n_boot=500)
        self.assertAlmostEqual(res['point'], float(np.mean(values)), places=6)
        self.assertLess(res['lo'], res['point'])
        self.assertGreater(res['hi'], res['point'])
        self.assertEqual(res['n_clusters'], 10)

    def test_single_cluster_no_ci(self):
        res = S.cluster_bootstrap_ci([1.0, 2.0, 3.0], ['a', 'a', 'a'])
        self.assertTrue(np.isnan(res['lo']))
        self.assertEqual(res['n_clusters'], 1)


class TestPermutation(unittest.TestCase):
    def test_strong_signal_low_p(self):
        # Group True has clearly higher values → small p.
        values = np.concatenate([np.full(20, 5.0), np.full(20, 0.0)])
        mask = np.array([True] * 20 + [False] * 20)
        res = S.permutation_test(values, mask, n_perm=500, seed=1)
        self.assertLess(res['p_value'], 0.05)
        self.assertAlmostEqual(res['observed'], 5.0, places=6)

    def test_null_high_p(self):
        rng = np.random.default_rng(3)
        values = rng.normal(0, 1, size=60)
        mask = np.array([True, False] * 30)
        res = S.permutation_test(values, mask, n_perm=500, seed=2)
        self.assertGreater(res['p_value'], 0.05)

    def test_within_stratum_shuffle_preserves_counts(self):
        values = np.arange(12, dtype=float)
        mask = np.array([True, False] * 6)
        strata = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        res = S.permutation_test(values, mask, strata=strata, n_perm=100)
        self.assertGreater(res['n_perm'], 0)


class TestEffectSizes(unittest.TestCase):
    def test_cohens_h_zero_when_equal(self):
        self.assertAlmostEqual(S.cohens_h(0.4, 0.4), 0.0, places=9)

    def test_cramers_v_strong(self):
        table = np.array([[40, 0], [0, 40]])
        res = S.cramers_v(table)
        self.assertGreater(res['cramers_v'], 0.9)

    def test_odds_ratio_gt_one(self):
        res = S.odds_ratio_ci(20, 5, 5, 20)
        self.assertGreater(res['odds_ratio'], 1.0)
        self.assertLess(res['lo'], res['odds_ratio'])
        self.assertGreater(res['hi'], res['odds_ratio'])

    def test_cliffs_delta_sign(self):
        self.assertGreater(S.cliffs_delta([5, 6, 7], [1, 2, 3]), 0.9)


class TestBH(unittest.TestCase):
    def test_rejects_small_pvalues(self):
        p = [0.001, 0.002, 0.5, 0.9]
        res = S.benjamini_hochberg(p, alpha=0.05)
        self.assertTrue(res['reject'][0])
        self.assertTrue(res['reject'][1])
        self.assertFalse(res['reject'][2])

    def test_all_null(self):
        res = S.benjamini_hochberg([0.6, 0.7, 0.8])
        self.assertFalse(any(res['reject']))


class TestMixedEffects(unittest.TestCase):
    def test_trend_positive_slope(self):
        rows = []
        rng = np.random.default_rng(7)
        for pid in range(8):
            base = rng.normal(0, 0.2)
            for t in range(4):
                rows.append(dict(pid=f'P{pid}', t=t,
                                 y=base + 0.5 * t + rng.normal(0, 0.1)))
        df = pd.DataFrame(rows)
        res = S.mixedlm_trend(df, 'y', 't', 'pid')
        self.assertGreater(res['slope'], 0.3)
        self.assertLess(res['p_value'], 0.05)
        self.assertIn(res['method'], ('mixedlm', 'ols'))

    def test_trend_graceful_single_group(self):
        df = pd.DataFrame({'y': [1.0, 2.0, 3.0], 't': [0, 1, 2], 'pid': ['A', 'A', 'A']})
        res = S.mixedlm_trend(df, 'y', 't', 'pid')
        # single group → OLS fallback still yields a slope
        self.assertAlmostEqual(res['slope'], 1.0, places=6)

    def test_sign_test(self):
        self.assertLess(S.sign_test(9, 10)['p_value'], 0.05)


if __name__ == '__main__':
    unittest.main()
