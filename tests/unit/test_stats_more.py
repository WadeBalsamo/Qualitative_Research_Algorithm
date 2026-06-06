"""
tests/unit/test_stats_more.py
-----------------------------
Expanded unit tests for analysis/stats.py — covering every public function
not already exercised by tests/test_stats.py.

Coverage added here (gaps from the top-level test_stats.py):
  wilson_ci         — 0/n and n/n stay in [0,1]; known numeric value
  permutation_test  — degenerate empty/all-same-group; n_perm / n_group keys
  cohens_h          — extreme values (0 vs 0, 1 vs 0); clipping at boundaries
  cramers_v         — perfect association (V=1.0); degenerate (all-zero, 1-row table)
  odds_ratio_ci     — symmetric OR > 1 with CI brackets; all-zero cells → OR = 1.0
  cliffs_delta      — fully separated +1 / −1; equal groups → 0.0; empty → nan
  benjamini_hochberg — textbook 4-p case with known q-values + reject mask;
                       all-large-p → no rejections; nan p-values handled; empty list
  sign_test         — n=0 returns nan; 5/10 → p=1.0; 0/10 small p
  cluster_bootstrap_ci — empty returns nan point; all-nan values returns nan point;
                          custom statistic (median); n/n_clusters keys correct
  mixedlm_trend     — positive slope detected; negative slope detected; empty df → nan/none
  mixedlm_delta     — deterministic R-vs-P contrast recoverable; single group → None;
                      1-row df → None
"""

import math
import os
import sys
import unittest
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis import stats as S


# ---------------------------------------------------------------------------
# wilson_ci — additional coverage
# ---------------------------------------------------------------------------

class TestWilsonCIMore(unittest.TestCase):
    """Known-value and boundary cases not in the top-level suite."""

    def test_zero_successes_stays_in_unit_interval(self):
        lo, hi = S.wilson_ci(0, 10)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        self.assertEqual(lo, 0.0)               # lower bound clipped to 0

    def test_all_successes_stays_in_unit_interval(self):
        lo, hi = S.wilson_ci(10, 10)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)           # upper bound clipped to 1

    def test_zero_n_returns_0_0(self):
        self.assertEqual(S.wilson_ci(0, 0), (0.0, 0.0))

    def test_large_n_zero_successes_upper_bound_small(self):
        lo, hi = S.wilson_ci(0, 100)
        self.assertEqual(lo, 0.0)
        self.assertLess(hi, 0.04)              # for n=100 and z≈1.96, hi ≈ 0.037

    def test_half_n_ci_straddles_0_5(self):
        # 5/10 → CI must straddle 0.5
        lo, hi = S.wilson_ci(5, 10)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_returns_tuple_of_two_floats(self):
        result = S.wilson_ci(3, 7)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)


# ---------------------------------------------------------------------------
# cliffs_delta — additional coverage
# ---------------------------------------------------------------------------

class TestCliffsDeltaMore(unittest.TestCase):

    def test_fully_separated_positive(self):
        # x always > y → delta = +1.0
        self.assertAlmostEqual(S.cliffs_delta([10, 11, 12], [1, 2, 3]), 1.0, places=9)

    def test_fully_separated_negative(self):
        # x always < y → delta = -1.0
        self.assertAlmostEqual(S.cliffs_delta([1, 2, 3], [10, 11, 12]), -1.0, places=9)

    def test_identical_groups_zero(self):
        # same values → all ties → delta = 0.0
        self.assertAlmostEqual(S.cliffs_delta([1, 2, 3], [1, 2, 3]), 0.0, places=9)

    def test_empty_x_returns_nan(self):
        result = S.cliffs_delta([], [1, 2, 3])
        self.assertTrue(math.isnan(result))

    def test_empty_y_returns_nan(self):
        result = S.cliffs_delta([1, 2, 3], [])
        self.assertTrue(math.isnan(result))

    def test_both_empty_returns_nan(self):
        result = S.cliffs_delta([], [])
        self.assertTrue(math.isnan(result))

    def test_all_nan_x_returns_nan(self):
        result = S.cliffs_delta([float('nan'), float('nan')], [1, 2])
        self.assertTrue(math.isnan(result))

    def test_return_type_is_float(self):
        self.assertIsInstance(S.cliffs_delta([1], [2]), float)

    def test_antisymmetric(self):
        d1 = S.cliffs_delta([5, 6, 7], [1, 2, 3])
        d2 = S.cliffs_delta([1, 2, 3], [5, 6, 7])
        self.assertAlmostEqual(d1, -d2, places=9)


# ---------------------------------------------------------------------------
# cohens_h — additional coverage
# ---------------------------------------------------------------------------

class TestCohensHMore(unittest.TestCase):

    def test_zero_vs_zero_is_zero(self):
        self.assertAlmostEqual(S.cohens_h(0.0, 0.0), 0.0, places=9)

    def test_one_vs_zero_equals_pi(self):
        # 2*arcsin(1) - 2*arcsin(0) = 2*(pi/2) - 0 = pi
        self.assertAlmostEqual(S.cohens_h(1.0, 0.0), math.pi, places=9)

    def test_zero_vs_one_equals_negative_pi(self):
        self.assertAlmostEqual(S.cohens_h(0.0, 1.0), -math.pi, places=9)

    def test_half_vs_half_is_zero(self):
        self.assertAlmostEqual(S.cohens_h(0.5, 0.5), 0.0, places=9)

    def test_out_of_range_clamped(self):
        # values outside [0,1] should be clamped, not raise
        try:
            h = S.cohens_h(1.5, -0.5)
            # clamped to (1.0, 0.0) → equals pi
            self.assertAlmostEqual(h, math.pi, places=9)
        except Exception as exc:
            self.fail(f"cohens_h raised with out-of-range input: {exc}")

    def test_sign_convention(self):
        # p1 > p2 → h > 0
        self.assertGreater(S.cohens_h(0.7, 0.3), 0.0)
        # p1 < p2 → h < 0
        self.assertLess(S.cohens_h(0.3, 0.7), 0.0)

    def test_return_type_is_float(self):
        self.assertIsInstance(S.cohens_h(0.4, 0.6), float)


# ---------------------------------------------------------------------------
# cramers_v — additional coverage
# ---------------------------------------------------------------------------

class TestCramersVMore(unittest.TestCase):

    def test_perfect_association_is_one(self):
        # Off-diagonal zeros → V = 1.0
        table = np.array([[10, 0], [0, 10]])
        res = S.cramers_v(table)
        self.assertAlmostEqual(res['cramers_v'], 1.0, places=9)

    def test_all_zero_table_returns_nan(self):
        res = S.cramers_v(np.zeros((2, 2)))
        self.assertTrue(math.isnan(res['cramers_v']))

    def test_single_row_table_returns_nan(self):
        # Only 1 row → r < 2, degenerate
        res = S.cramers_v(np.array([[5, 10]]))
        self.assertTrue(math.isnan(res['cramers_v']))

    def test_single_col_table_returns_nan(self):
        res = S.cramers_v(np.array([[5], [10]]))
        self.assertTrue(math.isnan(res['cramers_v']))

    def test_result_keys_present(self):
        res = S.cramers_v(np.array([[10, 5], [5, 10]]))
        self.assertIn('cramers_v', res)
        self.assertIn('chi2', res)
        self.assertIn('p_value', res)

    def test_range_zero_to_one(self):
        table = np.array([[5, 15], [15, 5]])
        res = S.cramers_v(table)
        self.assertGreaterEqual(res['cramers_v'], 0.0)
        self.assertLessEqual(res['cramers_v'], 1.0)

    def test_chi2_non_negative(self):
        table = np.array([[3, 7], [7, 3]])
        res = S.cramers_v(table)
        self.assertGreaterEqual(res['chi2'], 0.0)


# ---------------------------------------------------------------------------
# odds_ratio_ci — additional coverage
# ---------------------------------------------------------------------------

class TestOddsRatioCIMore(unittest.TestCase):

    def test_symmetric_cells_ratio_greater_than_one(self):
        # a=10, d=10, b=2, c=2 → strong positive association
        res = S.odds_ratio_ci(10, 2, 2, 10)
        self.assertGreater(res['odds_ratio'], 1.0)

    def test_ci_brackets_odds_ratio(self):
        res = S.odds_ratio_ci(10, 2, 2, 10)
        self.assertLess(res['lo'], res['odds_ratio'])
        self.assertGreater(res['hi'], res['odds_ratio'])

    def test_all_zeros_haldane_correction_gives_one(self):
        # a=b=c=d=0 → after +0.5: (0.5*0.5)/(0.5*0.5) = 1.0
        res = S.odds_ratio_ci(0, 0, 0, 0)
        self.assertAlmostEqual(res['odds_ratio'], 1.0, places=9)

    def test_result_keys_present(self):
        res = S.odds_ratio_ci(5, 3, 2, 8)
        self.assertIn('odds_ratio', res)
        self.assertIn('lo', res)
        self.assertIn('hi', res)

    def test_ci_always_positive(self):
        # log-normal CI → both bounds > 0
        res = S.odds_ratio_ci(0, 0, 0, 0)
        self.assertGreater(res['lo'], 0.0)
        self.assertGreater(res['hi'], 0.0)

    def test_inverted_cells_reciprocal_or(self):
        # swapping a↔d and b↔c inverts the OR
        r1 = S.odds_ratio_ci(10, 2, 2, 10)
        r2 = S.odds_ratio_ci(2, 10, 10, 2)
        self.assertAlmostEqual(r1['odds_ratio'] * r2['odds_ratio'], 1.0, places=6)


# ---------------------------------------------------------------------------
# benjamini_hochberg — additional coverage
# ---------------------------------------------------------------------------

class TestBenjaminiHochbergMore(unittest.TestCase):

    def test_textbook_four_pvalues_reject_first_two(self):
        # Sorted p=[0.001, 0.008, 0.04, 0.3], alpha=0.05
        # BH threshold at rank k: p <= (k/4)*0.05
        # rank1: 0.001 <= 0.0125 YES; rank2: 0.008 <= 0.025 YES
        # rank3: 0.04 <= 0.0375 NO; rank4: 0.3 <= 0.05 NO → kmax=2
        pvals = [0.001, 0.008, 0.04, 0.3]
        res = S.benjamini_hochberg(pvals, alpha=0.05)
        self.assertEqual(res['reject'], [True, True, False, False])

    def test_textbook_four_pvalues_known_qvalues(self):
        # q_raw = p*m/rank = [0.001*4/1, 0.008*4/2, 0.04*4/3, 0.3*4/4]
        #               = [0.004, 0.016, 0.05333, 0.3]
        pvals = [0.001, 0.008, 0.04, 0.3]
        res = S.benjamini_hochberg(pvals, alpha=0.05)
        q = res['qvalues']
        self.assertAlmostEqual(q[0], 0.004, places=9)
        self.assertAlmostEqual(q[1], 0.016, places=9)
        self.assertAlmostEqual(q[2], 0.04 * 4 / 3, places=6)
        self.assertAlmostEqual(q[3], 0.3, places=9)

    def test_all_large_pvalues_no_rejections(self):
        res = S.benjamini_hochberg([0.6, 0.7, 0.8, 0.9])
        self.assertFalse(any(res['reject']))

    def test_all_tiny_pvalues_all_rejected(self):
        res = S.benjamini_hochberg([0.0001, 0.0002, 0.0003], alpha=0.05)
        self.assertTrue(all(res['reject']))

    def test_single_pvalue_below_alpha_rejected(self):
        res = S.benjamini_hochberg([0.01], alpha=0.05)
        self.assertTrue(res['reject'][0])

    def test_single_pvalue_above_alpha_not_rejected(self):
        res = S.benjamini_hochberg([0.2], alpha=0.05)
        self.assertFalse(res['reject'][0])

    def test_nan_pvalue_not_rejected(self):
        res = S.benjamini_hochberg([float('nan'), 0.001, 0.5])
        self.assertFalse(res['reject'][0])           # nan position: not rejected
        self.assertTrue(res['reject'][1])             # 0.001 rejected
        self.assertFalse(res['reject'][2])            # 0.5 not rejected

    def test_nan_pvalue_qvalue_is_nan(self):
        res = S.benjamini_hochberg([float('nan'), 0.001])
        self.assertTrue(math.isnan(res['qvalues'][0]))

    def test_all_nan_returns_no_rejections(self):
        res = S.benjamini_hochberg([float('nan'), float('nan')])
        self.assertFalse(any(res['reject']))

    def test_empty_list_returns_empty(self):
        res = S.benjamini_hochberg([])
        self.assertEqual(res['reject'], [])
        self.assertEqual(res['qvalues'], [])

    def test_output_length_matches_input(self):
        pvals = [0.01, 0.05, 0.2, 0.8]
        res = S.benjamini_hochberg(pvals)
        self.assertEqual(len(res['reject']), 4)
        self.assertEqual(len(res['qvalues']), 4)

    def test_qvalues_non_decreasing_after_monotone_step(self):
        # q-values after monotone minimum accumulate must be non-decreasing
        res = S.benjamini_hochberg([0.001, 0.01, 0.1, 0.5])
        q = res['qvalues']
        for i in range(len(q) - 1):
            self.assertLessEqual(q[i], q[i + 1] + 1e-12)

    def test_qvalues_clipped_to_1(self):
        # q-values should never exceed 1.0
        res = S.benjamini_hochberg([0.9, 0.95, 0.99])
        for q in res['qvalues']:
            self.assertLessEqual(q, 1.0)


# ---------------------------------------------------------------------------
# sign_test — additional coverage
# ---------------------------------------------------------------------------

class TestSignTestMore(unittest.TestCase):

    def test_zero_n_total_returns_nan_p(self):
        res = S.sign_test(0, 0)
        self.assertEqual(res['n_total'], 0)
        self.assertTrue(math.isnan(res['p_value']))

    def test_exact_half_gives_p_1(self):
        # 5/10 → symmetric around null (H0: p=0.5) → p = 1.0
        res = S.sign_test(5, 10)
        self.assertAlmostEqual(res['p_value'], 1.0, places=6)

    def test_extreme_skew_small_p(self):
        # 9/10 positive → strong evidence against H0
        res = S.sign_test(9, 10)
        self.assertLess(res['p_value'], 0.05)

    def test_zero_positives_small_p(self):
        # 0/10 → very unlikely under H0: p=0.5
        res = S.sign_test(0, 10)
        self.assertLess(res['p_value'], 0.01)

    def test_result_keys_present(self):
        res = S.sign_test(3, 5)
        self.assertIn('n_positive', res)
        self.assertIn('n_total', res)
        self.assertIn('p_value', res)

    def test_n_positive_n_total_preserved(self):
        res = S.sign_test(7, 12)
        self.assertEqual(res['n_positive'], 7)
        self.assertEqual(res['n_total'], 12)

    def test_p_value_in_unit_interval(self):
        res = S.sign_test(8, 10)
        p = res['p_value']
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)


# ---------------------------------------------------------------------------
# permutation_test — additional coverage
# ---------------------------------------------------------------------------

class TestPermutationTestMore(unittest.TestCase):

    def test_all_in_group_returns_nan(self):
        # No out-of-group observations → undefined
        res = S.permutation_test([1.0, 2.0, 3.0], [True, True, True])
        self.assertTrue(math.isnan(res['observed']))
        self.assertTrue(math.isnan(res['p_value']))

    def test_none_in_group_returns_nan(self):
        res = S.permutation_test([1.0, 2.0, 3.0], [False, False, False])
        self.assertEqual(res['n_group'], 0)
        self.assertTrue(math.isnan(res['p_value']))

    def test_empty_inputs_returns_nan(self):
        res = S.permutation_test([], [])
        self.assertTrue(math.isnan(res['observed']))
        self.assertTrue(math.isnan(res['p_value']))

    def test_strong_signal_gives_small_p(self):
        # Group has clearly higher values
        values = np.concatenate([np.full(15, 3.0), np.zeros(15)])
        mask = np.array([True] * 15 + [False] * 15)
        res = S.permutation_test(values, mask, n_perm=500, seed=99)
        self.assertAlmostEqual(res['observed'], 3.0, places=6)
        self.assertLess(res['p_value'], 0.01)

    def test_n_group_key(self):
        values = np.arange(10, dtype=float)
        mask = np.array([True] * 3 + [False] * 7)
        res = S.permutation_test(values, mask, n_perm=100)
        self.assertEqual(res['n_group'], 3)

    def test_within_stratum_shuffle_runs(self):
        values = np.arange(12, dtype=float)
        mask = np.array([True, False] * 6)
        strata = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        res = S.permutation_test(values, mask, strata=strata, n_perm=100, seed=7)
        self.assertGreater(res['n_perm'], 0)

    def test_result_keys_present(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([True, True, False, False])
        res = S.permutation_test(values, mask, n_perm=50)
        self.assertIn('observed', res)
        self.assertIn('p_value', res)
        self.assertIn('n_perm', res)
        self.assertIn('n_group', res)

    def test_p_value_in_unit_interval_when_valid(self):
        values = np.array([1.0, 2.0, 5.0, 6.0])
        mask = np.array([False, False, True, True])
        res = S.permutation_test(values, mask, n_perm=200, seed=3)
        if not math.isnan(res['p_value']):
            self.assertGreaterEqual(res['p_value'], 0.0)
            self.assertLessEqual(res['p_value'], 1.0)


# ---------------------------------------------------------------------------
# cluster_bootstrap_ci — additional coverage
# ---------------------------------------------------------------------------

class TestClusterBootstrapCIMore(unittest.TestCase):

    def test_empty_values_returns_nan_point(self):
        res = S.cluster_bootstrap_ci([], [])
        self.assertTrue(math.isnan(res['point']))
        self.assertEqual(res['n'], 0)
        self.assertEqual(res['n_clusters'], 0)

    def test_all_nan_values_returns_nan_point(self):
        res = S.cluster_bootstrap_ci([float('nan'), float('nan')], ['a', 'b'])
        self.assertTrue(math.isnan(res['point']))

    def test_single_cluster_no_ci(self):
        # With only one cluster CI is undefined
        res = S.cluster_bootstrap_ci([1.0, 2.0, 3.0], ['x', 'x', 'x'])
        self.assertEqual(res['n_clusters'], 1)
        self.assertTrue(math.isnan(res['lo']))
        self.assertTrue(math.isnan(res['hi']))

    def test_n_and_n_clusters_keys(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        clusters = ['A', 'A', 'B', 'B', 'C', 'C']
        res = S.cluster_bootstrap_ci(values, clusters, n_boot=100)
        self.assertEqual(res['n'], 6)
        self.assertEqual(res['n_clusters'], 3)

    def test_custom_statistic_median(self):
        # Three clusters, each constant → median is same as mean
        values = [1.0] * 3 + [2.0] * 3 + [3.0] * 3
        clusters = ['A'] * 3 + ['B'] * 3 + ['C'] * 3
        res = S.cluster_bootstrap_ci(values, clusters, statistic=np.median,
                                     n_boot=200, seed=0)
        self.assertAlmostEqual(res['point'], 2.0, places=9)

    def test_point_estimate_equals_statistic_of_all_finite(self):
        values = [1.0, 2.0, 3.0, float('nan'), 4.0, 5.0]
        clusters = ['A', 'A', 'B', 'B', 'C', 'C']
        res = S.cluster_bootstrap_ci(values, clusters, n_boot=100, seed=1)
        finite = [v for v in values if not math.isnan(v)]
        self.assertAlmostEqual(res['point'], float(np.mean(finite)), places=9)

    def test_ci_brackets_point_for_well_separated_clusters(self):
        values = list(range(50))
        clusters = [i // 5 for i in range(50)]    # 10 clusters of 5
        res = S.cluster_bootstrap_ci(values, clusters, n_boot=500, seed=42)
        self.assertLess(res['lo'], res['point'])
        self.assertGreater(res['hi'], res['point'])

    def test_result_keys_present(self):
        res = S.cluster_bootstrap_ci([1.0, 2.0], ['a', 'b'], n_boot=50)
        for k in ('point', 'lo', 'hi', 'n', 'n_clusters'):
            self.assertIn(k, res)


# ---------------------------------------------------------------------------
# mixedlm_trend — additional coverage
# ---------------------------------------------------------------------------

class TestMixedLMTrendMore(unittest.TestCase):

    def _make_trend_df(self, slope: float, n_groups: int = 6,
                       n_times: int = 4, noise: float = 0.05,
                       seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for pid in range(n_groups):
            base = rng.normal(0, 0.2)
            for t in range(n_times):
                rows.append(dict(pid=f'P{pid}', t=float(t),
                                 y=base + slope * t + rng.normal(0, noise)))
        return pd.DataFrame(rows)

    def test_positive_slope_detected(self):
        df = self._make_trend_df(slope=0.5, seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = S.mixedlm_trend(df, 'y', 't', 'pid')
        self.assertGreater(res['slope'], 0.3)
        self.assertLess(res['p_value'], 0.05)
        self.assertIn(res['method'], ('mixedlm', 'ols'))

    def test_negative_slope_detected(self):
        df = self._make_trend_df(slope=-0.5, seed=2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = S.mixedlm_trend(df, 'y', 't', 'pid')
        self.assertLess(res['slope'], -0.3)
        self.assertLess(res['p_value'], 0.05)

    def test_empty_df_returns_nan_method_none(self):
        df = pd.DataFrame({'y': [], 't': [], 'pid': []})
        res = S.mixedlm_trend(df, 'y', 't', 'pid')
        self.assertTrue(math.isnan(res['slope']))
        self.assertEqual(res['method'], 'none')

    def test_single_group_falls_back_to_ols(self):
        df = pd.DataFrame({'y': [1.0, 2.0, 3.0, 4.0],
                           't': [0.0, 1.0, 2.0, 3.0],
                           'pid': ['A', 'A', 'A', 'A']})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = S.mixedlm_trend(df, 'y', 't', 'pid')
        # OLS fallback must still yield slope ≈ 1.0
        self.assertAlmostEqual(res['slope'], 1.0, places=5)
        self.assertIn(res['method'], ('ols', 'mixedlm', 'none'))

    def test_result_keys_present(self):
        df = self._make_trend_df(slope=0.2, seed=3)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = S.mixedlm_trend(df, 'y', 't', 'pid')
        for k in ('slope', 'se', 'p_value', 'ci_lo', 'ci_hi', 'n', 'n_groups', 'method'):
            self.assertIn(k, res)

    def test_n_and_n_groups_correct(self):
        df = self._make_trend_df(slope=0.1, n_groups=4, n_times=3, seed=4)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = S.mixedlm_trend(df, 'y', 't', 'pid')
        self.assertEqual(res['n'], 4 * 3)
        self.assertEqual(res['n_groups'], 4)


# ---------------------------------------------------------------------------
# mixedlm_delta — additional coverage
# ---------------------------------------------------------------------------

class TestMixedLMDeltaMore(unittest.TestCase):

    def _make_delta_df(self, r_effect: float = 1.0,
                       n_participants: int = 6) -> pd.DataFrame:
        rows = []
        for pid in range(n_participants):
            for beh in ['P', 'U', 'R']:
                rows.append({'participant_id': f'P{pid}', 'behavior': beh,
                             'delta_prog': (r_effect if beh == 'R' else 0.0)})
        return pd.DataFrame(rows)

    def test_returns_fitted_result_object(self):
        df = self._make_delta_df(r_effect=1.0, n_participants=6)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = S.mixedlm_delta(df)
        self.assertIsNotNone(res)

    def test_r_coefficient_close_to_known_effect(self):
        # R always has delta_prog=1.0, P and U always 0.0 → C(behavior)[T.R] ≈ 1.0
        df = self._make_delta_df(r_effect=1.0, n_participants=6)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = S.mixedlm_delta(df)
        self.assertIsNotNone(res)
        coeff = res.fe_params.get('C(behavior)[T.R]')
        self.assertIsNotNone(coeff)
        self.assertAlmostEqual(float(coeff), 1.0, places=4)

    def test_single_group_returns_none(self):
        # Only one participant → n_groups < 2 → None
        df = pd.DataFrame({'participant_id': ['A', 'A', 'A'],
                           'behavior': ['P', 'U', 'R'],
                           'delta_prog': [0.0, 0.0, 1.0]})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = S.mixedlm_delta(df)
        self.assertIsNone(res)

    def test_single_row_returns_none(self):
        df = pd.DataFrame({'participant_id': ['A'],
                           'behavior': ['P'],
                           'delta_prog': [1.0]})
        res = S.mixedlm_delta(df)
        self.assertIsNone(res)

    def test_empty_df_returns_none(self):
        df = pd.DataFrame({'participant_id': [], 'behavior': [], 'delta_prog': []})
        res = S.mixedlm_delta(df)
        self.assertIsNone(res)

    def test_fe_params_has_intercept(self):
        df = self._make_delta_df(n_participants=5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = S.mixedlm_delta(df)
        if res is not None:
            self.assertIn('Intercept', res.fe_params)

    def test_conf_int_accessible(self):
        df = self._make_delta_df(n_participants=6)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = S.mixedlm_delta(df)
        if res is not None:
            ci = res.conf_int()
            self.assertIn('C(behavior)[T.R]', ci.index)


if __name__ == '__main__':
    unittest.main()
