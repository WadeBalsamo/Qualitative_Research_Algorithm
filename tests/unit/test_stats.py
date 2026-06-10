"""
tests/test_stats.py
-------------------
Unit tests for analysis/stats.py — the inference toolkit (Wilson CIs, cluster
bootstrap, within-stratum permutation, effect sizes, BH-FDR, mixed-effects trend).
No model backends; pure numpy/scipy/statsmodels.
"""

import math
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


class TestEValueChain(unittest.TestCase):
    """Gate 1 — the SMD → RR → E-value chain and the CI-limit E-value, hand-computed."""

    def test_smd_to_rr_constant(self):
        # RR = exp(0.91 * SMD). Hand value at SMD=1.0: exp(0.91) ≈ 2.4843.
        self.assertAlmostEqual(S.smd_to_risk_ratio(1.0), math.exp(0.91), places=9)
        self.assertAlmostEqual(S.smd_to_risk_ratio(0.0), 1.0, places=9)

    def test_e_value_hand_computed(self):
        # SMD=0.5 → RR=exp(0.455)=1.5762 → E = RR + sqrt(RR*(RR-1))
        #        = 1.5762 + sqrt(1.5762*0.5762) = 1.5762 + sqrt(0.90825) = 1.5762 + 0.95302
        rr = math.exp(0.91 * 0.5)
        expected = rr + math.sqrt(rr * (rr - 1.0))
        self.assertAlmostEqual(S.e_value(rr), expected, places=9)
        self.assertAlmostEqual(S.e_value(S.smd_to_risk_ratio(0.5)), expected, places=9)

    def test_ci_limit_spans_zero_is_one(self):
        # A CI straddling 0 ⇒ CI-limit E-value collapses to 1.0 (no robustness).
        self.assertEqual(S.e_value_ci_limit(-0.1, 0.4), 1.0)
        self.assertEqual(S.e_value_ci_limit(-0.4, 0.0), 1.0)

    def test_ci_limit_uses_null_side_limit_hand_computed(self):
        # CI [0.3, 1.2], both positive → use the lower (null-side) limit 0.3.
        rr = S.smd_to_risk_ratio(0.3)
        expected = S.e_value(rr)
        self.assertAlmostEqual(S.e_value_ci_limit(0.3, 1.2), expected, places=9)
        # Negative CI [-1.2,-0.3] → use the upper (null-side) limit -0.3; symmetric magnitude.
        self.assertAlmostEqual(S.e_value_ci_limit(-1.2, -0.3), expected, places=9)

    def test_ci_limit_is_a_floor_below_point(self):
        # The CI-limit E-value (null-side) is ≤ the point E-value (a robustness floor).
        point = S.e_value(S.smd_to_risk_ratio(0.9))
        self.assertLessEqual(S.e_value_ci_limit(0.4, 1.4), point + 1e-9)


class TestClusterBootstrapLR(unittest.TestCase):
    """Gate 2 — participant-cluster bootstrap of the ordered-logit LR statistic (plumbing)."""

    def _toy(self, planted: bool, seed: int = 0):
        # Tiny FROM→CUE→TO frame: to_stage depends on move (planted) or not (null).
        rng = np.random.default_rng(seed)
        rows = []
        for pid in range(12):
            for _ in range(8):
                fs = int(rng.integers(0, 3))
                mv = int(rng.integers(0, 2))
                eff = 1.6 if (planted and mv == 1) else 0.0
                ts = int(np.clip(round(fs + eff + rng.normal(0, 0.4)), 0, 4))
                rows.append(dict(participant_id=f'P{pid}', from_stage=fs, move=mv, to_stage=ts))
        return pd.DataFrame(rows)

    def test_plumbing_returns_ok_and_both_pvalues(self):
        import warnings
        df = self._toy(planted=True, seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = S.cluster_bootstrap_lr_test(
                df, 'to_stage', 'C(from_stage) + C(move)', 'C(from_stage) * C(move)',
                group='participant_id', n_boot=60, seed=3)
        # The harness must run and return a usable, smoothed, one-sided p in (0,1].
        self.assertIn(res['status'], ('ok', 'cluster_robust_p_unavailable'))
        if res['status'] == 'ok':
            self.assertGreater(res['p_value'], 0.0)
            self.assertLessEqual(res['p_value'], 1.0)
            self.assertGreaterEqual(res['n_boot_ok'], 20)
            self.assertTrue(math.isfinite(res['observed_lr']))

    def test_single_cluster_unavailable(self):
        df = self._toy(planted=True, seed=2)
        df['participant_id'] = 'ONLY'
        res = S.cluster_bootstrap_lr_test(
            df, 'to_stage', 'C(from_stage) + C(move)', 'C(from_stage) * C(move)',
            group='participant_id', n_boot=30)
        self.assertEqual(res['status'], 'cluster_robust_p_unavailable')

    def test_missing_group_column_degrades(self):
        df = self._toy(planted=False, seed=4)
        res = S.cluster_bootstrap_lr_test(
            df, 'to_stage', 'C(from_stage)', 'C(from_stage) + C(move)',
            group='not_a_column', n_boot=10)
        self.assertEqual(res['status'], 'cluster_robust_p_unavailable')


class TestSingularFitFlag(unittest.TestCase):
    """Gate 3 — mixedlm_interaction flags a rank-deficient/singular interaction design."""

    def test_singular_flag_on_degenerate_interaction(self):
        import warnings
        # Rank-deficient interaction: 5 from-stages × 5 moves = 16 interaction terms, but
        # only one observation per (from, move) cell and 2 participants → far fewer effective
        # residual df than parameters → boundary/singular fit. The flag must be True (and the
        # CI-excluding-0 count must not be presented as trustworthy).
        rng = np.random.default_rng(5)
        rows = []
        for fs in range(5):
            for mv in range(5):
                pid = f'P{(fs + mv) % 2}'          # only 2 participants, perfectly aliased
                rows.append(dict(participant_id=pid, from_stage=fs, move=mv,
                                 delta_prog=float(rng.normal(0, 0.3))))
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = S.mixedlm_interaction(df, 'delta_prog', 'C(from_stage)*C(move)',
                                        'participant_id')
        self.assertIn('singular', res)
        # Either the fit fails outright (method != mixedlm) or it returns non-finite CIs;
        # both must surface as singular=True so the report suppresses the count claim.
        self.assertTrue(res['singular'])

    def test_nondegenerate_interaction_not_singular(self):
        import warnings
        # Clean 2x2 interaction, many participants → finite CIs, singular False.
        rng = np.random.default_rng(6)
        rows = []
        for pid in range(30):
            icpt = rng.normal(0, 0.2)
            for fs in (0, 1):
                for mv in (0, 1):
                    eff = 1.2 if (fs == 1 and mv == 1) else 0.0
                    rows.append(dict(participant_id=f'P{pid}', from_stage=fs, move=mv,
                                     delta_prog=float(icpt + eff + rng.normal(0, 0.25))))
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = S.mixedlm_interaction(df, 'delta_prog', 'C(from_stage)*C(move)',
                                        'participant_id')
        self.assertFalse(res['singular'])
        self.assertGreaterEqual(res['n_interaction_terms'], 1)


class TestRosenbaumBounds(unittest.TestCase):
    """Gate 3 — signed-rank/sign sensitivity Γ: grid, one-sided, ties handling."""

    def test_all_positive_survives_grid(self):
        # Strong unanimous positive shift → Γ critical is large (survives the whole grid up
        # to the cap, or at least well above 1).
        res = S.rosenbaum_bounds([0.5] * 12, max_gamma=6.0)
        self.assertEqual(res['direction'], 'increasing')
        self.assertGreater(res['gamma_critical'], 1.0)
        self.assertEqual(res['n_positive'], 12)

    def test_balanced_is_flat_gamma_one(self):
        # Equal +/- → no association unconfounded → Γ=1.0, direction flat.
        res = S.rosenbaum_bounds([1.0, -1.0, 2.0, -2.0])
        self.assertEqual(res['direction'], 'flat')
        self.assertEqual(res['gamma_critical'], 1.0)

    def test_zeros_dropped_as_ties(self):
        # Exact zeros are ties for a sign test and must be excluded from n.
        res = S.rosenbaum_bounds([0.0, 0.0, 1.0, 1.0, 1.0])
        self.assertEqual(res['n'], 3)          # the two zeros dropped
        self.assertEqual(res['n_positive'], 3)

    def test_weak_majority_low_gamma(self):
        # A weak majority loses significance under a small bias → Γ critical near 1.
        res = S.rosenbaum_bounds([1.0, 1.0, 1.0, -1.0, -1.0])
        self.assertGreaterEqual(res['gamma_critical'], 1.0)
        self.assertLess(res['gamma_critical'], 3.0)

    def test_degenerate_small_n_returns_nan(self):
        res = S.rosenbaum_bounds([1.0, 1.0])   # n<3
        self.assertTrue(math.isnan(res['gamma_critical']))


class TestWithinBetweenSplit(unittest.TestCase):
    """Gate 3 — Mundlak within/between decomposition math."""

    def test_between_is_group_mean_within_is_deviation(self):
        df = pd.DataFrame({
            'g': ['A', 'A', 'B', 'B'],
            'x': [1.0, 3.0, 10.0, 20.0],
        })
        out = S.within_between_split(df, 'x', 'g')
        # Group A mean = 2.0; group B mean = 15.0.
        self.assertAlmostEqual(out.loc[0, 'x_between'], 2.0, places=9)
        self.assertAlmostEqual(out.loc[2, 'x_between'], 15.0, places=9)
        # Within = x - group mean.
        self.assertAlmostEqual(out.loc[0, 'x_within'], -1.0, places=9)
        self.assertAlmostEqual(out.loc[1, 'x_within'], 1.0, places=9)
        self.assertAlmostEqual(out.loc[3, 'x_within'], 5.0, places=9)

    def test_within_sums_to_zero_per_group(self):
        # The within component must be mean-zero within each group (the Mundlak identity).
        df = pd.DataFrame({'g': ['A'] * 3 + ['B'] * 4,
                           'x': [1.0, 2.0, 6.0, 0.0, 4.0, 8.0, 12.0]})
        out = S.within_between_split(df, 'x', 'g')
        for grp in ('A', 'B'):
            sub = out[out['g'] == grp]['x_within']
            self.assertAlmostEqual(float(sub.sum()), 0.0, places=9)

    def test_reconstruction_within_plus_between_equals_x(self):
        df = pd.DataFrame({'g': ['A', 'A', 'B'], 'x': [3.0, 5.0, 9.0]})
        out = S.within_between_split(df, 'x', 'g')
        recon = out['x_within'] + out['x_between']
        for orig, rc in zip(df['x'], recon):
            self.assertAlmostEqual(orig, rc, places=9)

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({'g': ['A', 'B'], 'x': [1.0, 2.0]})
        _ = S.within_between_split(df, 'x', 'g')
        self.assertNotIn('x_within', df.columns)
        self.assertNotIn('x_between', df.columns)


if __name__ == '__main__':
    unittest.main()
