"""
tests/unit/test_mechanism_model.py
----------------------------------
Hermetic unit tests for the re-centered mechanism estimator (masterplan §3).
No network, no model downloads.

Covers:
  - e_value monotonicity (≥1, monotone in |SMD|, symmetric)
  - fit_adjacency_interaction RECOVERS a planted FROM_stage×move interaction
    (ordinal LR + Gaussian-mixed CI + earns-its-place CV) and does NOT on null data
  - sensitivity_bounds surfaces the planted cell with the largest |SMD| + a finite E-value
  - purer_noise_robustness returns a bounded Spearman rank-stability
  - fit_trajectory within/between split fits on planted consolidation data
  - BACKWARD-COMPAT: mechanism disabled ⇒ mechanism.py report is the legacy output
  - PipelineConfig.mechanism field exists and round-trips
"""

import json
import os
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from analysis import mechanism_model as MM
from analysis import stats as S

warnings.filterwarnings('ignore')   # statsmodels convergence/singular notices are expected at small n


_FRAMEWORK = {
    0: {'key': 'vigilance', 'short_name': 'Vigilance'},
    1: {'key': 'avoidance', 'short_name': 'Avoidance'},
    2: {'key': 'attn_reg', 'short_name': 'AttnReg'},
    3: {'key': 'metacog', 'short_name': 'Metacog'},
    4: {'key': 'reappraisal', 'short_name': 'Reappraisal'},
}

_MOVES = ['P(0)', 'U(1)', 'R(2)', 'E(3)']


def _make_triples(planted: bool, seed: int = 7, n_participants: int = 24, n_sessions: int = 7):
    """Synthetic FROM→CUE→TO triples. When ``planted``, move 'U(1)' helps a lot at
    from_stage 1 and hurts at from_stage 3 — a genuine FROM×move interaction."""
    rng = np.random.default_rng(seed)
    blocks = []
    pids = [f'P{i}' for i in range(n_participants)]
    for s in range(1, n_sessions + 1):
        for pid in pids:
            fs = int(rng.integers(0, 4))
            mv = _MOVES[int(rng.integers(0, 4))]
            eff = 0.0
            if planted:
                if mv == 'U(1)' and fs == 1:
                    eff = 1.3
                if mv == 'U(1)' and fs == 3:
                    eff = -1.1
            d = eff + rng.normal(0, 0.3)
            ts = int(np.clip(round(fs + d), 0, 4))
            blocks.append(dict(participant_id=pid, session_id=f's{s}', from_stage=fs,
                               to_stage=ts, dominant_purer=mv, delta_prog=d))
    return blocks


# ---------------------------------------------------------------------------
# E-value
# ---------------------------------------------------------------------------

class TestEValue(unittest.TestCase):
    def test_e_value_at_null_is_one(self):
        self.assertAlmostEqual(S.e_value(1.0), 1.0, places=6)

    def test_e_value_at_least_one(self):
        for rr in (0.1, 0.5, 1.0, 1.5, 3.0, 10.0):
            self.assertGreaterEqual(S.e_value(rr), 1.0 - 1e-9)

    def test_e_value_symmetric_under_inversion(self):
        self.assertAlmostEqual(S.e_value(2.0), S.e_value(0.5), places=6)

    def test_e_value_monotone_in_abs_smd(self):
        # Larger |SMD| ⇒ larger RR ⇒ larger E-value.
        evals = [S.e_value(S.smd_to_risk_ratio(d)) for d in (0.0, 0.2, 0.5, 1.0, 2.0)]
        for a, b in zip(evals, evals[1:]):
            self.assertLessEqual(a, b)
        self.assertGreater(evals[-1], evals[0])

    def test_e_value_nonfinite_returns_nan(self):
        self.assertTrue(np.isnan(S.e_value(0.0)))
        self.assertTrue(np.isnan(S.e_value(float('nan'))))


class TestEValueCiLimit(unittest.TestCase):
    """P0-1: E-value of the CI limit nearest the null (VanderWeele & Ding 2017)."""

    def test_ci_spanning_zero_is_one(self):
        # A 95% CI that straddles 0 ⇒ no robustness ⇒ CI-limit E-value = 1.0.
        self.assertEqual(S.e_value_ci_limit(-0.2, 0.5), 1.0)
        self.assertEqual(S.e_value_ci_limit(0.0, 0.5), 1.0)
        self.assertEqual(S.e_value_ci_limit(-0.5, 0.0), 1.0)

    def test_positive_ci_uses_lower_limit(self):
        # Both limits positive ⇒ E-value of the smaller (null-side) limit.
        self.assertAlmostEqual(S.e_value_ci_limit(0.3, 1.2),
                               S.e_value(S.smd_to_risk_ratio(0.3)), places=6)

    def test_negative_ci_uses_upper_limit(self):
        # Both limits negative ⇒ E-value of the limit nearer 0 (the larger/upper one).
        self.assertAlmostEqual(S.e_value_ci_limit(-1.2, -0.3),
                               S.e_value(S.smd_to_risk_ratio(-0.3)), places=6)

    def test_ci_limit_never_exceeds_point(self):
        # The CI-limit E-value (null-side) is ≤ the point E-value (a robustness floor).
        point = S.e_value(S.smd_to_risk_ratio(0.9))
        self.assertLessEqual(S.e_value_ci_limit(0.4, 1.4), point)

    def test_nonfinite_limits_return_nan(self):
        self.assertTrue(np.isnan(S.e_value_ci_limit(float('nan'), 0.5)))
        self.assertTrue(np.isnan(S.e_value_ci_limit(0.1, float('inf'))))


# ---------------------------------------------------------------------------
# Interaction model — planted vs null
# ---------------------------------------------------------------------------

# Fast config: skip the (slow) cluster-bootstrap LR refits except where explicitly tested.
_FAST = dict(lr_cluster_bootstrap_n_boot=0)


class TestInteractionModelPlanted(unittest.TestCase):
    def setUp(self):
        self.D = MM.build_design_frame(_make_triples(planted=True))
        self.adj = MM.fit_adjacency_interaction(self.D, MM.MechanismModelConfig(**_FAST))

    def test_ordinal_lr_detects_interaction(self):
        olr = self.adj['ordinal_lr']
        self.assertEqual(olr['status'], 'ok')
        self.assertLess(olr['p_value'], 0.05)

    def test_gaussian_interaction_ci_excludes_zero(self):
        g = self.adj['gaussian_interaction']
        self.assertEqual(g['method'], 'mixedlm')
        self.assertGreaterEqual(g['n_ci_excludes_0'], 1)

    def test_interaction_earns_its_place(self):
        eip = self.adj['earns_its_place']
        self.assertEqual(eip['status'], 'ok')
        self.assertTrue(eip['interaction_earns_place'])
        self.assertLess(eip['interaction_delta_logloss'], 0.0)

    def test_bayesian_not_requested_by_default(self):
        # Default estimator is 'frequentist' → the Bayesian arm is not run in-process.
        self.assertFalse(self.adj['bayesian']['ok'])
        self.assertEqual(self.adj['bayesian']['status'], 'not_requested')

    def test_cluster_bootstrap_lr_p_populated_when_enabled(self):
        # Gate 2: with the cluster-bootstrap enabled (small n_boot for speed) the ordinal_lr
        # dict carries BOTH the naive in-sample p and a cluster-bootstrap p, clearly labeled.
        adj = MM.fit_adjacency_interaction(
            self.D, MM.MechanismModelConfig(lr_cluster_bootstrap_n_boot=40))
        olr = adj['ordinal_lr']
        self.assertEqual(olr['status'], 'ok')
        self.assertIn('p_value_naive_label', olr)
        self.assertIn('NOT cluster-robust', olr['p_value_naive_label'])
        # Either a finite cluster-bootstrap p, or a clean 'unavailable' status — never silent.
        self.assertIn('cluster_bootstrap_status', olr)
        if olr['cluster_bootstrap_status'] == 'ok':
            self.assertIsNotNone(olr['p_value_cluster_bootstrap'])
            self.assertGreater(olr['p_value_cluster_bootstrap'], 0.0)
            self.assertLessEqual(olr['p_value_cluster_bootstrap'], 1.0)


class TestInteractionModelNull(unittest.TestCase):
    def setUp(self):
        self.D = MM.build_design_frame(_make_triples(planted=False))
        self.adj = MM.fit_adjacency_interaction(self.D, MM.MechanismModelConfig(**_FAST))

    def test_ordinal_lr_not_significant(self):
        olr = self.adj['ordinal_lr']
        self.assertEqual(olr['status'], 'ok')
        self.assertGreater(olr['p_value'], 0.05)

    def test_interaction_does_not_earn_its_place(self):
        # On null data the extra interaction params overfit → worse held-out log-loss.
        eip = self.adj['earns_its_place']
        self.assertEqual(eip['status'], 'ok')
        self.assertFalse(eip['interaction_earns_place'])


class TestBayesianOptInDegrades(unittest.TestCase):
    def test_bayesian_arm_degrades_gracefully(self):
        # bambi requires numpy>=2.0 (conflicts with the pinned transformers) → absent here.
        D = MM.build_design_frame(_make_triples(planted=True))
        out = MM.fit_adjacency_interaction(D, MM.MechanismModelConfig(estimator='bayesian', **_FAST))
        bay = out['bayesian']
        self.assertFalse(bay['ok'])
        # Clean degradation, not a crash; the note points to the isolated-env workaround.
        self.assertIn(bay['status'], ('bambi_unavailable',) )
        self.assertIn('isolated env', bay['note'])


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

class TestSensitivityBounds(unittest.TestCase):
    def test_planted_cell_has_largest_smd_and_finite_evalue(self):
        D = MM.build_design_frame(_make_triples(planted=True))
        sens = MM.sensitivity_bounds(D, min_n=4, n_boot=300)
        self.assertGreater(sens['n_cells'], 0)
        top = sens['cells'][0]   # sorted by CI-limit E-value, then |SMD| desc
        self.assertEqual(top['move'], 'U(1)')          # the planted move
        self.assertIn(top['from_stage'], (1, 3))       # planted at from-stage 1 (+) and 3 (−)
        self.assertIsNotNone(top['e_value'])
        self.assertGreaterEqual(top['e_value'], 1.0)

    def test_ci_limit_evalue_columns_present_and_bounded(self):
        # P0-1: every cell carries an SMD CI + a CI-limit E-value ≤ the point E-value.
        D = MM.build_design_frame(_make_triples(planted=True))
        sens = MM.sensitivity_bounds(D, min_n=4, n_boot=300)
        for c in sens['cells']:
            self.assertIn('e_value_ci_limit', c)
            self.assertIn('smd_ci_lo', c)
            self.assertIn('smd_ci_hi', c)
            if c['e_value'] is not None and c['e_value_ci_limit'] is not None:
                self.assertGreaterEqual(c['e_value_ci_limit'], 1.0)
                self.assertLessEqual(c['e_value_ci_limit'], c['e_value'] + 1e-6)

    def test_planted_strong_cell_ci_excludes_null(self):
        # The strongly-planted U(1) cell should have a CI-limit E-value > 1 (interval off 0).
        D = MM.build_design_frame(_make_triples(planted=True))
        sens = MM.sensitivity_bounds(D, min_n=4, n_boot=500)
        top = sens['cells'][0]
        self.assertEqual(top['move'], 'U(1)')
        self.assertGreater(top['e_value_ci_limit'], 1.0)

    def test_null_data_ci_limit_evalues_collapse_to_one(self):
        # P0-1: on null data no cell association survives its interval ⇒ CI-limit E-value = 1.0.
        D = MM.build_design_frame(_make_triples(planted=False))
        sens = MM.sensitivity_bounds(D, min_n=4, n_boot=500)
        vals = [c['e_value_ci_limit'] for c in sens['cells'] if c['e_value_ci_limit'] is not None]
        self.assertTrue(vals)                          # cells exist
        self.assertTrue(all(v == 1.0 for v in vals))   # none robust to its own CI


# ---------------------------------------------------------------------------
# PURER-noise robustness
# ---------------------------------------------------------------------------

class TestNoiseRobustness(unittest.TestCase):
    def test_rank_stability_is_bounded(self):
        D = MM.build_design_frame(_make_triples(planted=True))
        nz = MM.purer_noise_robustness(D, MM.MechanismModelConfig(), k=50)
        self.assertEqual(nz['status'], 'ok')
        self.assertEqual(nz['rate'], 0.30)             # default disagreement rate
        self.assertTrue(-1.0 <= nz['mean_spearman'] <= 1.0)

    def test_custom_disagreement_rate(self):
        D = MM.build_design_frame(_make_triples(planted=True))
        nz = MM.purer_noise_robustness(D, MM.MechanismModelConfig(purer_disagreement_rate=0.5), k=30)
        self.assertEqual(nz['rate'], 0.5)


# ---------------------------------------------------------------------------
# Trajectory within/between
# ---------------------------------------------------------------------------

class TestTrajectory(unittest.TestCase):
    def _planted_consolidation(self, seed=3, n_participants=20, n_sessions=6):
        rng = np.random.default_rng(seed)
        pids = [f'P{i}' for i in range(n_participants)]
        prows, blocks = [], []
        for pid in pids:
            icpt = rng.normal(0, 0.4)
            for s in range(1, n_sessions + 1):
                expo = int(rng.integers(0, 4))
                prows.append(dict(participant_id=pid, session_id=f's{s}', session_number=s,
                                  progression_coord=1.5 + 0.25 * s + 0.15 * expo + icpt + rng.normal(0, 0.2)))
                for _ in range(expo):
                    blocks.append(dict(participant_id=pid, session_id=f's{s}', from_stage=1,
                                       to_stage=2, dominant_purer='U(1)', delta_prog=0.3))
        return pd.DataFrame(prows), MM.build_design_frame(blocks)

    def test_within_between_fit(self):
        pdf, D = self._planted_consolidation()
        tj = MM.fit_trajectory(pdf, D, MM.MechanismModelConfig())
        self.assertEqual(tj['status'], 'ok')
        self.assertIsNotNone(tj['within'])
        self.assertIsNotNone(tj['between'])
        self.assertIn('estimate', tj['within'])

    def test_disabled_returns_status(self):
        pdf, D = self._planted_consolidation()
        tj = MM.fit_trajectory(pdf, D, MM.MechanismModelConfig(trajectory=False))
        self.assertEqual(tj['status'], 'disabled')


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TestRunMechanismModels(unittest.TestCase):
    def test_writes_csvs_and_available(self):
        tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmp, '03_analysis_data', 'mechanism'), exist_ok=True)
        blocks = _make_triples(planted=True)
        pdf = pd.DataFrame([
            dict(participant_id=f'P{i % 24}', session_id=f's{s}', session_number=s,
                 progression_coord=float(2 + 0.1 * s))
            for i in range(24) for s in range(1, 8)
        ])
        res = MM.run_mechanism_models(blocks, pdf, tmp, MM.MechanismModelConfig(**_FAST))
        self.assertTrue(res['available'])
        # At least the sensitivity + CV CSVs are written for planted data.
        names = {os.path.basename(p) for p in res['files_written']}
        self.assertIn('mechanism_sensitivity_evalues.csv', names)
        self.assertIn('mechanism_interaction_cv.csv', names)

    def test_disabled_returns_unavailable(self):
        tmp = tempfile.mkdtemp()
        res = MM.run_mechanism_models(_make_triples(True), pd.DataFrame(), tmp,
                                      MM.MechanismModelConfig(enabled=False))
        self.assertFalse(res['available'])
        self.assertEqual(res['status'], 'disabled')


# ---------------------------------------------------------------------------
# Backward-compat: mechanism.py report unchanged when the estimator is OFF
# ---------------------------------------------------------------------------

_LEGACY_SECTION1 = "1. ΔPROGRESSION BY THERAPIST BEHAVIOUR × FROM-STAGE (conditioned, inferential)"
_PRIMARY_SENTINEL = "PRIMARY ESTIMATOR — STAGE-MODERATED THERAPIST EFFECT"


def _synthetic_all_df(n_participants=6):
    rows = []
    for p in range(n_participants):
        for s in range(1, 4):
            sess = f'c1s{s}'
            t = 0
            seq = [('participant', 1), ('therapist', 1), ('participant', 2),
                   ('therapist', 2), ('participant', 3)]
            for j, (spk, stage) in enumerate(seq):
                t += 1000
                is_p = spk == 'participant'
                rows.append(dict(
                    segment_id=f'{sess}_p{p}_{j}', participant_id=f'P{p}', session_id=sess,
                    session_number=s, segment_index=j, speaker=spk, word_count=40,
                    text=f'{spk} text', start_time_ms=t, end_time_ms=t + 800,
                    final_label=(stage if is_p else np.nan),
                    primary_stage=(stage if is_p else np.nan), secondary_stage=np.nan,
                    llm_confidence_primary=0.8, llm_confidence_secondary=np.nan,
                    rater_votes=(json.dumps([{'stage': stage, 'confidence': 0.9}]) if is_p else None),
                    codebook_labels_ensemble=[], purer_primary=(1 if not is_p else np.nan),
                ))
    return pd.DataFrame(rows)


class TestBackwardCompat(unittest.TestCase):
    """mechanism.enabled=False ⇒ the legacy estimator-off report (no new lead section)."""

    def _run(self, cfg):
        from analysis.superposition import attach_superposition
        from analysis.mechanism import run_mechanism_analysis
        from process.config import SuperpositionConfig
        from process import output_paths as _paths
        tmp = tempfile.mkdtemp()
        df_all = _synthetic_all_df()
        attach_superposition(df_all, tmp, config=SuperpositionConfig())
        df = df_all[df_all['speaker'] == 'participant'].copy()
        run_mechanism_analysis(df, df_all, tmp, _FRAMEWORK, config=cfg)
        return open(os.path.join(_paths.reports_mechanism_dir(tmp), 'mechanism.txt'),
                    encoding='utf-8').read()

    def test_disabled_preserves_legacy_report(self):
        txt = self._run(MM.MechanismModelConfig(enabled=False))
        self.assertNotIn(_PRIMARY_SENTINEL, txt)
        self.assertIn(_LEGACY_SECTION1, txt)

    def test_enabled_leads_with_interaction(self):
        txt = self._run(MM.MechanismModelConfig(enabled=True, **_FAST))
        self.assertIn(_PRIMARY_SENTINEL, txt)
        self.assertIn('DESCRIPTIVE COMPANION', txt)        # the additive table is demoted
        self.assertNotIn(_LEGACY_SECTION1, txt)            # legacy header replaced


# ---------------------------------------------------------------------------
# Config field
# ---------------------------------------------------------------------------

class TestPipelineConfigMechanism(unittest.TestCase):
    def test_field_exists_and_defaults(self):
        from process.config import PipelineConfig
        c = PipelineConfig()
        self.assertTrue(hasattr(c, 'mechanism'))
        self.assertEqual(c.mechanism.estimator, 'frequentist')   # safe in-process default
        self.assertTrue(c.mechanism.enabled)

    def test_serialization_round_trip(self):
        from process.config import PipelineConfig
        c = PipelineConfig()
        j = c.to_json()
        self.assertIn('mechanism', j)
        j['mechanism']['estimator'] = 'both'
        c2 = PipelineConfig.from_json(j)
        self.assertEqual(c2.mechanism.estimator, 'both')
        self.assertEqual(type(c2.mechanism).__name__, 'MechanismModelConfig')

    def test_legacy_config_without_mechanism_loads(self):
        from process.config import PipelineConfig
        j = PipelineConfig().to_json()
        del j['mechanism']
        c = PipelineConfig.from_json(j)   # must not raise
        self.assertEqual(c.mechanism.estimator, 'frequentist')


if __name__ == '__main__':
    unittest.main()
