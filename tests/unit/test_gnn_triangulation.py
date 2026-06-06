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
from gnn_layer import triangulation as tri


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


# ===========================================================================
# PRE-REGISTERED cue→transition triangulation (gnn_layer/influence.py §1A/§9)
# ---------------------------------------------------------------------------
# These exercise the mechanism-vs-GNN counterfactual triangulation metric — a
# DIFFERENT object from the GNN↔LLM↔human construct triangulation above. Purely
# synthetic / hermetic: no model weights, no data/, no network. They validate the
# join against mechanism.py's real CSV schema and the §1A success arithmetic.
# ===========================================================================

from unittest import mock

from gnn_layer import influence as INF
from process import output_paths as _paths


def _mech_df(cells, include_motif=True):
    """Synthetic mechanism_delta_progression.csv rows matching mechanism.py's real schema.

    ``cells``: {(from_stage:int, move:int): (mean_delta_prog:float, fdr_significant:bool)}.
    """
    rows = []
    for (s, m), (d, f) in cells.items():
        rows.append({
            'grouping': 'purer', 'from_stage': s, 'from_stage_name': f'stage{s}',
            'behavior': f'{INF.PURER_NAMES.get(m, m)}({m})',
            'n': 6, 'n_participants': 3,
            'mean_delta_prog': d, 'sd_delta_prog': 0.1,
            'ci_lo': (d - 0.1 if d == d else float('nan')),
            'ci_hi': (d + 0.1 if d == d else float('nan')),
            'perm_p': 0.02, 'n_progress': 3, 'n_stabilize': 2, 'n_regress': 1,
            'mean_from_entropy': 0.5, 'fdr_q': 0.03, 'fdr_significant': bool(f),
        })
    if include_motif:                     # motif rows must be EXCLUDED by the purer filter
        rows.append({
            'grouping': 'motif', 'from_stage': 1, 'from_stage_name': 'stage1',
            'behavior': 7, 'n': 4, 'n_participants': 2, 'mean_delta_prog': 0.9,
            'sd_delta_prog': 0.1, 'ci_lo': 0.8, 'ci_hi': 1.0, 'perm_p': 0.5,
            'n_progress': 2, 'n_stabilize': 1, 'n_regress': 1, 'mean_from_entropy': 0.4,
            'fdr_q': 0.5, 'fdr_significant': False,
        })
    return pd.DataFrame(rows)


def _influence_result(gnn_cells, n_participants=5):
    """Synthetic GNN influence dict: per-cell means + block rows (identical per participant).

    Block rows are identical across participants per cell, so the participant-clustered
    bootstrap reproduces the per-cell means on every resample → a deterministic, tight CI.
    """
    rows, per_stage_move = [], []
    for (s, m), val in gnn_cells.items():
        per_stage_move.append({'from_stage': s, 'move': m,
                               'mean_influence': float(val), 'n_blocks': n_participants})
        for pi in range(n_participants):
            rows.append({'from_stage': s, 'move': m, 'influence': float(val),
                         'participant_id': f'p{pi}'})
    return {'per_stage_move': per_stage_move, 'rows': rows}


def _observed(spec):
    """{(s,m): (mean_delta, fdr_bool)} → the parsed-observed mapping shape."""
    return {k: {'mean_delta': float(d), 'fdr_significant': bool(f)}
            for k, (d, f) in spec.items()}


# Shared cell geometry: obs and GNN rank-identical → Spearman ρ = 1.0.
_OBS_CONVERGE = {(0, 0): (0.10, False), (1, 1): (0.40, True),
                 (2, 2): (-0.20, True), (3, 3): (0.25, False)}
_GNN_CONVERGE = {(0, 0): 0.05, (1, 1): 0.30, (2, 2): -0.15, (3, 3): 0.20}


class TestMechanismObservedParse(unittest.TestCase):
    """The (from_stage × move) join against mechanism.py's real CSV schema."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_parse_stage_move_keys_signed_effect_and_fdr_flag(self):
        cells = {(1, 1): (0.4, True), (2, 2): (-0.2, True),
                 (0, 0): (0.1, False), (4, 4): (float('nan'), False)}
        mdir = _paths.mechanism_dir(self.tmp)
        os.makedirs(mdir, exist_ok=True)
        _mech_df(cells).to_csv(os.path.join(mdir, 'mechanism_delta_progression.csv'), index=False)

        obs = INF._observed_per_stage_move(self.tmp)
        # Keyed on (from_stage, move); signed effect = mean_delta_prog; flag = fdr_significant.
        self.assertEqual(set(obs), {(1, 1), (2, 2), (0, 0)})
        self.assertAlmostEqual(obs[(1, 1)]['mean_delta'], 0.4, places=6)
        self.assertAlmostEqual(obs[(2, 2)]['mean_delta'], -0.2, places=6)
        self.assertTrue(obs[(1, 1)]['fdr_significant'])
        self.assertTrue(obs[(2, 2)]['fdr_significant'])
        self.assertFalse(obs[(0, 0)]['fdr_significant'])
        self.assertNotIn((4, 4), obs)              # NaN effect skipped
        self.assertNotIn((1, 7), obs)              # motif grouping excluded

    def test_read_observed_csv_explicit_path(self):
        path = os.path.join(self.tmp, 'mech.csv')
        _mech_df({(1, 1): (0.4, True)}, include_motif=False).to_csv(path, index=False)
        obs = INF._read_observed_csv(path)
        self.assertIn((1, 1), obs)
        self.assertTrue(obs[(1, 1)]['fdr_significant'])
        self.assertEqual(INF._read_observed_csv(os.path.join(self.tmp, 'missing.csv')), {})

    def test_triangulate_v2_none_without_csv(self):
        res = _influence_result(_GNN_CONVERGE)
        self.assertIsNone(INF.triangulate_v2(res, tempfile.mkdtemp()))


class TestTriangulationMetric(unittest.TestCase):
    """The §1A success arithmetic: Spearman ρ + bootstrap CI + FDR-restricted sign agreement."""

    def test_spearman_value_is_real_rank_correlation(self):
        # Swap two GNN ranks vs observed → ρ should be exactly 0.8 (not merely sign-based).
        gnn = {(0, 0): 0.05, (1, 1): 0.20, (2, 2): -0.15, (3, 3): 0.30}
        m = INF.triangulation_metric(_influence_result(gnn), _observed(_OBS_CONVERGE),
                                     n_boot=200, seed=42)
        self.assertEqual(m['unit'], 'cue_transition (from_stage × PURER move)')
        self.assertEqual(m['n_cells'], 4)
        self.assertAlmostEqual(m['spearman_rho'], 0.8, places=4)

    def test_bootstrap_ci_returned_and_excludes_zero(self):
        m = INF.triangulation_metric(_influence_result(_GNN_CONVERGE), _observed(_OBS_CONVERGE),
                                     n_boot=200, seed=42)
        self.assertAlmostEqual(m['spearman_rho'], 1.0, places=4)
        self.assertIsNotNone(m['ci_lo'])
        self.assertIsNotNone(m['ci_hi'])
        self.assertLessEqual(m['ci_lo'], m['ci_hi'])
        self.assertTrue(m['ci_excludes_zero'])
        self.assertEqual(m['n_participants'], 5)

    def test_sign_agreement_restricted_to_fdr_significant(self):
        # 3 FDR-significant cells, GNN sign disagrees on exactly one → 2/3 ≈ 0.667.
        obs = {(0, 0): (0.10, False), (1, 1): (0.40, True),
               (2, 2): (-0.20, True), (3, 3): (0.25, True)}
        gnn = {(0, 0): 0.05, (1, 1): 0.30, (2, 2): 0.02, (3, 3): 0.20}   # (2,2) flips sign
        m = INF.triangulation_metric(_influence_result(gnn), _observed(obs),
                                     n_boot=200, seed=42)
        self.assertEqual(m['n_fdr_significant'], 3)
        self.assertAlmostEqual(m['sign_agreement'], 2.0 / 3.0, places=4)
        self.assertGreater(m['spearman_rho'], 0)      # ρ stays positive (ranks unchanged)
        self.assertTrue(m['ci_excludes_zero'])
        self.assertFalse(m['converges'])              # fails ONLY the sign gate

    def test_converges_true_when_all_three_conditions_met(self):
        m = INF.triangulation_metric(_influence_result(_GNN_CONVERGE), _observed(_OBS_CONVERGE),
                                     n_boot=200, seed=42)
        self.assertTrue(m['spearman_rho'] > 0)
        self.assertTrue(m['ci_excludes_zero'])
        self.assertGreaterEqual(m['sign_agreement'], 0.70)
        self.assertTrue(m['converges'])

    def test_converges_false_when_rho_negative(self):
        gnn = {k: -v for k, v in _GNN_CONVERGE.items()}     # anti-correlated → ρ = -1
        obs = {(0, 0): (0.10, True), (1, 1): (0.40, True),
               (2, 2): (-0.20, True), (3, 3): (0.25, True)}
        m = INF.triangulation_metric(_influence_result(gnn), _observed(obs),
                                     n_boot=200, seed=42)
        self.assertLess(m['spearman_rho'], 0)
        self.assertFalse(m['converges'])              # fails the ρ>0 gate

    def test_converges_false_when_no_fdr_significant_cells(self):
        obs = {k: (d, False) for k, (d, f) in _OBS_CONVERGE.items()}   # nothing FDR-significant
        m = INF.triangulation_metric(_influence_result(_GNN_CONVERGE), _observed(obs),
                                     n_boot=200, seed=42)
        self.assertAlmostEqual(m['spearman_rho'], 1.0, places=4)
        self.assertTrue(m['ci_excludes_zero'])
        self.assertEqual(m['n_fdr_significant'], 0)
        self.assertIsNone(m['sign_agreement'])
        self.assertFalse(m['converges'])              # cannot claim convergence w/o FDR cells

    def test_per_cell_detail_shape(self):
        m = INF.triangulation_metric(_influence_result(_GNN_CONVERGE), _observed(_OBS_CONVERGE),
                                     n_boot=100, seed=42)
        self.assertEqual(len(m['per_cell']), 4)
        for r in m['per_cell']:
            self.assertEqual(set(r) >= {'from_stage', 'move', 'move_name', 'observed_delta',
                                        'counterfactual_influence', 'fdr_significant',
                                        'sign_match'}, True)

    def test_too_few_common_cells_degrades(self):
        m = INF.triangulation_metric(_influence_result({(0, 0): 0.1}),
                                     _observed({(0, 0): (0.2, True)}), n_boot=50)
        self.assertEqual(m['n_cells'], 1)
        self.assertIsNone(m['spearman_rho'])
        self.assertFalse(m['converges'])


class TestRunCounterfactualExperiment(unittest.TestCase):
    """Un-gated experiment entry point — gate κ echoed as trust context, swap logic reused."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.csv = os.path.join(self.tmp, 'mech.csv')
        _mech_df(_OBS_CONVERGE, include_motif=False).to_csv(self.csv, index=False)
        self.cfg = GnnLayerConfig(influence_bootstrap_n=200, seed=42)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_runs_ungated_and_triangulates_against_csv(self):
        infl = _influence_result(_GNN_CONVERGE)
        with mock.patch.object(INF, 'counterfactual_influence', return_value=infl) as cf:
            res = INF.run_counterfactual_experiment(
                model='M', graph='G', df_all='D', config=self.cfg,
                gate_kappa=0.21, mechanism_csv=self.csv)
        cf.assert_called_once()                        # reuses the existing swap logic
        self.assertIs(res['influence'], infl)
        self.assertEqual(res['gate_kappa'], 0.21)      # echoed trust context, unmodified
        self.assertIsNotNone(res['triangulation'])
        self.assertTrue(res['triangulation']['converges'])

    def test_no_mechanism_csv_means_no_triangulation(self):
        infl = _influence_result(_GNN_CONVERGE)
        with mock.patch.object(INF, 'counterfactual_influence', return_value=infl):
            res = INF.run_counterfactual_experiment(None, None, None, self.cfg, gate_kappa=0.5)
        self.assertIsNone(res['triangulation'])
        self.assertEqual(res['gate_kappa'], 0.5)

    def test_status_passthrough_when_no_centroids(self):
        with mock.patch.object(INF, 'counterfactual_influence',
                               return_value={'status': 'skipped: no centroids'}):
            res = INF.run_counterfactual_experiment(None, None, None, self.cfg,
                                                    gate_kappa=0.3, mechanism_csv=self.csv)
        self.assertEqual(res['status'], 'skipped: no centroids')
        self.assertIsNone(res['triangulation'])
        self.assertEqual(res['gate_kappa'], 0.3)


class TestTriangulateBackwardCompat(unittest.TestCase):
    """Legacy triangulate() keeps its keys and additively carries cue_transition."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        mdir = _paths.mechanism_dir(self.tmp)
        os.makedirs(mdir, exist_ok=True)
        _mech_df(_OBS_CONVERGE).to_csv(
            os.path.join(mdir, 'mechanism_delta_progression.csv'), index=False)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_legacy_keys_preserved_plus_cue_transition(self):
        per_move = [{'move': m, 'mean_influence': v}
                    for (s, m), v in _GNN_CONVERGE.items()]
        res = {**_influence_result(_GNN_CONVERGE), 'per_move': per_move}
        tri_out = INF.triangulate(res, self.tmp)
        self.assertIsNotNone(tri_out)
        for k in ('n_moves', 'spearman', 'sign_agreement', 'per_move'):
            self.assertIn(k, tri_out)                  # legacy contract intact
        self.assertIn('cue_transition', tri_out)       # additive §1A metric
        self.assertTrue(tri_out['cue_transition']['converges'])


if __name__ == '__main__':
    unittest.main()
