"""
tests/unit/test_gnn_transition.py
---------------------------------
Unit tests for WS-T — the dyadic FROM→CUE→TO transition model (gnn_layer/transition.py).

Hermetic: a synthetic block set where the cue carries TO signal (move 0 pushes the next
participant high, move 1 low) while FROM is uninformative — so the earns-its-place CV must show
the cue helps and the learned counterfactual must order the moves correctly. Also exercises the
real cue-block dataset builder on make_master_df, the observed-CSV reader + triangulation, and
the report writer. torch + sklearn + numpy; no network, no qra.db.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from tests.testhelpers import make_master_df
from gnn_layer import transition as T
from gnn_layer.config import GnnLayerConfig


def _fast_cfg(**kw):
    base = dict(transition_epochs=120, transition_hidden=16, transition_cue_dim=4,
                transition_folds=3, transition_bootstrap_n=200, transition_dropout=0.2, seed=1)
    base.update(kw)
    return GnnLayerConfig(**base)


def _synthetic_ds(n_per=12, seed=0):
    """Cue carries the TO signal (move 0 → stage 3, move 1 → stage 1); FROM uninformative."""
    rng = np.random.default_rng(seed)
    S, K = 5, 4
    F, St, Y, C, parts, move, fstage = [], [], [], [], [], [], []
    for p in range(8):
        for j in range(n_per):
            m = (p * n_per + j) % 2
            c = (np.array([1.0, 0, 0, 0]) if m == 0 else np.array([-1.0, 0, 0, 0]))
            c = c + rng.standard_normal(K) * 0.1
            ystage = 3 if m == 0 else 1
            y = np.full(S, 0.05); y[ystage] = 0.8; y = y / y.sum()
            oh = np.zeros(S); oh[2] = 1.0                 # all FROM stage 2 (constant)
            F.append(np.full(S, 0.2)); St.append(oh); Y.append(y); C.append(c)
            parts.append(f'P{p}'); move.append(m); fstage.append(2)
    n = len(F)
    return {'F': np.array(F), 'St': np.array(St), 'Y': np.array(Y), 'C': np.array(C),
            'parts': parts, 'from_stage': np.array(fstage), 'move': np.array(move),
            'session': ['S'] * n, 'from_seg': [f's{i}' for i in range(n)],
            'to_seg': [f't{i}' for i in range(n)], 'n_stages': S}


def _rand_emb(df, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    return {str(r['segment_id']): rng.standard_normal(dim).astype('float32') for _, r in df.iterrows()}


class TestBlockDataset(unittest.TestCase):

    def test_builds_aligned_arrays(self):
        df = make_master_df(n_sessions=4, n_participants=8)
        ds = T.build_block_dataset(df, _rand_emb(df), n_stages=5)
        self.assertIsNotNone(ds)
        n = ds['F'].shape[0]
        self.assertEqual(ds['Y'].shape[0], n)
        self.assertEqual(ds['C'].shape[0], n)
        self.assertEqual(len(ds['parts']), n)
        self.assertEqual(ds['F'].shape[1], 5)

    def test_too_few_returns_none(self):
        df = make_master_df(n_sessions=1, n_participants=1)
        self.assertIsNone(T.build_block_dataset(df, _rand_emb(df), n_stages=5))


class TestEarnsItsPlace(unittest.TestCase):

    def test_cue_lowers_heldout_error(self):
        ds = _synthetic_ds(seed=0)
        cue_proj, _, _ = T._project_cue(ds['C'], cue_dim=4, seed=1)
        cv = T.crossval(ds, cue_proj, _fast_cfg(), seed=1)
        d = cv['delta_cue_minus_from']
        # cue carries the signal → held-out KL and E[stage] MAE drop when it is included
        self.assertIsNotNone(d.get('kl'))
        self.assertLess(d['kl'], 0.0)
        self.assertLess(d['mae'], 0.0)


class TestCounterfactual(unittest.TestCase):

    def test_move_order_recovered(self):
        ds = _synthetic_ds(seed=0)
        cue_proj, _, _ = T._project_cue(ds['C'], cue_dim=4, seed=1)
        cf = T.counterfactual_response(ds, cue_proj, _fast_cfg(), seed=1)
        self.assertEqual(cf['status'], 'ok')
        infl = {r['move']: r['mean_influence'] for r in cf['per_move']}
        # move 0 pushes TO toward stage 3, move 1 toward stage 1 → influence(0) > influence(1)
        self.assertGreater(infl[0], infl[1])
        self.assertEqual(cf['per_move'][0]['move'], 0)   # sorted desc by influence


class TestTriangulation(unittest.TestCase):

    def _write_mech_csv(self, tmp):
        d = os.path.join(tmp, '03_analysis_data', 'mechanism')
        os.makedirs(d, exist_ok=True)
        rows = [{'grouping': 'purer', 'from_stage': s, 'behavior': f'M({m})',
                 'mean_delta_prog': (0.3 if m == 0 else -0.3), 'fdr_significant': False}
                for s in (0, 1) for m in (0, 1)]
        pd.DataFrame(rows).to_csv(os.path.join(d, 'mechanism_delta_progression.csv'), index=False)

    def test_observed_reader_and_triangulate(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_mech_csv(tmp)
            obs = T.observed_per_stage_move(tmp)
            self.assertIn((0, 0), obs)
            cf = {'status': 'ok', 'per_stage_move': [
                {'from_stage': s, 'move': m, 'mean_influence': (0.1 if m == 0 else -0.1)}
                for s in (0, 1) for m in (0, 1)]}
            tri = T.triangulate(cf, tmp)
            self.assertEqual(tri['n_cells'], 4)
            self.assertIsNotNone(tri['sign_agreement'])
            self.assertGreaterEqual(tri['sign_agreement'], 0.99)   # signs all match


class TestReportWriter(unittest.TestCase):

    def test_report_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            cv = {'with_cue': {'kl': 0.5, 'mae': 0.4, 'acc': 0.5},
                  'from_only': {'kl': 0.8, 'mae': 0.6, 'acc': 0.3},
                  'delta_cue_minus_from': {'kl': -0.3, 'mae': -0.2, 'acc': 0.2},
                  'n_blocks': 100, 'n_participants': 18, 'n_folds': 5}
            cf = {'status': 'ok',
                  'per_move': [{'move': 1, 'move_name': 'Utilization', 'mean_influence': 0.03,
                                'ci_lo': 0.01, 'ci_hi': 0.05, 'n_blocks': 100, 'n_participants': 18,
                                'centroid_support': 40}],
                  'per_stage_move': [{'from_stage': 2, 'move': 1, 'move_name': 'Utilization',
                                      'mean_influence': 0.03, 'n_blocks': 40}]}
            tri = {'n_cells': 12, 'spearman': -0.13, 'sign_agreement': 0.4,
                   'sign_agreement_fdr': None, 'n_fdr_cells': 0}
            rep = T.write_transition_report(cv, cf, tri, tmp)
            self.assertTrue(os.path.isfile(rep))
            txt = open(rep, encoding='utf-8').read()
            self.assertIn('TRANSITION MODEL', txt)
            self.assertIn('EARN ITS PLACE', txt.upper())
            self.assertIn('COUNTERFACTUAL', txt.upper())


if __name__ == '__main__':
    unittest.main()
