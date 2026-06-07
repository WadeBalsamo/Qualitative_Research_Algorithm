"""
tests/unit/test_gnn_scale_sim.py
--------------------------------
Unit tests for Track A5 — scale-mode simulation gate.

Holds whole sessions out, attaches them inductively via attach_new_segments, and
compares held-out κ to the in-sample CV κ. Covers (hermetic): the simulation runs and
returns both κ values + a gap + a risk flag; the <2-session and missing-session-column
guards; the report is written; the risk flag respects the configured max gap.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from tests.testhelpers import synthetic_df
from gnn_layer.config import GnnLayerConfig
from gnn_layer.classifier import validation as VAL


def _seg_emb(df, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    return {sid: rng.standard_normal(dim).astype('float32') for sid in df['segment_id']}


def _cfg(**kw):
    return GnnLayerConfig(hidden_dim=16, n_layers=2, knn_k=3, epochs=6, validation_folds=2,
                          cache_embeddings=False, seed=1, scale_sim_holdout_sessions=1,
                          objectives=['soft_vaamr', 'progression', 'purer'], **kw)


class TestScaleModeSimulation(unittest.TestCase):

    def test_runs_and_returns_both_kappas(self):
        df = synthetic_df(n_sessions=5)
        res = VAL.scale_mode_simulation(df, _seg_emb(df), _cfg())
        self.assertIn('kappa_cv_insample', res)
        self.assertIn('kappa_inductive_holdout', res)
        self.assertIn('gap', res)
        self.assertIn('domain_shift_risk', res)
        self.assertGreater(res['n_attached_scored'], 0)

    def test_single_session_skips(self):
        df = synthetic_df(n_sessions=1)
        res = VAL.scale_mode_simulation(df, _seg_emb(df), _cfg())
        self.assertIn('skipped', res.get('status', ''))

    def test_missing_session_column_skips(self):
        df = synthetic_df(n_sessions=3).drop(columns=['session_id'])
        res = VAL.scale_mode_simulation(df, _seg_emb(df), _cfg())
        self.assertIn('skipped', res.get('status', ''))

    def test_report_written(self):
        df = synthetic_df(n_sessions=4)
        res = VAL.scale_mode_simulation(df, _seg_emb(df), _cfg())
        out = tempfile.mkdtemp()
        path = VAL.write_scale_sim_report(res, out)
        self.assertTrue(os.path.isfile(path))
        with open(path) as f:
            self.assertIn('SCALE-MODE SIMULATION GATE', f.read())

    def test_risk_flag_respects_max_gap(self):
        # A huge max_gap can never be exceeded → never flags risk.
        df = synthetic_df(n_sessions=4)
        res = VAL.scale_mode_simulation(df, _seg_emb(df), _cfg(scale_sim_max_gap=5.0))
        self.assertFalse(res['domain_shift_risk'])

    def test_off_by_default(self):
        self.assertFalse(GnnLayerConfig().run_scale_sim)


if __name__ == '__main__':
    unittest.main()
