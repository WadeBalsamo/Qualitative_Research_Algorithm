"""
tests/unit/test_gnn_propagation.py
----------------------------------
Unit tests for Track A4 — semi-supervised label propagation (measured).

Covers (hermetic): neighbour_weighted_mean matches a hand graph; propagate row-normalizes
and equals the input when alpha=0; propagation_contribution returns a verdict + Δκ and
writes a report; the OFF default is respected.
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
from gnn_layer import propagation as PROP


def _seg_emb(df, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    return {sid: rng.standard_normal(dim).astype('float32') for sid in df['segment_id']}


class TestPropagateMechanics(unittest.TestCase):

    def test_neighbour_weighted_mean_simple(self):
        # nodes 0,1,2; edge 0->2 (w1), 1->2 (w3). node 2 should get weighted mean.
        F = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        ei = np.array([[0, 1], [2, 2]])
        ew = np.array([1.0, 3.0])
        out = PROP.neighbour_weighted_mean(F, ei, ew, 3)
        # node2 = (1*[1,0] + 3*[0,1]) / 4 = [0.25, 0.75]
        self.assertTrue(np.allclose(out[2], [0.25, 0.75]))
        self.assertTrue(np.allclose(out[0], [0, 0]))  # no incoming edges

    def test_propagate_alpha_zero_is_identity(self):
        df = synthetic_df(n_sessions=2)
        from gnn_layer import graph_builder as gb
        g = gb.build_graph(df, _seg_emb(df), GnnLayerConfig(knn_k=3, cache_embeddings=False))
        rng = np.random.default_rng(3)
        P = rng.random((g.x.shape[0], 5)); P /= P.sum(1, keepdims=True)
        out = PROP.propagate(P, g, alpha=0.0, n_iter=10)
        self.assertTrue(np.allclose(out, P, atol=1e-6))

    def test_propagate_rows_sum_to_one(self):
        df = synthetic_df(n_sessions=2)
        from gnn_layer import graph_builder as gb
        g = gb.build_graph(df, _seg_emb(df), GnnLayerConfig(knn_k=3, cache_embeddings=False))
        rng = np.random.default_rng(4)
        P = rng.random((g.x.shape[0], 5)); P /= P.sum(1, keepdims=True)
        out = PROP.propagate(P, g, alpha=0.6, n_iter=15)
        self.assertTrue(np.allclose(out.sum(axis=1), 1.0, atol=1e-6))


class TestPropagationContribution(unittest.TestCase):

    def _setup(self):
        from gnn_layer import graph_builder as gb, train as tr
        from gnn_layer.soft_labels import build_soft_targets
        cfg = GnnLayerConfig(hidden_dim=16, n_layers=2, knn_k=3, epochs=6, validation_folds=2,
                             cache_embeddings=False, seed=1, propagation_alpha=0.5,
                             propagation_iters=10,
                             objectives=['soft_vaamr', 'progression', 'purer'])
        df = synthetic_df(n_sessions=4)
        g = gb.build_graph(df, _seg_emb(df, seed=2), cfg)
        tgts = tr.assemble_targets(g, build_soft_targets(df, 'weak'), cfg, df_all=df)
        return g, tgts, cfg, df

    def test_contribution_returns_verdict_and_deltas(self):
        g, tgts, cfg, df = self._setup()
        res = PROP.propagation_contribution(g, tgts, cfg, df, n_vce=0)
        self.assertIn('verdict', res)
        self.assertIn(res['verdict'],
                      ('propagation_helps', 'propagation_harms', 'propagation_neutral',
                       'inconclusive'))
        self.assertGreater(res['n_heldout'], 0)
        self.assertIn('delta_kappa', res)

    def test_report_written(self):
        g, tgts, cfg, df = self._setup()
        res = PROP.propagation_contribution(g, tgts, cfg, df, n_vce=0)
        out = tempfile.mkdtemp()
        path = PROP.write_propagation_report(res, out)
        self.assertTrue(os.path.isfile(path))
        with open(path) as f:
            self.assertIn('LABEL-PROPAGATION CONTRIBUTION', f.read())

    def test_off_by_default(self):
        self.assertFalse(GnnLayerConfig().label_propagation)


if __name__ == '__main__':
    unittest.main()
