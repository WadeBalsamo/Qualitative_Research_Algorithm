"""
tests/unit/test_gnn_influence.py
--------------------------------
Unit tests for Track B3/B4/B5 — model-counterfactual influence + triangulation.

Hermetic: builds a tiny graph with random embeddings, trains a few epochs, and exercises
purer_centroids, counterfactual_influence (per-move + per-stage tables, cluster-bootstrap
CIs, block cap), triangulate against a hand-written mechanism CSV, the report/CSV writers,
the subgroup sidecar, and the OFF-by-default flags. No model weights download.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from tests.testhelpers import synthetic_df
from gnn_layer.config import GnnLayerConfig
from gnn_layer import influence as INF
from process import output_paths as _paths


def _seg_emb(df, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    return {sid: rng.standard_normal(dim).astype('float32') for sid in df['segment_id']}


def _trained(n_sessions=4, seed=1):
    from gnn_layer import graph_builder as gb, train as tr
    from gnn_layer.soft_labels import build_soft_targets
    cfg = GnnLayerConfig(hidden_dim=16, n_layers=2, knn_k=3, epochs=6, validation_folds=2,
                         cache_embeddings=False, seed=seed, influence_bootstrap_n=64,
                         objectives=['soft_vaamr', 'progression', 'purer'])
    df = synthetic_df(n_sessions=n_sessions)
    g = gb.build_graph(df, _seg_emb(df, seed=seed + 1), cfg)
    tgts = tr.assemble_targets(g, build_soft_targets(df, 'weak'), cfg, df_all=df)
    model, _ = tr.train_model(g, tgts, cfg, n_vce=0)
    return model, g, df, cfg


class TestCentroids(unittest.TestCase):

    def test_centroids_and_null(self):
        _, g, df, _ = _trained()
        cents, null, counts = INF.purer_centroids(g, df)
        self.assertTrue(cents)                       # some moves have therapist support
        self.assertEqual(null.shape[0], g.x.shape[1])
        for m, vec in cents.items():
            self.assertEqual(vec.shape[0], g.x.shape[1])
            self.assertGreaterEqual(counts.get(m, 0), 1)


class TestCounterfactual(unittest.TestCase):

    def test_influence_tables(self):
        model, g, df, cfg = _trained()
        res = INF.counterfactual_influence(model, g, df, cfg)
        self.assertNotIn('status', res)
        self.assertGreater(res['n_blocks'], 0)
        self.assertTrue(res['per_move'])
        for r in res['per_move']:
            self.assertIn('mean_influence', r)
            self.assertIn('move_name', r)
            self.assertGreaterEqual(r['n_blocks'], 1)

    def test_block_cap_logged(self):
        model, g, df, cfg = _trained()
        cfg.counterfactual_max_blocks = 1
        res = INF.counterfactual_influence(model, g, df, cfg)
        self.assertGreaterEqual(res['n_capped'], 0)
        # n_blocks counts distinct (from,to) pairs actually scored — at most the cap.
        self.assertLessEqual(res['n_blocks'], 1)

    def test_no_centroids_skips(self):
        model, g, df, cfg = _trained()
        df2 = df.copy()
        df2['purer_primary'] = np.nan      # remove all PURER labels → no centroids
        res = INF.counterfactual_influence(model, g, df2, cfg)
        self.assertIn('status', res)


class TestTriangulation(unittest.TestCase):

    def _write_mechanism_csv(self, out):
        mdir = _paths.mechanism_dir(out)
        os.makedirs(mdir, exist_ok=True)
        rows = [
            {'grouping': 'purer', 'from_stage': 1, 'behavior': 'Utilization(1)', 'mean_delta_prog': 0.4},
            {'grouping': 'purer', 'from_stage': 2, 'behavior': 'Reframing(2)', 'mean_delta_prog': -0.2},
            {'grouping': 'purer', 'from_stage': 0, 'behavior': 'Phenomenology(0)', 'mean_delta_prog': 0.1},
            {'grouping': 'motif', 'from_stage': 1, 'behavior': 3, 'mean_delta_prog': 0.9},
        ]
        pd.DataFrame(rows).to_csv(os.path.join(mdir, 'mechanism_delta_progression.csv'), index=False)

    def test_observed_parse(self):
        out = tempfile.mkdtemp()
        self._write_mechanism_csv(out)
        obs = INF._observed_per_move(out)
        self.assertIn(1, obs)
        self.assertIn(2, obs)
        self.assertNotIn(3, obs)              # motif rows excluded
        self.assertAlmostEqual(obs[1], 0.4, places=5)

    def test_triangulate_structure(self):
        out = tempfile.mkdtemp()
        self._write_mechanism_csv(out)
        model, g, df, cfg = _trained()
        res = INF.counterfactual_influence(model, g, df, cfg)
        tri = INF.triangulate(res, out)
        self.assertIsNotNone(tri)
        self.assertIn('spearman', tri)
        self.assertIn('sign_agreement', tri)
        self.assertTrue(all('converges' in r for r in tri['per_move']))

    def test_triangulate_none_without_csv(self):
        model, g, df, cfg = _trained()
        res = INF.counterfactual_influence(model, g, df, cfg)
        self.assertIsNone(INF.triangulate(res, tempfile.mkdtemp()))


class TestWritersAndSubgroup(unittest.TestCase):

    def test_report_and_csv(self):
        out = tempfile.mkdtemp()
        model, g, df, cfg = _trained()
        res = INF.counterfactual_influence(model, g, df, cfg)
        csv_path = INF.write_influence_csv(res, out)
        self.assertTrue(os.path.isfile(csv_path))
        rep = INF.write_influence_report(res, None, out)
        self.assertTrue(os.path.isfile(rep))
        with open(rep) as f:
            txt = f.read()
        self.assertIn('MODEL-COUNTERFACTUAL INFLUENCE', txt)
        self.assertIn('NOT CAUSATION', txt)

    def test_subgroup_sidecar(self):
        out = tempfile.mkdtemp()
        model, g, df, cfg = _trained(n_sessions=6)
        res = INF.counterfactual_influence(model, g, df, cfg)
        # synthetic_df has no session_number column → subgroup returns None gracefully.
        self.assertIsNone(INF.subgroup_influence(res, df, out))


class TestDefaults(unittest.TestCase):

    def test_off_by_default(self):
        c = GnnLayerConfig()
        self.assertFalse(c.counterfactual)
        self.assertFalse(c.counterfactual_subgroups)
        self.assertIsNone(c.counterfactual_max_blocks)


if __name__ == '__main__':
    unittest.main()
