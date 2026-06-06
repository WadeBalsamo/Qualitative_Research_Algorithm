"""
tests/unit/test_gnn_precipitates.py
-----------------------------------
Unit tests for Track A1 — typed therapist->participant ``precipitates`` edges and
the learnable per-edge-type gate in SAGEConv.

Covers (all hermetic — random embeddings, no model download):
  * build_graph: family OFF (default) leaves the graph unchanged + edge_type_ids
    aligned; family ON adds precipitates edges + records the vocab.
  * edge_type_ids stay aligned through save/load and inductive attach.
  * build_model: learnable gate present only when precipitates_edges=True; the OFF
    path is parameter-identical to the original fixed-weight model.
  * SAGEConv forward accepts (and at neutral init is unchanged by) edge_type_ids.
  * ablation.precipitates_contribution: returns the expected fields and stays
    'inconclusive' (recommend OFF) without a decisive human subset.
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


def _seg_emb(df, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    return {sid: rng.standard_normal(dim).astype('float32') for sid in df['segment_id']}


class TestPrecipitatesGraph(unittest.TestCase):

    def _build(self, precipitates):
        from gnn_layer.graph_builder import build_graph
        df = synthetic_df(n_sessions=3)
        cfg = GnnLayerConfig(knn_k=3, cache_embeddings=False, precipitates_edges=precipitates)
        return build_graph(df, _seg_emb(df), cfg)

    def test_off_by_default_no_precipitates(self):
        cfg = GnnLayerConfig()
        self.assertFalse(cfg.precipitates_edges)
        g = self._build(precipitates=False)
        self.assertEqual(g.meta.get('n_precipitates', 0), 0)
        self.assertNotIn('precipitates', g.meta['edge_types'])

    def test_on_adds_precipitates_edges(self):
        g = self._build(precipitates=True)
        self.assertGreater(g.meta['n_precipitates'], 0)
        self.assertIn('precipitates', g.meta['edge_types'])

    def test_precipitates_increases_edge_count(self):
        g_off = self._build(precipitates=False)
        g_on = self._build(precipitates=True)
        self.assertGreater(g_on.edge_index.size(1), g_off.edge_index.size(1))

    def test_edge_type_ids_aligned_with_edges(self):
        for prec in (False, True):
            g = self._build(precipitates=prec)
            self.assertIsNotNone(g.edge_type_ids)
            self.assertEqual(g.edge_type_ids.numel(), g.edge_index.size(1))

    def test_edge_type_vocab_recorded(self):
        from gnn_layer.graph_builder import EDGE_TYPE_VOCAB
        g = self._build(precipitates=True)
        self.assertEqual(tuple(g.meta['edge_type_vocab']), EDGE_TYPE_VOCAB)

    def test_precipitates_connect_therapist_to_participant(self):
        # Every precipitates edge's two endpoints must be one therapist + one participant.
        g = self._build(precipitates=True)
        from gnn_layer.graph_builder import EDGE_TYPE_TO_ID
        pid = EDGE_TYPE_TO_ID['precipitates']
        et = g.edge_type_ids.tolist()
        ei = g.edge_index.t().tolist()
        nt = g.node_types
        seen = 0
        for k, (a, b) in enumerate(ei):
            if et[k] == pid:
                seen += 1
                types = {nt[a], nt[b]}
                self.assertEqual(types, {'therapist_segment', 'participant_segment'})
        self.assertGreater(seen, 0)


class TestPrecipitatesSaveLoadAttach(unittest.TestCase):

    def _build(self):
        from gnn_layer.graph_builder import build_graph
        df = synthetic_df(n_sessions=3)
        cfg = GnnLayerConfig(knn_k=3, cache_embeddings=False, precipitates_edges=True)
        return build_graph(df, _seg_emb(df), cfg), cfg

    def test_save_load_roundtrip_preserves_edge_type_ids(self):
        from gnn_layer.graph_builder import save_graph, load_graph
        g, _ = self._build()
        d = tempfile.mkdtemp()
        save_graph(g, d)
        g2 = load_graph(d)
        self.assertIsNotNone(g2.edge_type_ids)
        self.assertEqual(g2.edge_type_ids.numel(), g.edge_type_ids.numel())
        self.assertTrue(bool((g2.edge_type_ids == g.edge_type_ids).all()))

    def test_attach_keeps_edge_type_ids_aligned(self):
        from gnn_layer.graph_builder import attach_new_segments
        g, cfg = self._build()
        rng = np.random.default_rng(9)
        new = {'cNEW_0': rng.standard_normal(16).astype('float32')}
        g3 = attach_new_segments(g, new, cfg, node_type_of={'cNEW_0': 'participant_segment'})
        self.assertEqual(g3.edge_type_ids.numel(), g3.edge_index.size(1))


class TestLearnableEdgeTypeGate(unittest.TestCase):

    def _graph(self, precipitates):
        from gnn_layer.graph_builder import build_graph
        df = synthetic_df(n_sessions=2)
        cfg = GnnLayerConfig(knn_k=3, hidden_dim=16, n_layers=2,
                             cache_embeddings=False, precipitates_edges=precipitates)
        return build_graph(df, _seg_emb(df), cfg), cfg

    def test_gate_absent_when_precipitates_off(self):
        from gnn_layer.model import build_model
        g, cfg = self._graph(precipitates=False)
        model = build_model(g, cfg)
        for conv in model.convs:
            self.assertEqual(conv.n_edge_types, 0)
            self.assertIsNone(conv.edge_type_gate)

    def test_gate_present_when_precipitates_on(self):
        from gnn_layer.model import build_model
        from gnn_layer.graph_builder import EDGE_TYPE_VOCAB
        g, cfg = self._graph(precipitates=True)
        model = build_model(g, cfg)
        for conv in model.convs:
            self.assertEqual(conv.n_edge_types, len(EDGE_TYPE_VOCAB))
            self.assertIsNotNone(conv.edge_type_gate)

    def test_neutral_gate_leaves_aggregation_unchanged_at_init(self):
        # At init softplus(gate)==1, so passing edge_type_ids must NOT change the forward
        # output vs not passing it (backward-compatible default).
        import torch
        from gnn_layer.model import build_model
        g, cfg = self._graph(precipitates=True)
        model = build_model(g, cfg)
        model.eval()
        with torch.no_grad():
            out_no_ids = model(g.x, g.edge_index, g.edge_weight, None)
            out_ids = model(g.x, g.edge_index, g.edge_weight, g.edge_type_ids)
        self.assertTrue(torch.allclose(out_no_ids['emb'], out_ids['emb'], atol=1e-5))

    def test_off_model_param_count_unchanged(self):
        # The OFF path must add zero parameters vs a precipitates-unaware model.
        from gnn_layer.model import build_model
        g_off, cfg_off = self._graph(precipitates=False)
        model_off = build_model(g_off, cfg_off)
        g_on, cfg_on = self._graph(precipitates=True)
        model_on = build_model(g_on, cfg_on)
        n_off = sum(p.numel() for p in model_off.parameters())
        n_on = sum(p.numel() for p in model_on.parameters())
        # ON adds exactly the per-edge-type gate params (n_edge_types per conv layer).
        from gnn_layer.graph_builder import EDGE_TYPE_VOCAB
        self.assertEqual(n_on - n_off, len(EDGE_TYPE_VOCAB) * len(model_on.convs))


class TestPrecipitatesAblation(unittest.TestCase):

    def test_contribution_inconclusive_without_human_subset(self):
        from gnn_layer import ablation as ABL
        df = synthetic_df(n_sessions=4)
        cfg = GnnLayerConfig(hidden_dim=16, n_layers=2, knn_k=3, epochs=6,
                             validation_folds=2, cache_embeddings=False, seed=1,
                             objectives=['soft_vaamr', 'progression', 'purer'])
        res = ABL.precipitates_contribution(df, _seg_emb(df, seed=3), cfg)
        self.assertIn('verdict', res)
        self.assertGreater(res['n_precipitates_edges'], 0)
        # synthetic_df has no in_human_coded_subset rows → cannot confirm validity → OFF
        self.assertEqual(res['verdict'], 'inconclusive')
        self.assertFalse(res['recommend_precipitates'])

    def test_contribution_report_written(self):
        from gnn_layer import ablation as ABL
        df = synthetic_df(n_sessions=4)
        cfg = GnnLayerConfig(hidden_dim=16, n_layers=2, knn_k=3, epochs=4,
                             validation_folds=2, cache_embeddings=False, seed=1,
                             objectives=['soft_vaamr', 'progression', 'purer'])
        res = ABL.precipitates_contribution(df, _seg_emb(df, seed=3), cfg)
        out = tempfile.mkdtemp()
        path = ABL.write_precipitates_contribution_report(res, out)
        self.assertTrue(os.path.isfile(path))
        with open(path) as f:
            self.assertIn('PRECIPITATES-EDGE CONTRIBUTION', f.read())


if __name__ == '__main__':
    unittest.main()
