"""
tests/unit/test_gnn_model.py
------------------------------
Unit tests for gnn_layer/model.py.

Covers: build_model head construction, forward output shapes for different
objectives subsets (emb dim, soft_vaamr=5, progression=1, purer=5),
compute_losses returns expected keys and masked heads contribute 0 / are
skipped, link_prediction_loss, supervised_contrastive loss.
Uses a tiny graph from graph_builder + synthetic_df(n_sessions=1).
"""

import os
import sys
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.testhelpers import synthetic_df, embedding_patch, make_master_df
from gnn_layer.config import GnnLayerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(n_sessions=1, dim=16, knn_k=2, seed=0):
    from gnn_layer import graph_builder
    df = synthetic_df(n_sessions=n_sessions)
    cfg = GnnLayerConfig(knn_k=knn_k, cache_embeddings=False)
    rng = np.random.default_rng(seed)
    seg_emb = {sid: rng.standard_normal(dim).astype('float32') for sid in df['segment_id']}
    return graph_builder.build_graph(df, seg_emb, cfg), df


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

class TestBuildModel(unittest.TestCase):

    def setUp(self):
        self.g, self.df = _make_graph()
        from gnn_layer.model import build_model
        self._build = build_model

    def test_soft_vaamr_head_present(self):
        cfg = GnnLayerConfig(objectives=['soft_vaamr'], knn_k=2, cache_embeddings=False, hidden_dim=8)
        model = self._build(self.g, cfg)
        self.assertIn('soft_vaamr', model.heads)

    def test_progression_head_present(self):
        cfg = GnnLayerConfig(objectives=['progression'], knn_k=2, cache_embeddings=False, hidden_dim=8)
        model = self._build(self.g, cfg)
        self.assertIn('progression', model.heads)

    def test_purer_head_present(self):
        cfg = GnnLayerConfig(objectives=['purer'], knn_k=2, cache_embeddings=False, hidden_dim=8)
        model = self._build(self.g, cfg)
        self.assertIn('purer', model.heads)

    def test_no_purer_head_without_objective(self):
        cfg = GnnLayerConfig(objectives=['soft_vaamr', 'progression'],
                             knn_k=2, cache_embeddings=False, hidden_dim=8)
        model = self._build(self.g, cfg)
        self.assertNotIn('purer', model.heads)

    def test_empty_objectives_still_builds_soft_vaamr_head(self):
        """Empty objectives must still produce at least one head (soft_vaamr fallback)."""
        cfg = GnnLayerConfig(objectives=[], knn_k=2, cache_embeddings=False, hidden_dim=8)
        model = self._build(self.g, cfg)
        self.assertIn('soft_vaamr', model.heads)

    def test_vce_head_absent_without_n_vce(self):
        cfg = GnnLayerConfig(objectives=['vce_multilabel'], knn_k=2,
                             cache_embeddings=False, hidden_dim=8)
        model = self._build(self.g, cfg, n_vce=0)
        self.assertNotIn('vce', model.heads)

    def test_vce_head_present_with_n_vce(self):
        cfg = GnnLayerConfig(objectives=['vce_multilabel'], knn_k=2,
                             cache_embeddings=False, hidden_dim=8)
        model = self._build(self.g, cfg, n_vce=10)
        self.assertIn('vce', model.heads)

    def test_model_is_callable(self):
        cfg = GnnLayerConfig(objectives=['soft_vaamr'], knn_k=2, cache_embeddings=False, hidden_dim=8)
        model = self._build(self.g, cfg)
        out = model(self.g.x, self.g.edge_index, self.g.edge_weight)
        self.assertIsInstance(out, dict)


# ---------------------------------------------------------------------------
# Forward pass — output shapes
# ---------------------------------------------------------------------------

class TestForwardShapes(unittest.TestCase):

    def setUp(self):
        self.g, self.df = _make_graph()
        from gnn_layer.model import build_model
        self._build = build_model

    def _forward(self, objectives, hidden_dim=8, n_vce=0):
        cfg = GnnLayerConfig(objectives=objectives, knn_k=2,
                             cache_embeddings=False, hidden_dim=hidden_dim, n_layers=2)
        model = self._build(self.g, cfg, n_vce=n_vce)
        model.eval()
        return model(self.g.x, self.g.edge_index, self.g.edge_weight), cfg

    def test_emb_shape_matches_hidden_dim(self):
        out, cfg = self._forward(['soft_vaamr'])
        N = self.g.x.shape[0]
        self.assertEqual(out['emb'].shape, (N, 8))

    def test_emb_shape_with_larger_hidden_dim(self):
        out, _ = self._forward(['soft_vaamr'], hidden_dim=32)
        self.assertEqual(out['emb'].shape[1], 32)

    def test_soft_vaamr_shape(self):
        out, _ = self._forward(['soft_vaamr'])
        N = self.g.x.shape[0]
        self.assertEqual(out['soft_vaamr'].shape, (N, 5))

    def test_progression_shape(self):
        out, _ = self._forward(['progression'])
        N = self.g.x.shape[0]
        self.assertEqual(out['progression'].shape, (N, 1))

    def test_purer_shape(self):
        out, _ = self._forward(['purer'])
        N = self.g.x.shape[0]
        self.assertEqual(out['purer'].shape, (N, 5))

    def test_all_objectives_all_keys_present(self):
        out, _ = self._forward(['soft_vaamr', 'progression', 'purer'])
        for k in ('emb', 'soft_vaamr', 'progression', 'purer'):
            self.assertIn(k, out, f"Key '{k}' missing from forward output")

    def test_vce_shape_with_n_vce(self):
        cfg = GnnLayerConfig(objectives=['vce_multilabel'], knn_k=2,
                             cache_embeddings=False, hidden_dim=8)
        from gnn_layer.model import build_model
        model = build_model(self.g, cfg, n_vce=7)
        model.eval()
        out = model(self.g.x, self.g.edge_index, self.g.edge_weight)
        N = self.g.x.shape[0]
        self.assertEqual(out['vce'].shape, (N, 7))

    def test_forward_without_edge_weight_is_valid(self):
        cfg = GnnLayerConfig(objectives=['soft_vaamr'], knn_k=2,
                             cache_embeddings=False, hidden_dim=8)
        from gnn_layer.model import build_model
        model = build_model(self.g, cfg)
        out = model(self.g.x, self.g.edge_index, edge_weight=None)
        self.assertIn('soft_vaamr', out)


# ---------------------------------------------------------------------------
# compute_losses
# ---------------------------------------------------------------------------

class TestComputeLosses(unittest.TestCase):

    def setUp(self):
        import torch
        self.g, self.df = _make_graph()
        from gnn_layer.model import build_model, compute_losses
        from gnn_layer.soft_labels import build_soft_targets
        from gnn_layer.train import assemble_targets

        self.cfg = GnnLayerConfig(
            objectives=['soft_vaamr', 'progression', 'purer'],
            knn_k=2, cache_embeddings=False, hidden_dim=8, n_layers=1,
        )
        self.model = build_model(self.g, self.cfg)
        self.model.eval()
        self.out = self.model(self.g.x, self.g.edge_index, self.g.edge_weight)
        soft = build_soft_targets(self.df)
        self.targets = assemble_targets(self.g, soft, self.cfg, df_all=self.df)
        self._compute = compute_losses

    def test_total_key_present(self):
        losses = self._compute(self.out, self.targets, self.cfg)
        self.assertIn('total', losses)

    def test_total_is_scalar_tensor(self):
        import torch
        losses = self._compute(self.out, self.targets, self.cfg)
        self.assertEqual(losses['total'].dim(), 0)

    def test_supervised_term_keys_present(self):
        losses = self._compute(self.out, self.targets, self.cfg)
        self.assertIn('soft_vaamr', losses)
        self.assertIn('progression', losses)
        self.assertIn('purer', losses)

    def test_total_at_least_as_large_as_each_term(self):
        import torch
        losses = self._compute(self.out, self.targets, self.cfg)
        for k, v in losses.items():
            if k != 'total':
                self.assertGreaterEqual(
                    float(losses['total'].item()),
                    float(v.item()) - 1e-5,
                    f"total < {k}: {losses['total'].item()} < {v.item()}"
                )

    def test_empty_supervised_targets_gives_zero_total(self):
        import torch
        empty_t = {
            'vaamr_idx': torch.zeros(0, dtype=torch.long),
            'vaamr_mix': torch.zeros((0, 5)),
            'prog_val': torch.zeros(0),
            'contrast_idx': torch.zeros(0, dtype=torch.long),
            'contrast_label': torch.zeros(0, dtype=torch.long),
            'purer_idx': torch.zeros(0, dtype=torch.long),
            'purer_label': torch.zeros(0, dtype=torch.long),
            'pos_edges': torch.zeros((2, 0), dtype=torch.long),
            'neg_edges': torch.zeros((2, 0), dtype=torch.long),
        }
        losses = self._compute(self.out, empty_t, self.cfg)
        self.assertAlmostEqual(float(losses['total'].item()), 0.0, places=6)
        # supervised term keys should not appear when targets empty
        for k in ('soft_vaamr', 'progression', 'purer'):
            self.assertNotIn(k, losses)

    def test_contrastive_term_present_when_in_objectives(self):
        from gnn_layer.model import build_model, compute_losses
        cfg_c = GnnLayerConfig(
            objectives=['soft_vaamr', 'contrastive'],
            knn_k=2, cache_embeddings=False, hidden_dim=8, contrastive_temp=0.1
        )
        model_c = build_model(self.g, cfg_c)
        model_c.eval()
        out_c = model_c(self.g.x, self.g.edge_index, self.g.edge_weight)
        losses_c = compute_losses(out_c, self.targets, cfg_c)
        # contrastive fires when contrast_idx has > 1 element
        if self.targets.get('contrast_idx') is not None and self.targets['contrast_idx'].numel() > 1:
            self.assertIn('contrastive', losses_c)

    def test_link_prediction_term_present_when_in_objectives(self):
        from gnn_layer.model import build_model, compute_losses
        import torch
        cfg_lp = GnnLayerConfig(
            objectives=['soft_vaamr', 'link_prediction'],
            knn_k=2, cache_embeddings=False, hidden_dim=8,
        )
        model_lp = build_model(self.g, cfg_lp)
        model_lp.eval()
        out_lp = model_lp(self.g.x, self.g.edge_index, self.g.edge_weight)
        # supply non-empty pos/neg edges
        targets_lp = dict(self.targets)
        targets_lp['pos_edges'] = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        targets_lp['neg_edges'] = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
        losses_lp = compute_losses(out_lp, targets_lp, cfg_lp)
        self.assertIn('link_prediction', losses_lp)


# ---------------------------------------------------------------------------
# supervised_contrastive
# ---------------------------------------------------------------------------

class TestSupervisedContrastive(unittest.TestCase):

    def setUp(self):
        from gnn_layer.model import supervised_contrastive
        self._fn = supervised_contrastive

    def test_scalar_output(self):
        import torch
        emb = torch.randn(6, 8)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        loss = self._fn(emb, labels)
        self.assertEqual(loss.dim(), 0)

    def test_single_sample_returns_zero(self):
        import torch
        emb = torch.randn(1, 8)
        labels = torch.tensor([0])
        loss = self._fn(emb, labels)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_loss_is_non_negative(self):
        import torch
        emb = torch.randn(8, 8)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = self._fn(emb, labels)
        self.assertGreaterEqual(float(loss.item()), -1e-6)


# ---------------------------------------------------------------------------
# link_prediction_loss
# ---------------------------------------------------------------------------

class TestLinkPredictionLoss(unittest.TestCase):

    def setUp(self):
        from gnn_layer.model import link_prediction_loss
        self._fn = link_prediction_loss

    def test_scalar_output(self):
        import torch
        emb = torch.randn(10, 8)
        pos = torch.tensor([[0, 1, 2], [1, 2, 3]])
        neg = torch.tensor([[0, 1, 2], [4, 5, 6]])
        loss = self._fn(emb, pos, neg)
        self.assertEqual(loss.dim(), 0)

    def test_empty_pos_returns_zero(self):
        import torch
        emb = torch.randn(10, 8)
        pos = torch.zeros((2, 0), dtype=torch.long)
        neg = torch.zeros((2, 0), dtype=torch.long)
        loss = self._fn(emb, pos, neg)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_loss_is_non_negative(self):
        import torch
        emb = torch.randn(8, 8)
        pos = torch.tensor([[0, 2], [1, 3]])
        neg = torch.tensor([[4, 6], [5, 7]])
        loss = self._fn(emb, pos, neg)
        self.assertGreaterEqual(float(loss.item()), -1e-6)


if __name__ == '__main__':
    unittest.main()
