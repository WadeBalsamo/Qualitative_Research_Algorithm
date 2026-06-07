"""
tests/unit/test_gnn_train.py
-----------------------------
Unit tests for gnn_layer/train.py.

Covers: _subset_targets masking, _replace_seed, set_seed determinism,
assemble_targets (shapes/dtypes/edges/optional heads), crossval_predictions
(every labeled node predicted once; determinism with fixed seed),
train_model (returns metrics dict with expected keys; early-stop path),
export_checkpoint / load_checkpoint roundtrip.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_graph(df, cfg=None, n_sessions=1, dim=16, seed=0):
    """Build a test HeteroGraph from synthetic_df without touching the embedding model."""
    from gnn_layer.classifier import graph_builder
    if cfg is None:
        cfg = GnnLayerConfig(knn_k=2, cache_embeddings=False, hidden_dim=8)
    rng = np.random.default_rng(seed)
    seg_emb = {sid: rng.standard_normal(dim).astype('float32') for sid in df['segment_id']}
    return graph_builder.build_graph(df, seg_emb, cfg), seg_emb


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------

class TestSetSeed(unittest.TestCase):

    def test_produces_identical_tensors_on_repeat(self):
        import torch
        from gnn_layer.classifier.train import set_seed
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        self.assertTrue(torch.allclose(a, b),
                        "set_seed must make two consecutive randn calls identical")

    def test_different_seeds_produce_different_tensors(self):
        import torch
        from gnn_layer.classifier.train import set_seed
        set_seed(1)
        a = torch.randn(10)
        set_seed(2)
        b = torch.randn(10)
        self.assertFalse(torch.allclose(a, b))

    def test_numpy_seeded(self):
        from gnn_layer.classifier.train import set_seed
        set_seed(99)
        a = np.random.rand(5)
        set_seed(99)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# _replace_seed
# ---------------------------------------------------------------------------

class TestReplaceSeed(unittest.TestCase):

    def test_new_seed_value(self):
        from gnn_layer.classifier.train import _replace_seed
        cfg = GnnLayerConfig(seed=42)
        cfg2 = _replace_seed(cfg, 99)
        self.assertEqual(cfg2.seed, 99)

    def test_original_unchanged(self):
        from gnn_layer.classifier.train import _replace_seed
        cfg = GnnLayerConfig(seed=42)
        _replace_seed(cfg, 99)
        self.assertEqual(cfg.seed, 42)

    def test_other_fields_preserved(self):
        from gnn_layer.classifier.train import _replace_seed
        cfg = GnnLayerConfig(seed=1, hidden_dim=64, lr=0.01)
        cfg2 = _replace_seed(cfg, 7)
        self.assertEqual(cfg2.hidden_dim, 64)
        self.assertAlmostEqual(cfg2.lr, 0.01)


# ---------------------------------------------------------------------------
# _subset_targets
# ---------------------------------------------------------------------------

class TestSubsetTargets(unittest.TestCase):

    def setUp(self):
        import torch
        from gnn_layer.classifier.train import _subset_targets
        self._subset = _subset_targets
        self.targets = {
            'vaamr_idx': torch.tensor([10, 11, 12, 13]),
            'vaamr_mix': torch.zeros((4, 5)),
            'prog_val': torch.zeros(4),
            'contrast_idx': torch.tensor([10, 11, 12, 13]),
            'contrast_label': torch.tensor([0, 1, 2, 3]),
            'purer_idx': torch.tensor([20, 21, 22]),
            'purer_label': torch.tensor([0, 1, 2]),
            'pos_edges': torch.zeros((2, 3), dtype=torch.long),
            'neg_edges': torch.zeros((2, 3), dtype=torch.long),
        }

    def test_vaamr_subset_keeps_specified_positions(self):
        sub = self._subset(self.targets, keep_v_pos=[0, 2], keep_p_pos=[0, 1, 2])
        self.assertEqual(sub['vaamr_idx'].tolist(), [10, 12])

    def test_vaamr_mix_subset_shape(self):
        sub = self._subset(self.targets, keep_v_pos=[1, 3], keep_p_pos=[])
        self.assertEqual(sub['vaamr_mix'].shape, (2, 5))

    def test_prog_val_subset_shape(self):
        sub = self._subset(self.targets, keep_v_pos=[0], keep_p_pos=[])
        self.assertEqual(sub['prog_val'].shape, (1,))

    def test_contrast_label_matches_vaamr_keep(self):
        sub = self._subset(self.targets, keep_v_pos=[0, 2], keep_p_pos=[])
        self.assertEqual(sub['contrast_label'].tolist(), [0, 2])

    def test_purer_subset(self):
        sub = self._subset(self.targets, keep_v_pos=[0], keep_p_pos=[1])
        self.assertEqual(sub['purer_idx'].tolist(), [21])
        self.assertEqual(sub['purer_label'].tolist(), [1])

    def test_auxiliary_targets_untouched(self):
        import torch
        sub = self._subset(self.targets, keep_v_pos=[0], keep_p_pos=[0])
        # link-prediction edges are self-supervised, never masked
        self.assertEqual(sub['pos_edges'].shape, (2, 3))
        self.assertEqual(sub['neg_edges'].shape, (2, 3))

    def test_empty_keep_returns_empty_tensors(self):
        sub = self._subset(self.targets, keep_v_pos=[], keep_p_pos=[])
        self.assertEqual(sub['vaamr_idx'].numel(), 0)
        self.assertEqual(sub['purer_idx'].numel(), 0)

    def test_no_purer_key_in_targets(self):
        """_subset_targets should not crash when purer_idx absent."""
        targets_no_purer = {k: v for k, v in self.targets.items()
                             if k not in ('purer_idx', 'purer_label')}
        sub = self._subset(targets_no_purer, keep_v_pos=[0], keep_p_pos=[])
        self.assertEqual(sub['vaamr_idx'].tolist(), [10])


# ---------------------------------------------------------------------------
# assemble_targets
# ---------------------------------------------------------------------------

class TestAssembleTargets(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = synthetic_df(n_sessions=2)
        self.cfg = GnnLayerConfig(
            objectives=['soft_vaamr', 'progression', 'purer'],
            knn_k=2, cache_embeddings=False, hidden_dim=8,
        )
        self.g, _ = _build_graph(self.df, self.cfg)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _soft(self):
        from gnn_layer.soft_labels import build_soft_targets
        return build_soft_targets(self.df)

    def test_vaamr_idx_and_mix_present(self):
        from gnn_layer.classifier.train import assemble_targets
        t = assemble_targets(self.g, self._soft(), self.cfg, df_all=self.df)
        self.assertIn('vaamr_idx', t)
        self.assertIn('vaamr_mix', t)

    def test_vaamr_mix_shape(self):
        from gnn_layer.classifier.train import assemble_targets
        t = assemble_targets(self.g, self._soft(), self.cfg, df_all=self.df)
        n = t['vaamr_idx'].numel()
        self.assertEqual(t['vaamr_mix'].shape, (n, 5))

    def test_prog_val_shape_matches_vaamr_idx(self):
        from gnn_layer.classifier.train import assemble_targets
        t = assemble_targets(self.g, self._soft(), self.cfg, df_all=self.df)
        self.assertEqual(t['prog_val'].shape, t['vaamr_idx'].shape)

    def test_purer_idx_present_when_purer_in_objectives(self):
        from gnn_layer.classifier.train import assemble_targets
        t = assemble_targets(self.g, self._soft(), self.cfg, df_all=self.df)
        self.assertIn('purer_idx', t)
        self.assertIn('purer_label', t)
        self.assertGreater(t['purer_idx'].numel(), 0)

    def test_purer_absent_without_objective(self):
        from gnn_layer.classifier.train import assemble_targets
        cfg_no_purer = GnnLayerConfig(
            objectives=['soft_vaamr', 'progression'], knn_k=2, cache_embeddings=False
        )
        t = assemble_targets(self.g, self._soft(), cfg_no_purer, df_all=self.df)
        self.assertNotIn('purer_idx', t)

    def test_pos_and_neg_edges_present(self):
        from gnn_layer.classifier.train import assemble_targets
        t = assemble_targets(self.g, self._soft(), self.cfg, df_all=self.df)
        self.assertIn('pos_edges', t)
        self.assertIn('neg_edges', t)

    def test_contrast_idx_and_label_match_vaamr(self):
        import torch
        from gnn_layer.classifier.train import assemble_targets
        t = assemble_targets(self.g, self._soft(), self.cfg, df_all=self.df)
        self.assertEqual(t['contrast_idx'].tolist(), t['vaamr_idx'].tolist())
        self.assertEqual(t['contrast_label'].shape, t['vaamr_idx'].shape)

    def test_empty_soft_targets_gives_zero_length_tensors(self):
        from gnn_layer.classifier.train import assemble_targets
        t = assemble_targets(self.g, {}, self.cfg, df_all=self.df)
        self.assertEqual(t['vaamr_idx'].numel(), 0)
        self.assertEqual(t['vaamr_mix'].shape, (0, 5))


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

class TestTrainModel(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = synthetic_df(n_sessions=1)
        self.cfg = GnnLayerConfig(
            objectives=['soft_vaamr', 'progression', 'purer'],
            knn_k=2, cache_embeddings=False, hidden_dim=8, n_layers=1,
            epochs=5, seed=1,
        )
        self.g, _ = _build_graph(self.df, self.cfg)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _targets(self):
        from gnn_layer.soft_labels import build_soft_targets
        from gnn_layer.classifier.train import assemble_targets
        soft = build_soft_targets(self.df)
        return assemble_targets(self.g, soft, self.cfg, df_all=self.df)

    def test_returns_model_and_metrics(self):
        from gnn_layer.classifier.train import train_model
        model, metrics = train_model(self.g, self._targets(), self.cfg)
        self.assertIsNotNone(model)
        self.assertIsInstance(metrics, dict)

    def test_metrics_contains_expected_keys(self):
        from gnn_layer.classifier.train import train_model
        _, metrics = train_model(self.g, self._targets(), self.cfg)
        self.assertIn('best_loss', metrics)
        self.assertIn('epochs_run', metrics)
        self.assertIn('final_terms', metrics)

    def test_best_loss_is_float(self):
        from gnn_layer.classifier.train import train_model
        _, metrics = train_model(self.g, self._targets(), self.cfg)
        self.assertIsInstance(metrics['best_loss'], float)

    def test_epochs_run_within_budget(self):
        from gnn_layer.classifier.train import train_model
        _, metrics = train_model(self.g, self._targets(), self.cfg)
        self.assertGreaterEqual(metrics['epochs_run'], 1)
        self.assertLessEqual(metrics['epochs_run'], self.cfg.epochs)

    def test_final_terms_keys(self):
        from gnn_layer.classifier.train import train_model
        _, metrics = train_model(self.g, self._targets(), self.cfg)
        # objectives present → their loss keys should appear in final_terms
        for term in ('soft_vaamr', 'progression', 'purer', 'total'):
            self.assertIn(term, metrics['final_terms'])

    def test_early_stop_fires_with_tiny_patience(self):
        """patience=1 must stop well before epochs budget."""
        from gnn_layer.classifier.train import train_model
        cfg_fast = GnnLayerConfig(
            objectives=['soft_vaamr'], knn_k=2, cache_embeddings=False,
            hidden_dim=8, n_layers=1, epochs=50, seed=1, patience=1,
        )
        _, metrics = train_model(self.g, self._targets(), cfg_fast)
        self.assertLess(metrics['epochs_run'], 50)


# ---------------------------------------------------------------------------
# export_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------

class TestCheckpointRoundtrip(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = synthetic_df(n_sessions=1)
        self.cfg = GnnLayerConfig(
            objectives=['soft_vaamr', 'progression', 'purer'],
            knn_k=2, cache_embeddings=False, hidden_dim=8, n_layers=1,
            epochs=3, seed=1,
        )
        self.g, _ = _build_graph(self.df, self.cfg)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _trained_model(self):
        from gnn_layer.soft_labels import build_soft_targets
        from gnn_layer.classifier.train import assemble_targets, train_model
        soft = build_soft_targets(self.df)
        t = assemble_targets(self.g, soft, self.cfg, df_all=self.df)
        model, metrics = train_model(self.g, t, self.cfg)
        return model, metrics

    def test_weights_file_created(self):
        from gnn_layer.classifier.train import export_checkpoint
        model, metrics = self._trained_model()
        wp = export_checkpoint(model, self.cfg, self.tmp, metrics)
        self.assertTrue(os.path.isfile(wp))
        self.assertEqual(os.path.basename(wp), 'weights.pt')

    def test_manifest_file_created(self):
        from gnn_layer.classifier.train import export_checkpoint
        model, metrics = self._trained_model()
        export_checkpoint(model, self.cfg, self.tmp, metrics)
        manifest_p = os.path.join(self.tmp, 'manifest.json')
        self.assertTrue(os.path.isfile(manifest_p))

    def test_manifest_contains_seed(self):
        import json
        from gnn_layer.classifier.train import export_checkpoint
        model, metrics = self._trained_model()
        export_checkpoint(model, self.cfg, self.tmp, metrics)
        manifest_p = os.path.join(self.tmp, 'manifest.json')
        data = json.load(open(manifest_p))
        self.assertIn('seed', data)
        self.assertEqual(int(data['seed']), self.cfg.seed)

    def test_loaded_weights_match_saved(self):
        import torch
        from gnn_layer.classifier.train import export_checkpoint, load_checkpoint
        model, metrics = self._trained_model()
        export_checkpoint(model, self.cfg, self.tmp, metrics)
        model2 = load_checkpoint(self.tmp, self.g, self.cfg)
        s1 = {k: v.cpu() for k, v in model.state_dict().items()}
        s2 = {k: v.cpu() for k, v in model2.state_dict().items()}
        self.assertEqual(set(s1.keys()), set(s2.keys()))
        for k in s1:
            self.assertTrue(torch.allclose(s1[k], s2[k]),
                            f"Mismatch in layer {k}")

    def test_load_nonexistent_raises(self):
        from gnn_layer.classifier.train import load_checkpoint
        with self.assertRaises(Exception):
            load_checkpoint(os.path.join(self.tmp, 'nonexistent'), self.g, self.cfg)


# ---------------------------------------------------------------------------
# crossval_predictions
# ---------------------------------------------------------------------------

class TestCrossvalPredictions(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = synthetic_df(n_sessions=2)
        self.cfg = GnnLayerConfig(
            objectives=['soft_vaamr', 'progression', 'purer'],
            knn_k=2, cache_embeddings=False, hidden_dim=8, n_layers=1,
            epochs=3, seed=42, validation_folds=2,
        )
        self.g, _ = _build_graph(self.df, self.cfg)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _targets(self):
        from gnn_layer.soft_labels import build_soft_targets
        from gnn_layer.classifier.train import assemble_targets
        soft = build_soft_targets(self.df)
        return assemble_targets(self.g, soft, self.cfg, df_all=self.df)

    def test_returns_vaamr_and_purer_keys(self):
        from gnn_layer.classifier.train import crossval_predictions
        t = self._targets()
        cv = crossval_predictions(self.g, t, self.cfg)
        self.assertIn('vaamr', cv)
        self.assertIn('purer', cv)

    def test_every_labeled_vaamr_node_predicted_once(self):
        from gnn_layer.classifier.train import crossval_predictions
        t = self._targets()
        cv = crossval_predictions(self.g, t, self.cfg)
        n_labeled = t['vaamr_idx'].numel()
        self.assertEqual(len(cv['vaamr']), n_labeled,
                         "Every labeled node must appear exactly once across folds")

    def test_no_duplicate_segment_ids_in_vaamr(self):
        from gnn_layer.classifier.train import crossval_predictions
        t = self._targets()
        cv = crossval_predictions(self.g, t, self.cfg)
        ids = [sid for sid, _ in cv['vaamr']]
        self.assertEqual(len(ids), len(set(ids)),
                         "Each segment must be held out in exactly one fold")

    def test_every_labeled_purer_node_predicted_once(self):
        from gnn_layer.classifier.train import crossval_predictions
        t = self._targets()
        cv = crossval_predictions(self.g, t, self.cfg)
        n_labeled_p = t['purer_idx'].numel()
        self.assertEqual(len(cv['purer']), n_labeled_p)

    def test_predictions_are_valid_class_indices(self):
        from gnn_layer.classifier.train import crossval_predictions
        t = self._targets()
        cv = crossval_predictions(self.g, t, self.cfg)
        for _, pred in cv['vaamr']:
            self.assertIn(pred, range(5))
        for _, pred in cv['purer']:
            self.assertIn(pred, range(5))

    def test_determinism_with_same_seed(self):
        from gnn_layer.classifier.train import crossval_predictions
        t = self._targets()
        cv1 = crossval_predictions(self.g, t, self.cfg)
        cv2 = crossval_predictions(self.g, t, self.cfg)
        # sort both to compare irrespective of order
        v1 = sorted(cv1['vaamr'], key=lambda x: x[0])
        v2 = sorted(cv2['vaamr'], key=lambda x: x[0])
        self.assertEqual(v1, v2)

    def test_empty_targets_returns_empty(self):
        import torch
        from gnn_layer.classifier.train import crossval_predictions, assemble_targets
        t_empty = assemble_targets(self.g, {}, self.cfg, df_all=None)
        cv = crossval_predictions(self.g, t_empty, self.cfg)
        self.assertEqual(cv['vaamr'], [])
        self.assertEqual(cv['purer'], [])


if __name__ == '__main__':
    unittest.main()
