"""
tests/unit/test_gnn_config.py
-----------------------------
Unit tests for gnn_layer/config.py — GnnLayerConfig safety defaults and field types.
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


class TestGnnLayerConfigDefaults(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = GnnLayerConfig()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # --- safety defaults ---
    def test_enabled_is_true(self):
        # GNN is ON by default (degrades gracefully); confirmed in source comment
        self.assertTrue(self.cfg.enabled)

    def test_gnn_authoritative_is_false(self):
        # Authoritative mode only after human review
        self.assertFalse(self.cfg.gnn_authoritative)

    def test_include_vce_nodes_is_false(self):
        self.assertFalse(self.cfg.include_vce_nodes)

    def test_irr_target(self):
        self.assertAlmostEqual(self.cfg.irr_target, 0.70, places=6)

    def test_label_mode_is_weak(self):
        self.assertEqual(self.cfg.label_mode, 'weak')

    def test_validation_folds_is_5(self):
        self.assertEqual(self.cfg.validation_folds, 5)

    def test_produce_consensus_labels_is_true(self):
        self.assertTrue(self.cfg.produce_consensus_labels)

    # --- field types ---
    def test_enabled_is_bool(self):
        self.assertIsInstance(self.cfg.enabled, bool)

    def test_gnn_authoritative_is_bool(self):
        self.assertIsInstance(self.cfg.gnn_authoritative, bool)

    def test_include_vce_nodes_is_bool(self):
        self.assertIsInstance(self.cfg.include_vce_nodes, bool)

    def test_irr_target_is_float(self):
        self.assertIsInstance(self.cfg.irr_target, float)

    def test_label_mode_is_str(self):
        self.assertIsInstance(self.cfg.label_mode, str)

    def test_validation_folds_is_int(self):
        self.assertIsInstance(self.cfg.validation_folds, int)

    def test_produce_consensus_labels_is_bool(self):
        self.assertIsInstance(self.cfg.produce_consensus_labels, bool)

    def test_objectives_is_list(self):
        self.assertIsInstance(self.cfg.objectives, list)

    def test_default_objectives_nonempty(self):
        self.assertGreater(len(self.cfg.objectives), 0)

    def test_embedding_model_is_str(self):
        self.assertIsInstance(self.cfg.embedding_model, str)

    def test_n_motif_clusters_is_int(self):
        self.assertIsInstance(self.cfg.n_motif_clusters, int)

    def test_n_latent_factors_is_int(self):
        self.assertIsInstance(self.cfg.n_latent_factors, int)

    def test_knn_k_is_int(self):
        self.assertIsInstance(self.cfg.knn_k, int)

    def test_hidden_dim_is_int(self):
        self.assertIsInstance(self.cfg.hidden_dim, int)

    def test_epochs_is_int(self):
        self.assertIsInstance(self.cfg.epochs, int)

    def test_run_gnn_ablation_default_false(self):
        self.assertFalse(self.cfg.run_gnn_ablation)

    def test_test_vce_layer_default_false(self):
        self.assertFalse(self.cfg.test_vce_layer)

    def test_interpret_against_cf_ic_default_true(self):
        # Default is True per source; tests override with False to stay hermetic
        self.assertTrue(self.cfg.interpret_against_cf_ic)

    def test_include_purer_nodes_default_true(self):
        self.assertTrue(self.cfg.include_purer_nodes)

    # --- keyword-argument overrides work ---
    def test_override_enabled(self):
        cfg = GnnLayerConfig(enabled=False)
        self.assertFalse(cfg.enabled)

    def test_override_epochs(self):
        cfg = GnnLayerConfig(epochs=5)
        self.assertEqual(cfg.epochs, 5)

    def test_override_label_mode(self):
        cfg = GnnLayerConfig(label_mode='human')
        self.assertEqual(cfg.label_mode, 'human')

    def test_cache_embeddings_default_true(self):
        self.assertTrue(self.cfg.cache_embeddings)


if __name__ == '__main__':
    unittest.main()
