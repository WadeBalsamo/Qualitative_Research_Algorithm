"""
tests/unit/test_gnn_ablation.py
--------------------------------
Unit tests for gnn_layer/ablation.py:
  - run_ablation (loss-delta sign/structure)
  - discover_subtypes
All hermetic: embedding_patch prevents model downloads; small config.
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
from gnn_layer.classifier import ablation as _abl


def _hermetic_cfg(**kw):
    defaults = dict(
        enabled=True, hidden_dim=8, n_layers=2, knn_k=2,
        epochs=5, patience=5, n_motif_clusters=3,
        cache_embeddings=False, seed=0,
        interpret_against_cf_ic=False,
        objectives=['soft_vaamr', 'progression', 'purer'],
        run_gnn_ablation=True,
    )
    defaults.update(kw)
    return GnnLayerConfig(**defaults)


def _build_graph_and_targets(df, cfg):
    """Build a minimal graph + targets for ablation/subtype tests."""
    from gnn_layer.classifier import graph_builder as _gb
    from gnn_layer import soft_labels as _sl
    from gnn_layer.classifier import train as _train
    rng = np.random.default_rng(42)
    seg_emb = {str(sid): rng.standard_normal(16).astype('float32')
               for sid in df['segment_id']}
    graph = _gb.build_graph(df, seg_emb, cfg)
    soft = _sl.build_soft_targets(df, cfg.label_mode)
    targets = _train.assemble_targets(graph, soft, cfg, df_all=df)
    return graph, targets, seg_emb


class TestRunAblation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = synthetic_df(n_sessions=2)
        self.cfg = _hermetic_cfg()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ablation_result_keys(self):
        with embedding_patch(dim=16):
            graph, targets, _ = _build_graph_and_targets(self.df, self.cfg)
            result = _abl.run_ablation(graph, targets, self.cfg, ablate='purer')
        self.assertIn('ablate', result)
        self.assertIn('best_loss_full', result)
        self.assertIn('best_loss_ablated', result)
        self.assertIn('delta', result)
        self.assertIn('objective_removed', result)

    def test_ablate_field_matches_input(self):
        with embedding_patch(dim=16):
            graph, targets, _ = _build_graph_and_targets(self.df, self.cfg)
            result = _abl.run_ablation(graph, targets, self.cfg, ablate='purer')
        self.assertEqual(result['ablate'], 'purer')
        self.assertEqual(result['objective_removed'], 'purer')

    def test_delta_is_float(self):
        with embedding_patch(dim=16):
            graph, targets, _ = _build_graph_and_targets(self.df, self.cfg)
            result = _abl.run_ablation(graph, targets, self.cfg, ablate='purer')
        self.assertIsInstance(result['delta'], float)

    def test_delta_is_best_loss_ablated_minus_full(self):
        """delta must equal best_loss_ablated - best_loss_full."""
        with embedding_patch(dim=16):
            graph, targets, _ = _build_graph_and_targets(self.df, self.cfg)
            result = _abl.run_ablation(graph, targets, self.cfg, ablate='purer')
        computed = result['best_loss_ablated'] - result['best_loss_full']
        self.assertAlmostEqual(result['delta'], computed, places=5)

    def test_ablate_unknown_key_returns_none_objective(self):
        """If ablate is not in _ABLATE_TO_OBJECTIVE, objective_removed is None."""
        with embedding_patch(dim=16):
            graph, targets, _ = _build_graph_and_targets(self.df, self.cfg)
            result = _abl.run_ablation(graph, targets, self.cfg, ablate='nonexistent')
        self.assertIsNone(result['objective_removed'])

    def test_ablate_vce_objective_name(self):
        cfg = _hermetic_cfg(objectives=['soft_vaamr', 'progression', 'vce_multilabel'])
        with embedding_patch(dim=16):
            graph, targets, _ = _build_graph_and_targets(self.df, cfg)
            result = _abl.run_ablation(graph, targets, cfg, ablate='vce')
        self.assertEqual(result['objective_removed'], 'vce_multilabel')


class TestDiscoverSubtypes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = _hermetic_cfg()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _seg_embeddings(self, df):
        rng = np.random.default_rng(7)
        return {str(sid): rng.standard_normal(16).astype('float32')
                for sid in df['segment_id']}

    def test_returns_dict(self):
        df = make_master_df(n_sessions=2, n_participants=2)
        emb = self._seg_embeddings(df)
        out = _abl.discover_subtypes(emb, df, by='vaamr_stage', k_per_group=2, config=self.cfg)
        self.assertIsInstance(out, dict)

    def test_keys_are_stage_strings(self):
        df = make_master_df(n_sessions=2, n_participants=2)
        emb = self._seg_embeddings(df)
        out = _abl.discover_subtypes(emb, df, by='vaamr_stage', k_per_group=2, config=self.cfg)
        # Keys should correspond to VAAMR stage values present in df
        for k in out.keys():
            self.assertIsInstance(k, str)

    def test_each_stage_has_clusters(self):
        df = make_master_df(n_sessions=3, n_participants=2)
        emb = self._seg_embeddings(df)
        out = _abl.discover_subtypes(emb, df, by='vaamr_stage', k_per_group=3, config=self.cfg)
        for group, info in out.items():
            self.assertIn('n_total', info)
            self.assertIn('clusters', info)
            self.assertIsInstance(info['clusters'], dict)

    def test_cluster_has_n_and_exemplars(self):
        df = make_master_df(n_sessions=2, n_participants=2)
        emb = self._seg_embeddings(df)
        out = _abl.discover_subtypes(emb, df, by='vaamr_stage', k_per_group=2, config=self.cfg)
        for group, info in out.items():
            for ci, cd in info['clusters'].items():
                self.assertIn('n', cd)
                self.assertIn('exemplar_ids', cd)

    def test_purer_mode(self):
        df = make_master_df(n_sessions=2, n_participants=2)
        emb = self._seg_embeddings(df)
        out = _abl.discover_subtypes(emb, df, by='purer_move', k_per_group=2, config=self.cfg)
        self.assertIsInstance(out, dict)

    def test_missing_column_returns_empty(self):
        df = make_master_df(n_sessions=2, n_participants=1)
        df = df.drop(columns=['final_label'], errors='ignore')
        emb = self._seg_embeddings(df)
        out = _abl.discover_subtypes(emb, df, by='vaamr_stage', config=self.cfg)
        self.assertEqual(out, {})


if __name__ == '__main__':
    unittest.main()
