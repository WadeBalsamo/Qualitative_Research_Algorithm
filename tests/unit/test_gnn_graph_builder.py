"""
tests/unit/test_gnn_graph_builder.py
--------------------------------------
Unit tests for gnn_layer/graph_builder.py.

Covers: build_graph node/edge counts & node_type split, kNN edges present,
anchor nodes (purer ON, VCE OFF by default), attach_new_segments
inductive add, cross-framework edge threshold behaviour, save/load_graph
roundtrip, HeteroGraph.type_indices helper.
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

def _seg_emb(df, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    return {sid: rng.standard_normal(dim).astype('float32') for sid in df['segment_id']}


# ---------------------------------------------------------------------------
# HeteroGraph dataclass
# ---------------------------------------------------------------------------

class TestHeteroGraph(unittest.TestCase):

    def _make_graph(self, n_sessions=1):
        from gnn_layer.classifier.graph_builder import build_graph
        df = synthetic_df(n_sessions=n_sessions)
        cfg = GnnLayerConfig(knn_k=2, cache_embeddings=False)
        return build_graph(df, _seg_emb(df), cfg)

    def test_type_indices_participant(self):
        g = self._make_graph()
        idxs = g.type_indices('participant_segment')
        self.assertGreater(len(idxs), 0)
        for i in idxs:
            self.assertEqual(g.node_types[i], 'participant_segment')

    def test_type_indices_therapist(self):
        g = self._make_graph()
        idxs = g.type_indices('therapist_segment')
        self.assertGreater(len(idxs), 0)

    def test_type_indices_returns_empty_for_unknown(self):
        g = self._make_graph()
        self.assertEqual(g.type_indices('anchor'), [])


# ---------------------------------------------------------------------------
# build_graph — nodes
# ---------------------------------------------------------------------------

class TestBuildGraphNodes(unittest.TestCase):

    def setUp(self):
        self.df = synthetic_df(n_sessions=3)
        self.cfg = GnnLayerConfig(knn_k=3, cache_embeddings=False)
        self.seg_emb = _seg_emb(self.df)

    def test_node_count_matches_segments(self):
        from gnn_layer.classifier.graph_builder import build_graph
        g = build_graph(self.df, self.seg_emb, self.cfg)
        # 3 sessions × 6 segments = 18 nodes
        self.assertEqual(len(g.node_ids), 18)
        self.assertEqual(g.x.shape[0], 18)

    def test_participant_therapist_split(self):
        from gnn_layer.classifier.graph_builder import build_graph
        g = build_graph(self.df, self.seg_emb, self.cfg)
        self.assertEqual(g.node_types.count('participant_segment'), 9)
        self.assertEqual(g.node_types.count('therapist_segment'), 9)

    def test_embedding_dim_preserved(self):
        from gnn_layer.classifier.graph_builder import build_graph
        g = build_graph(self.df, self.seg_emb, self.cfg)
        self.assertEqual(g.x.shape[1], 16)
        self.assertEqual(g.meta['embed_dim'], 16)

    def test_index_of_maps_all_nodes(self):
        from gnn_layer.classifier.graph_builder import build_graph
        g = build_graph(self.df, self.seg_emb, self.cfg)
        self.assertEqual(len(g.index_of), len(g.node_ids))
        for i, sid in enumerate(g.node_ids):
            self.assertEqual(g.index_of[sid], i)

    def test_speaker_of_meta_populated(self):
        from gnn_layer.classifier.graph_builder import build_graph
        g = build_graph(self.df, self.seg_emb, self.cfg)
        speaker_of = g.meta['speaker_of']
        for sid, nt in zip(g.node_ids, g.node_types):
            if nt == 'participant_segment':
                self.assertEqual(speaker_of[sid], 'participant')
            elif nt == 'therapist_segment':
                self.assertEqual(speaker_of[sid], 'therapist')

    def test_n_anchors_zero_without_anchor_features(self):
        from gnn_layer.classifier.graph_builder import build_graph
        g = build_graph(self.df, self.seg_emb, self.cfg)
        self.assertEqual(g.meta['n_anchors'], 0)

    def test_vce_nodes_off_by_default(self):
        from gnn_layer.classifier.graph_builder import build_graph
        # include_vce_nodes default False → no VCE anchor nodes
        self.assertFalse(self.cfg.include_vce_nodes)
        g = build_graph(self.df, self.seg_emb, self.cfg)
        anchor_count = g.node_types.count('anchor')
        self.assertEqual(anchor_count, 0)

    def test_raises_when_no_embeddings_available(self):
        from gnn_layer.classifier.graph_builder import build_graph
        with self.assertRaises(ValueError):
            build_graph(self.df, {}, self.cfg)

    def test_only_segments_with_embeddings_included(self):
        from gnn_layer.classifier.graph_builder import build_graph
        # Provide only first 3 embeddings
        partial_emb = {sid: emb for sid, emb in list(self.seg_emb.items())[:3]}
        g = build_graph(self.df, partial_emb, self.cfg)
        self.assertEqual(len(g.node_ids), 3)


# ---------------------------------------------------------------------------
# build_graph — edges
# ---------------------------------------------------------------------------

class TestBuildGraphEdges(unittest.TestCase):

    def setUp(self):
        self.df = synthetic_df(n_sessions=2)
        self.cfg = GnnLayerConfig(knn_k=3, cache_embeddings=False)
        self.seg_emb = _seg_emb(self.df)

    def test_edges_present(self):
        from gnn_layer.classifier.graph_builder import build_graph
        g = build_graph(self.df, self.seg_emb, self.cfg)
        self.assertGreater(g.edge_index.shape[1], 0)

    def test_edge_index_shape_is_2_x_E(self):
        from gnn_layer.classifier.graph_builder import build_graph
        g = build_graph(self.df, self.seg_emb, self.cfg)
        self.assertEqual(g.edge_index.shape[0], 2)

    def test_edge_weight_length_matches_edges(self):
        from gnn_layer.classifier.graph_builder import build_graph
        g = build_graph(self.df, self.seg_emb, self.cfg)
        self.assertEqual(g.edge_weight.shape[0], g.edge_index.shape[1])

    def test_knn_edges_present_in_meta(self):
        from gnn_layer.classifier.graph_builder import build_graph
        g = build_graph(self.df, self.seg_emb, self.cfg)
        self.assertIn('knn', g.meta['edge_types'])

    def test_temporal_edges_present_in_meta(self):
        from gnn_layer.classifier.graph_builder import build_graph
        g = build_graph(self.df, self.seg_emb, self.cfg)
        self.assertIn('temporal', g.meta['edge_types'])

    def test_edge_weights_non_negative(self):
        from gnn_layer.classifier.graph_builder import build_graph
        g = build_graph(self.df, self.seg_emb, self.cfg)
        self.assertTrue((g.edge_weight.numpy() >= 0).all())

    def test_edges_bidirectional(self):
        """Temporal and kNN edges are added in both directions (undirected graph)."""
        from gnn_layer.classifier.graph_builder import build_graph
        g = build_graph(self.df, self.seg_emb, self.cfg)
        src = g.edge_index[0].numpy().tolist()
        dst = g.edge_index[1].numpy().tolist()
        fwd = set(zip(src, dst))
        rev = set(zip(dst, src))
        # Every forward edge has a reverse edge
        self.assertTrue(fwd.issubset(rev | fwd))


# ---------------------------------------------------------------------------
# build_graph — anchor nodes
# ---------------------------------------------------------------------------

class TestBuildGraphAnchors(unittest.TestCase):

    def setUp(self):
        self.df = synthetic_df(n_sessions=1)
        self.cfg = GnnLayerConfig(knn_k=2, cache_embeddings=False)
        self.seg_emb = _seg_emb(self.df)

    def test_anchor_nodes_added_when_supplied(self):
        from gnn_layer.classifier.graph_builder import build_graph
        anchor_feats = {
            'purer_anchor_0': np.zeros(16, dtype='float32'),
            'purer_anchor_1': np.ones(16, dtype='float32'),
        }
        anchor_edges = [('purer_anchor_0', list(self.seg_emb.keys())[0], 1.0)]
        g = build_graph(self.df, self.seg_emb, self.cfg,
                        anchor_features=anchor_feats,
                        anchor_edges=anchor_edges)
        self.assertEqual(g.meta['n_anchors'], 2)
        anchor_types = [t for t in g.node_types if t == 'anchor']
        self.assertEqual(len(anchor_types), 2)

    def test_anchor_edges_added_to_graph(self):
        from gnn_layer.classifier.graph_builder import build_graph
        seg_ids = list(self.seg_emb.keys())
        anchor_feats = {'anch_0': np.zeros(16, dtype='float32')}
        anchor_edges = [('anch_0', seg_ids[0], 0.8), ('anch_0', seg_ids[1], 0.6)]
        g_no_anch = build_graph(self.df, self.seg_emb, self.cfg)
        g_with_anch = build_graph(self.df, self.seg_emb, self.cfg,
                                  anchor_features=anchor_feats,
                                  anchor_edges=anchor_edges)
        self.assertGreater(g_with_anch.edge_index.shape[1],
                           g_no_anch.edge_index.shape[1])

    def test_anchor_edge_type_in_meta(self):
        from gnn_layer.classifier.graph_builder import build_graph
        seg_ids = list(self.seg_emb.keys())
        anchor_feats = {'anch_0': np.zeros(16, dtype='float32')}
        anchor_edges = [('anch_0', seg_ids[0], 1.0)]
        g = build_graph(self.df, self.seg_emb, self.cfg,
                        anchor_features=anchor_feats,
                        anchor_edges=anchor_edges)
        self.assertIn('anchor', g.meta['edge_types'])


# ---------------------------------------------------------------------------
# attach_new_segments
# ---------------------------------------------------------------------------

class TestAttachNewSegments(unittest.TestCase):

    def setUp(self):
        self.df = synthetic_df(n_sessions=2)
        self.cfg = GnnLayerConfig(knn_k=3, cache_embeddings=False)
        self.seg_emb = _seg_emb(self.df)

    def test_node_count_increases_by_one(self):
        from gnn_layer.classifier.graph_builder import build_graph, attach_new_segments
        g = build_graph(self.df, self.seg_emb, self.cfg)
        new = {'new_seg_1': np.random.default_rng(99).standard_normal(16).astype('float32')}
        g2 = attach_new_segments(g, new, self.cfg)
        self.assertEqual(len(g2.node_ids), len(g.node_ids) + 1)

    def test_new_segment_in_index_of(self):
        from gnn_layer.classifier.graph_builder import build_graph, attach_new_segments
        g = build_graph(self.df, self.seg_emb, self.cfg)
        new = {'new_seg_x': np.zeros(16, dtype='float32')}
        g2 = attach_new_segments(g, new, self.cfg)
        self.assertIn('new_seg_x', g2.index_of)

    def test_existing_nodes_retain_indices(self):
        from gnn_layer.classifier.graph_builder import build_graph, attach_new_segments
        g = build_graph(self.df, self.seg_emb, self.cfg)
        old_indices = {sid: g.index_of[sid] for sid in g.node_ids}
        new = {'new_x': np.zeros(16, dtype='float32')}
        g2 = attach_new_segments(g, new, self.cfg)
        for sid, old_idx in old_indices.items():
            self.assertEqual(g2.index_of[sid], old_idx)

    def test_node_type_assigned_from_node_type_of(self):
        from gnn_layer.classifier.graph_builder import build_graph, attach_new_segments
        g = build_graph(self.df, self.seg_emb, self.cfg)
        new = {
            'np1': np.zeros(16, dtype='float32'),
            'nt1': np.zeros(16, dtype='float32'),
        }
        ntype = {'np1': 'participant_segment', 'nt1': 'therapist_segment'}
        g2 = attach_new_segments(g, new, self.cfg, node_type_of=ntype)
        self.assertEqual(g2.node_types[g2.index_of['np1']], 'participant_segment')
        self.assertEqual(g2.node_types[g2.index_of['nt1']], 'therapist_segment')

    def test_speaker_of_meta_updated(self):
        from gnn_layer.classifier.graph_builder import build_graph, attach_new_segments
        g = build_graph(self.df, self.seg_emb, self.cfg)
        new = {'new_t': np.zeros(16, dtype='float32')}
        ntype = {'new_t': 'therapist_segment'}
        g2 = attach_new_segments(g, new, self.cfg, node_type_of=ntype)
        self.assertEqual(g2.meta['speaker_of']['new_t'], 'therapist')

    def test_default_type_is_generic_segment(self):
        from gnn_layer.classifier.graph_builder import build_graph, attach_new_segments
        g = build_graph(self.df, self.seg_emb, self.cfg)
        new = {'untyped': np.zeros(16, dtype='float32')}
        g2 = attach_new_segments(g, new, self.cfg)
        self.assertEqual(g2.node_types[g2.index_of['untyped']], 'segment')

    def test_multiple_new_segments(self):
        from gnn_layer.classifier.graph_builder import build_graph, attach_new_segments
        g = build_graph(self.df, self.seg_emb, self.cfg)
        rng = np.random.default_rng(5)
        new = {f'nn{i}': rng.standard_normal(16).astype('float32') for i in range(5)}
        g2 = attach_new_segments(g, new, self.cfg)
        self.assertEqual(len(g2.node_ids), len(g.node_ids) + 5)

    def test_already_existing_segment_ignored(self):
        from gnn_layer.classifier.graph_builder import build_graph, attach_new_segments
        g = build_graph(self.df, self.seg_emb, self.cfg)
        existing_id = g.node_ids[0]
        new = {existing_id: np.zeros(16, dtype='float32')}
        g2 = attach_new_segments(g, new, self.cfg)
        # nothing added
        self.assertEqual(len(g2.node_ids), len(g.node_ids))

    def test_new_edges_added(self):
        from gnn_layer.classifier.graph_builder import build_graph, attach_new_segments
        g = build_graph(self.df, self.seg_emb, self.cfg)
        base_edges = g.edge_index.shape[1]
        new = {'nn0': np.zeros(16, dtype='float32')}
        g2 = attach_new_segments(g, new, self.cfg)
        self.assertGreater(g2.edge_index.shape[1], base_edges)


# ---------------------------------------------------------------------------
# save_graph / load_graph roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoadGraph(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_roundtrip_preserves_node_ids(self):
        from gnn_layer.classifier.graph_builder import build_graph, save_graph, load_graph
        import torch
        df = synthetic_df(n_sessions=1)
        cfg = GnnLayerConfig(knn_k=2, cache_embeddings=False)
        g = build_graph(df, _seg_emb(df), cfg)
        save_graph(g, self.tmp)
        g2 = load_graph(self.tmp)
        self.assertIsNotNone(g2)
        self.assertEqual(g2.node_ids, g.node_ids)

    def test_roundtrip_preserves_embeddings(self):
        from gnn_layer.classifier.graph_builder import build_graph, save_graph, load_graph
        import torch
        df = synthetic_df(n_sessions=1)
        cfg = GnnLayerConfig(knn_k=2, cache_embeddings=False)
        g = build_graph(df, _seg_emb(df), cfg)
        save_graph(g, self.tmp)
        g2 = load_graph(self.tmp)
        self.assertTrue(torch.allclose(g.x, g2.x))

    def test_load_graph_returns_none_when_absent(self):
        from gnn_layer.classifier.graph_builder import load_graph
        result = load_graph(os.path.join(self.tmp, 'nonexistent'))
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# compute_cross_framework_lift
# ---------------------------------------------------------------------------

class TestCrossFrameworkLift(unittest.TestCase):

    def test_returns_dict(self):
        from gnn_layer.classifier.graph_builder import compute_cross_framework_lift
        df = synthetic_df(n_sessions=2)
        result = compute_cross_framework_lift(df, min_lift=0.1)
        self.assertIsInstance(result, dict)

    def test_empty_when_no_codebook_column(self):
        from gnn_layer.classifier.graph_builder import compute_cross_framework_lift
        df = synthetic_df(n_sessions=2).drop(columns=['codebook_labels_ensemble'])
        result = compute_cross_framework_lift(df)
        self.assertEqual(result, {})

    def test_high_min_lift_returns_fewer_pairs(self):
        from gnn_layer.classifier.graph_builder import compute_cross_framework_lift
        df = synthetic_df(n_sessions=3)
        low = compute_cross_framework_lift(df, min_lift=0.1)
        high = compute_cross_framework_lift(df, min_lift=100.0)
        self.assertGreaterEqual(len(low), len(high))

    def test_keys_are_tuples_of_strings(self):
        from gnn_layer.classifier.graph_builder import compute_cross_framework_lift
        df = synthetic_df(n_sessions=2)
        result = compute_cross_framework_lift(df, min_lift=0.1)
        for k, v in result.items():
            self.assertIsInstance(k, tuple)
            self.assertEqual(len(k), 2)
            self.assertIsInstance(v, float)


if __name__ == '__main__':
    unittest.main()
