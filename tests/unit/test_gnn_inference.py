"""
tests/unit/test_gnn_inference.py
----------------------------------
Unit tests for gnn_layer/inference.py.

Covers: infer_segment_positions (mixtures sum to 1; correct keys/shapes),
infer_head_predictions (keys per head), build_cue_blocks_with_segments
(delegates to process.cue_blocks; returns list of dicts with expected keys;
empty when required columns absent), cue_block_embeddings (mean-pool; empty
input → empty), _graph_tensors_on_model_device (device alignment).
Embeddings are never downloaded — uses a pre-built tiny graph.
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

def _make_model_and_graph(objectives=None, n_sessions=1, dim=16, knn_k=2, seed=0):
    """Build a tiny trained model + graph without touching the embedding model."""
    from gnn_layer import graph_builder
    from gnn_layer.model import build_model
    if objectives is None:
        objectives = ['soft_vaamr', 'progression', 'purer']
    df = synthetic_df(n_sessions=n_sessions)
    cfg = GnnLayerConfig(objectives=objectives, knn_k=knn_k,
                         cache_embeddings=False, hidden_dim=8, n_layers=1)
    rng = np.random.default_rng(seed)
    seg_emb = {sid: rng.standard_normal(dim).astype('float32') for sid in df['segment_id']}
    g = graph_builder.build_graph(df, seg_emb, cfg)
    model = build_model(g, cfg)
    model.eval()
    return model, g, df, seg_emb, cfg


# ---------------------------------------------------------------------------
# _graph_tensors_on_model_device
# ---------------------------------------------------------------------------

class TestGraphTensorsOnModelDevice(unittest.TestCase):

    def test_returns_four_tensors(self):
        from gnn_layer.inference import _graph_tensors_on_model_device
        model, g, _, _, _ = _make_model_and_graph()
        x, ei, ew, eti = _graph_tensors_on_model_device(model, g)
        self.assertIsNotNone(x)
        self.assertIsNotNone(ei)
        # ew may be None if graph.edge_weight is None, but here it should be a tensor

    def test_tensors_on_cpu_for_cpu_model(self):
        import torch
        from gnn_layer.inference import _graph_tensors_on_model_device
        model, g, _, _, _ = _make_model_and_graph()
        x, ei, ew, eti = _graph_tensors_on_model_device(model, g)
        self.assertEqual(str(x.device), 'cpu')
        self.assertEqual(str(ei.device), 'cpu')

    def test_shapes_preserved(self):
        from gnn_layer.inference import _graph_tensors_on_model_device
        model, g, _, _, _ = _make_model_and_graph()
        x, ei, ew, eti = _graph_tensors_on_model_device(model, g)
        self.assertEqual(x.shape, g.x.shape)
        self.assertEqual(ei.shape, g.edge_index.shape)


# ---------------------------------------------------------------------------
# infer_segment_positions
# ---------------------------------------------------------------------------

class TestInferSegmentPositions(unittest.TestCase):

    def setUp(self):
        from gnn_layer.inference import infer_segment_positions
        self.model, self.g, self.df, _, self.cfg = _make_model_and_graph(
            objectives=['soft_vaamr', 'progression']
        )
        self._infer = infer_segment_positions

    def test_returns_dict(self):
        res = self._infer(self.model, self.g, self.cfg)
        self.assertIsInstance(res, dict)

    def test_expected_keys_present(self):
        res = self._infer(self.model, self.g, self.cfg)
        for k in ('segment_id', 'node_type', 'gnn_embedding',
                  'vaamr_mixture', 'progression_coord'):
            self.assertIn(k, res, f"Key '{k}' missing from infer_segment_positions output")

    def test_segment_id_length_matches_graph(self):
        res = self._infer(self.model, self.g, self.cfg)
        self.assertEqual(len(res['segment_id']), len(self.g.node_ids))

    def test_node_type_length_matches_graph(self):
        res = self._infer(self.model, self.g, self.cfg)
        self.assertEqual(len(res['node_type']), len(self.g.node_ids))

    def test_vaamr_mixture_shape(self):
        res = self._infer(self.model, self.g, self.cfg)
        N = len(self.g.node_ids)
        self.assertEqual(res['vaamr_mixture'].shape, (N, 5))

    def test_vaamr_mixtures_sum_to_one(self):
        res = self._infer(self.model, self.g, self.cfg)
        row_sums = res['vaamr_mixture'].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4,
                                   err_msg="VAAMR mixtures must sum to 1")

    def test_progression_coord_shape(self):
        res = self._infer(self.model, self.g, self.cfg)
        self.assertEqual(len(res['progression_coord']), len(self.g.node_ids))

    def test_gnn_embedding_shape(self):
        res = self._infer(self.model, self.g, self.cfg)
        N = len(self.g.node_ids)
        self.assertEqual(res['gnn_embedding'].shape, (N, 8))  # hidden_dim=8

    def test_progression_without_soft_vaamr(self):
        from gnn_layer.inference import infer_segment_positions
        from gnn_layer.model import build_model
        from gnn_layer import graph_builder
        df = synthetic_df(n_sessions=1)
        cfg = GnnLayerConfig(objectives=['progression'], knn_k=2,
                             cache_embeddings=False, hidden_dim=8)
        rng = np.random.default_rng(0)
        seg_emb = {sid: rng.standard_normal(16).astype('float32') for sid in df['segment_id']}
        g = graph_builder.build_graph(df, seg_emb, cfg)
        model = build_model(g, cfg)
        model.eval()
        res = infer_segment_positions(model, g, cfg)
        self.assertIsNone(res['vaamr_mixture'])
        self.assertEqual(len(res['progression_coord']), len(g.node_ids))

    def test_no_vaamr_or_progression_gives_zero_coord(self):
        from gnn_layer.inference import infer_segment_positions
        from gnn_layer.model import build_model
        from gnn_layer import graph_builder
        df = synthetic_df(n_sessions=1)
        cfg = GnnLayerConfig(objectives=['purer'], knn_k=2,
                             cache_embeddings=False, hidden_dim=8)
        rng = np.random.default_rng(0)
        seg_emb = {sid: rng.standard_normal(16).astype('float32') for sid in df['segment_id']}
        g = graph_builder.build_graph(df, seg_emb, cfg)
        model = build_model(g, cfg)
        model.eval()
        res = infer_segment_positions(model, g, cfg)
        self.assertIsNone(res['vaamr_mixture'])
        np.testing.assert_array_equal(res['progression_coord'],
                                      np.zeros(len(g.node_ids)))


# ---------------------------------------------------------------------------
# infer_head_predictions
# ---------------------------------------------------------------------------

class TestInferHeadPredictions(unittest.TestCase):

    def setUp(self):
        from gnn_layer.inference import infer_head_predictions
        self._infer = infer_head_predictions

    def test_vaamr_keys_present_with_soft_vaamr_head(self):
        model, g, _, _, cfg = _make_model_and_graph(objectives=['soft_vaamr', 'purer'])
        res = self._infer(model, g, cfg)
        self.assertIn('gnn_vaamr_pred', res)
        self.assertIn('gnn_vaamr_conf', res)

    def test_purer_keys_present_with_purer_head(self):
        model, g, _, _, cfg = _make_model_and_graph(objectives=['soft_vaamr', 'purer'])
        res = self._infer(model, g, cfg)
        self.assertIn('gnn_purer_pred', res)
        self.assertIn('gnn_purer_conf', res)

    def test_purer_keys_absent_without_purer_head(self):
        model, g, _, _, cfg = _make_model_and_graph(objectives=['soft_vaamr', 'progression'])
        res = self._infer(model, g, cfg)
        self.assertNotIn('gnn_purer_pred', res)

    def test_vaamr_pred_values_in_valid_range(self):
        model, g, _, _, cfg = _make_model_and_graph(objectives=['soft_vaamr'])
        res = self._infer(model, g, cfg)
        for pred in res['gnn_vaamr_pred']:
            self.assertIn(pred, range(5))

    def test_purer_pred_values_in_valid_range(self):
        model, g, _, _, cfg = _make_model_and_graph(objectives=['purer'])
        res = self._infer(model, g, cfg)
        for pred in res['gnn_purer_pred']:
            self.assertIn(pred, range(5))

    def test_conf_values_in_0_1(self):
        model, g, _, _, cfg = _make_model_and_graph(objectives=['soft_vaamr'])
        res = self._infer(model, g, cfg)
        for conf in res['gnn_vaamr_conf']:
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0 + 1e-4)

    def test_segment_id_and_node_type_always_present(self):
        model, g, _, _, cfg = _make_model_and_graph()
        res = self._infer(model, g, cfg)
        self.assertIn('segment_id', res)
        self.assertIn('node_type', res)
        self.assertEqual(len(res['segment_id']), len(g.node_ids))

    def test_length_matches_graph_nodes(self):
        model, g, _, _, cfg = _make_model_and_graph()
        res = self._infer(model, g, cfg)
        N = len(g.node_ids)
        self.assertEqual(len(res['gnn_vaamr_pred']), N)
        self.assertEqual(len(res['gnn_purer_pred']), N)


# ---------------------------------------------------------------------------
# build_cue_blocks_with_segments
# ---------------------------------------------------------------------------

class TestBuildCueBlocksWithSegments(unittest.TestCase):

    def setUp(self):
        from gnn_layer.inference import build_cue_blocks_with_segments
        self._fn = build_cue_blocks_with_segments

    def test_returns_list(self):
        df = synthetic_df(n_sessions=2)
        result = self._fn(df)
        self.assertIsInstance(result, list)

    def test_blocks_are_dicts(self):
        df = synthetic_df(n_sessions=2)
        result = self._fn(df)
        for b in result:
            self.assertIsInstance(b, dict)

    def test_expected_keys_in_each_block(self):
        df = synthetic_df(n_sessions=2)
        result = self._fn(df)
        expected_keys = {'session_id', 'from_seg_id', 'to_seg_id',
                         'from_stage', 'to_stage', 'transition_type',
                         'therapist_seg_ids'}
        for b in result:
            self.assertEqual(set(b.keys()), expected_keys,
                             f"Block has unexpected keys: {set(b.keys())}")

    def test_therapist_seg_ids_is_list(self):
        df = synthetic_df(n_sessions=2)
        result = self._fn(df)
        for b in result:
            self.assertIsInstance(b['therapist_seg_ids'], list)

    def test_returns_empty_when_required_columns_absent(self):
        df = synthetic_df(n_sessions=2).drop(columns=['start_time_ms'])
        result = self._fn(df)
        self.assertEqual(result, [])

    def test_each_block_produced_from_consecutive_participant_pairs(self):
        """Each session with k labeled participant segments yields k-1 blocks."""
        df = synthetic_df(n_sessions=1)
        result = self._fn(df)
        # synthetic_df n_sessions=1: 3 participant segments → 2 blocks
        session_blocks = [b for b in result if b['session_id'] == 'c1s1']
        part_segs = df[(df['session_id'] == 'c1s1') & (df['speaker'] == 'participant') &
                       (df['final_label'].notna())].shape[0]
        self.assertEqual(len(session_blocks), part_segs - 1)

    def test_transition_type_values_are_valid(self):
        df = synthetic_df(n_sessions=2)
        result = self._fn(df)
        valid = {'forward', 'backward', 'lateral'}
        for b in result:
            self.assertIn(b['transition_type'], valid)

    def test_empty_df_returns_empty_list(self):
        df = pd.DataFrame(columns=['session_id', 'speaker', 'final_label',
                                   'start_time_ms', 'end_time_ms', 'segment_id'])
        result = self._fn(df)
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# cue_block_embeddings
# ---------------------------------------------------------------------------

class TestCueBlockEmbeddings(unittest.TestCase):

    def setUp(self):
        from gnn_layer.inference import cue_block_embeddings
        self._fn = cue_block_embeddings

    def test_mean_pool_two_segments(self):
        blocks = [{'therapist_seg_ids': ['t1', 't2']}]
        emb = {
            't1': np.array([1.0, 0.0], dtype='float32'),
            't2': np.array([3.0, 0.0], dtype='float32'),
        }
        rows, mat = self._fn(blocks, emb)
        self.assertEqual(len(rows), 1)
        self.assertEqual(mat.shape, (1, 2))
        np.testing.assert_allclose(mat[0], [2.0, 0.0], atol=1e-6)

    def test_single_segment_gives_same_vector(self):
        blocks = [{'therapist_seg_ids': ['t1']}]
        emb = {'t1': np.array([0.5, 1.5], dtype='float32')}
        rows, mat = self._fn(blocks, emb)
        np.testing.assert_allclose(mat[0], [0.5, 1.5], atol=1e-6)

    def test_empty_block_excluded(self):
        blocks = [
            {'therapist_seg_ids': ['t1']},
            {'therapist_seg_ids': []},     # empty — excluded
            {'therapist_seg_ids': ['t2']},
        ]
        emb = {
            't1': np.array([1.0, 0.0], dtype='float32'),
            't2': np.array([2.0, 0.0], dtype='float32'),
        }
        rows, mat = self._fn(blocks, emb)
        self.assertEqual(len(rows), 2)
        self.assertEqual(mat.shape, (2, 2))

    def test_missing_embedding_key_excluded(self):
        blocks = [{'therapist_seg_ids': ['t1', 't_missing']}]
        emb = {'t1': np.array([1.0, 0.0], dtype='float32')}
        rows, mat = self._fn(blocks, emb)
        # t_missing not in emb → only t1 contributes → mean = t1
        np.testing.assert_allclose(mat[0], [1.0, 0.0], atol=1e-6)

    def test_all_missing_embeddings_block_excluded(self):
        blocks = [{'therapist_seg_ids': ['missing_1', 'missing_2']}]
        rows, mat = self._fn(blocks, {})
        self.assertEqual(rows, [])

    def test_empty_input_returns_empty(self):
        rows, mat = self._fn([], {})
        self.assertEqual(rows, [])
        self.assertEqual(mat.shape, (0, 0))

    def test_output_dtype_is_float32(self):
        blocks = [{'therapist_seg_ids': ['t1']}]
        emb = {'t1': np.array([1.0, 2.0], dtype='float64')}
        rows, mat = self._fn(blocks, emb)
        self.assertEqual(mat.dtype, np.float32)

    def test_matrix_shape_rows_x_dim(self):
        dim = 16
        blocks = [{'therapist_seg_ids': ['t1']},
                  {'therapist_seg_ids': ['t2']},
                  {'therapist_seg_ids': ['t3']}]
        rng = np.random.default_rng(0)
        emb = {f't{i+1}': rng.standard_normal(dim).astype('float32') for i in range(3)}
        rows, mat = self._fn(blocks, emb)
        self.assertEqual(mat.shape, (3, dim))

    def test_block_metadata_preserved_in_rows(self):
        blocks = [{'session_id': 'sess1', 'therapist_seg_ids': ['t1']}]
        emb = {'t1': np.array([0.0, 1.0], dtype='float32')}
        rows, mat = self._fn(blocks, emb)
        self.assertEqual(rows[0]['session_id'], 'sess1')


if __name__ == '__main__':
    unittest.main()
