"""
tests/test_gnn_layer.py
-----------------------
Unit tests for the GNN representation-and-discovery layer.

These tests inject synthetic embeddings (monkeypatching the Qwen3 encoder) so the
16 GB embedding model is never downloaded — mirroring how the suite avoids LLM/
embedding backends elsewhere.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gnn_layer.embeddings as emb
from gnn_layer.config import GnnLayerConfig
from gnn_layer import soft_labels, graph_builder, runner


def _synthetic_df(n_sessions=3, dim=16):
    rows = []
    for s in range(n_sessions):
        sess = f'c1s{s + 1}'
        t = 0
        for j in range(6):
            spk = 'participant' if j % 2 == 0 else 'therapist'
            t += 1000
            stage = (j // 2) % 5
            rv = [{'stage': stage, 'confidence': 0.8},
                  {'stage': (stage + 1) % 5, 'confidence': 0.4, 'secondary_stage': stage}]
            rows.append(dict(
                segment_id=f'{sess}_{j}', session_id=sess, speaker=spk,
                text=f'utterance {sess} {j}', start_time_ms=t, end_time_ms=t + 800,
                final_label=(stage if spk == 'participant' else np.nan),
                primary_stage=(stage if spk == 'participant' else np.nan),
                rater_votes=(rv if spk == 'participant' else None),
                codebook_labels_ensemble=(['affect_x', 'somatic_y'] if spk == 'participant' else []),
                purer_primary=((j // 2) % 5 if spk == 'therapist' else np.nan),
                microskill_labels_ensemble=(['open_ended_question', 'validation'] if spk == 'therapist' else []),
                in_human_coded_subset=False, human_label=np.nan,
            ))
    return pd.DataFrame(rows)


class TestSoftLabels(unittest.TestCase):
    def test_mixture_normalizes(self):
        m = soft_labels.ballots_to_mixture(
            [{'stage': 1, 'confidence': 0.9}, {'stage': 2, 'confidence': 0.1, 'secondary_stage': 1}])
        self.assertEqual(m.shape, (5,))
        self.assertAlmostEqual(float(m.sum()), 1.0, places=6)
        self.assertGreater(m[1], m[3])  # stage 1 carried weight

    def test_progression_is_expected_stage(self):
        m = soft_labels.one_hot(3)
        self.assertAlmostEqual(soft_labels.mixture_to_progression(m), 3.0, places=6)

    def test_empty_ballots_uniform(self):
        m = soft_labels.ballots_to_mixture(None)
        self.assertTrue(np.allclose(m, 0.2))


class TestGraphBuilder(unittest.TestCase):
    def setUp(self):
        self.df = _synthetic_df()
        self.cfg = GnnLayerConfig(enabled=True, knn_k=3, cache_embeddings=False)
        rng = np.random.default_rng(0)
        self.seg_emb = {sid: rng.standard_normal(16).astype('float32')
                        for sid in self.df['segment_id']}

    def test_node_and_edge_counts(self):
        g = graph_builder.build_graph(self.df, self.seg_emb, self.cfg)
        self.assertEqual(len(g.node_ids), 18)               # 3 sessions x 6 segs
        self.assertEqual(g.x.shape[0], 18)
        self.assertEqual(g.x.shape[1], 16)
        self.assertGreater(g.edge_index.shape[1], 0)        # temporal + knn edges
        self.assertEqual(g.edge_index.shape[0], 2)
        # node types split participant/therapist
        self.assertEqual(g.node_types.count('participant_segment'), 9)
        self.assertEqual(g.node_types.count('therapist_segment'), 9)

    def test_inductive_attach(self):
        g = graph_builder.build_graph(self.df, self.seg_emb, self.cfg)
        new = {'new_1': np.random.default_rng(1).standard_normal(16).astype('float32')}
        g2 = graph_builder.attach_new_segments(g, new, self.cfg)
        self.assertEqual(len(g2.node_ids), len(g.node_ids) + 1)
        self.assertIn('new_1', g2.index_of)


class TestModelForward(unittest.TestCase):
    def test_forward_shapes(self):
        import torch
        from gnn_layer.model import build_model
        df = _synthetic_df(n_sessions=1)
        rng = np.random.default_rng(0)
        seg_emb = {sid: rng.standard_normal(16).astype('float32') for sid in df['segment_id']}
        cfg = GnnLayerConfig(enabled=True, hidden_dim=8, n_layers=2, knn_k=2,
                             cache_embeddings=False, objectives=['soft_vaamr', 'progression'])
        g = graph_builder.build_graph(df, seg_emb, cfg)
        model = build_model(g, cfg)
        out = model(g.x, g.edge_index, g.edge_weight)
        self.assertEqual(out['emb'].shape, (g.x.shape[0], 8))
        self.assertEqual(out['soft_vaamr'].shape, (g.x.shape[0], 5))
        self.assertEqual(out['progression'].shape, (g.x.shape[0], 1))


class TestRunnerEndToEnd(unittest.TestCase):
    def setUp(self):
        self._orig = emb.embed_segment_texts
        rng = np.random.default_rng(7)
        emb.embed_segment_texts = lambda texts, config: rng.standard_normal((len(texts), 16)).astype('float32')

    def tearDown(self):
        emb.embed_segment_texts = self._orig

    def test_run_writes_artifacts(self):
        df = _synthetic_df()
        cfg = GnnLayerConfig(enabled=True, hidden_dim=16, n_layers=2, knn_k=3, epochs=20,
                             n_motif_clusters=3, cache_embeddings=False, seed=1,
                             interpret_against_cf_ic=False)  # keep test hermetic (no 16GB model)
        out_dir = tempfile.mkdtemp()
        res = runner.run_gnn_analysis(df, out_dir, config=cfg, verbose=False)
        self.assertEqual(res['status'], 'ok')
        names = {os.path.basename(f) for f in res['files_written']}
        self.assertIn('segment_positions.csv', names)
        self.assertIn('cue_motifs.csv', names)
        self.assertIn('gnn_vs_llm_lift.csv', names)
        for f in res['files_written']:
            self.assertTrue(os.path.isfile(f), f)
        # segment_positions mixtures are valid probability vectors
        sp = pd.read_csv(os.path.join(out_dir, '03_analysis_data', 'gnn', 'segment_positions.csv'))
        mix_cols = [c for c in sp.columns if c.startswith('vaamr_mix_')]
        self.assertEqual(len(mix_cols), 5)
        self.assertTrue(np.allclose(sp[mix_cols].sum(axis=1), 1.0, atol=1e-3))

    def test_head_predictions_and_triangulation(self):
        # With the purer head trained, head predictions + triangulation are produced.
        df = _synthetic_df()
        cfg = GnnLayerConfig(enabled=True, hidden_dim=16, n_layers=2, knn_k=3, epochs=20,
                             n_motif_clusters=3, cache_embeddings=False, seed=1,
                             interpret_against_cf_ic=False,  # keep test hermetic (no 16GB model)
                             objectives=['soft_vaamr', 'progression', 'purer'])
        out_dir = tempfile.mkdtemp()
        res = runner.run_gnn_analysis(df, out_dir, config=cfg, verbose=False)
        names = {os.path.basename(f) for f in res['files_written']}
        self.assertIn('gnn_head_predictions.csv', names)
        self.assertIn('report_gnn_triangulation.txt', names)
        hp = pd.read_csv(os.path.join(out_dir, '03_analysis_data', 'gnn', 'gnn_head_predictions.csv'))
        self.assertIn('gnn_vaamr_pred', hp.columns)
        self.assertIn('gnn_purer_pred', hp.columns)

    def test_triangulation_agreement_keys(self):
        from gnn_layer import triangulation as tri
        head_preds = {
            'segment_id': ['a', 'b', 'c', 'd'],
            'node_type': ['participant_segment'] * 4,
            'gnn_vaamr_pred': [0, 1, 2, 3],
        }
        dfa = pd.DataFrame({
            'segment_id': ['a', 'b', 'c', 'd'],
            'node_type': ['participant_segment'] * 4,
            'final_label': [0, 1, 2, 2],
        })
        out = tri.compute_triangulation(head_preds, dfa)
        self.assertIn('vaamr_gnn_vs_llm', out)
        self.assertEqual(out['vaamr_gnn_vs_llm']['n'], 4)
        self.assertAlmostEqual(out['vaamr_gnn_vs_llm']['percent_agreement'], 0.75, places=3)

    def test_disabled_default(self):
        self.assertFalse(GnnLayerConfig().enabled)


if __name__ == '__main__':
    unittest.main()
