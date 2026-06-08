"""
tests/unit/test_gnn_anchors_arm.py
----------------------------------
Hermetic tests for the VAAMR concept-anchor GNN arm
(experiments/gnn_reliability/anchors_arm.py — design_decisions.md §6, arm B1).

Covers:
  * build_vaamr_anchors — exactly the 5 VAAMR construct anchors (no PURER/VCE),
    embedded into the segment space; the anchored graph gains those nodes/edges.
  * run_anchored_gnn_arm — the grouped-CV out-of-fold engine returns predictions for
    every labeled participant segment (in range), never for anchors or therapists,
    for BOTH the 5-class and 6-class ("No code") configs.

No network, no model downloads: the anchor embedder is monkeypatched (so the VAAMR
definitions are "embedded" by a deterministic RNG into the same dim as the synthetic
segment vectors), framework definitions load from the markdown registry, and segment
embeddings are precomputed random vectors.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from experiments.gnn_reliability import anchors_arm
from experiments.gnn_reliability import harness
from gnn_layer.config import GnnLayerConfig
from gnn_layer.classifier import graph_builder as gb


_DIM = 16


def _patch_anchor_embedder(dim=_DIM, seed=11):
    """Deterministic anchor embedder (NO model download); returns a restore callable."""
    import gnn_layer.embeddings as emb
    orig = emb.embed_anchor_texts
    rng = np.random.default_rng(seed)

    def _fake(texts, config):
        return rng.standard_normal((len(texts), dim)).astype('float32')

    emb.embed_anchor_texts = _fake
    return orig


def _synthetic(dim=_DIM, seed=0):
    """Interleaved participant/therapist frame + random embeddings.

    6 participants; each contributes 3 labeled participant turns (VAAMR 0..4 spread
    across participants), 1 "No code" participant turn (null final_label), and a
    therapist turn after each (purer_primary set). Returns (df, {segment_id: vec}).
    """
    rng = np.random.default_rng(seed)
    rows, emb = [], {}
    for p in range(6):
        sess = f'sess{p}'
        t = 0
        for j in range(4):
            t += 1000
            sid = f'p{p}_u{j}'
            no_code = (j == 3)  # the 4th participant turn is "No code"
            rows.append({
                'segment_id': sid, 'participant_id': f'P{p}', 'session_id': sess,
                'speaker': 'participant', 'text': f'participant text {p} {j}',
                'start_time_ms': t,
                'final_label': (np.nan if no_code else (p + j) % 5),
                'rater_votes': None, 'purer_primary': np.nan,
            })
            emb[sid] = rng.standard_normal(dim).astype('float32')

            t += 500
            tsid = f'p{p}_t{j}'
            rows.append({
                'segment_id': tsid, 'participant_id': f'P{p}', 'session_id': sess,
                'speaker': 'therapist', 'text': f'therapist text {p} {j}',
                'start_time_ms': t,
                'final_label': np.nan, 'rater_votes': None,
                'purer_primary': (p + j) % 5,
            })
            emb[tsid] = rng.standard_normal(dim).astype('float32')
    return pd.DataFrame(rows), emb


def _labeled_participant_ids(df):
    m = (df['speaker'] == 'participant') & df['final_label'].notna()
    return [str(s) for s in df.loc[m, 'segment_id']]


def _therapist_ids(df):
    return [str(s) for s in df.loc[df['speaker'] == 'therapist', 'segment_id']]


class TestBuildVaamrAnchors(unittest.TestCase):
    """The arm adds exactly the 5 VAAMR construct anchors to the graph."""

    def test_anchored_graph_has_more_nodes_and_anchor_types(self):
        df, emb = _synthetic()
        cfg = GnnLayerConfig(knn_k=3, anchor_knn_m=3)
        restore = _patch_anchor_embedder()
        try:
            plain = gb.build_graph(df, emb, cfg)
            feats, edges = anchors_arm.build_vaamr_anchors(df, emb, cfg)
            anchored = gb.build_graph(df, emb, cfg,
                                      anchor_features=feats, anchor_edges=edges)
        finally:
            import gnn_layer.embeddings as e
            e.embed_anchor_texts = restore

        # exactly the 5 VAAMR construct anchors — no PURER, no VCE, no cross edges
        self.assertEqual(len(feats), 5)
        self.assertTrue(all(a.startswith('anchor:vaamr:') for a in feats))
        self.assertTrue(all(src.startswith('anchor:vaamr:') for src, _dst, _w in edges))
        self.assertEqual(len(edges), 5 * 3)  # 5 anchors * anchor_knn_m similarity edges

        # the anchored graph gains the 5 anchor nodes (and 'anchor' node types)
        self.assertEqual(len(anchored.node_ids) - len(plain.node_ids), 5)
        self.assertGreater(len(anchored.node_ids), len(plain.node_ids))
        self.assertIn('anchor', anchored.node_types)
        self.assertEqual(anchored.meta.get('n_anchors'), 5)
        self.assertEqual(plain.meta.get('n_anchors'), 0)

    def test_anchors_embedded_in_segment_space(self):
        """Anchor features share the segment embedding dimensionality (same space)."""
        df, emb = _synthetic()
        restore = _patch_anchor_embedder()
        try:
            feats, _edges = anchors_arm.build_vaamr_anchors(
                df, emb, GnnLayerConfig(anchor_knn_m=4))
        finally:
            import gnn_layer.embeddings as e
            e.embed_anchor_texts = restore
        seg_dim = len(next(iter(emb.values())))
        for vec in feats.values():
            self.assertEqual(len(np.asarray(vec)), seg_dim)


class TestRunAnchoredGnnArm(unittest.TestCase):
    """Grouped-CV OOF contract: identical to run_gnn_arm, anchors never scored."""

    def _run(self, n_classes, framework=None):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest('torch unavailable')
        df, emb = _synthetic()
        folds = harness.build_folds(df, n_folds=3, seed=42, verbose=False)
        cfg = GnnLayerConfig(knn_k=3, hidden_dim=8, n_layers=1, epochs=5, patience=5,
                             seed=42, anchor_knn_m=3, vaamr_n_classes=n_classes)
        restore = _patch_anchor_embedder()
        try:
            oof = anchors_arm.run_anchored_gnn_arm(df, emb, folds, cfg,
                                                   framework=framework)
        finally:
            import gnn_layer.embeddings as e
            e.embed_anchor_texts = restore
        return df, oof

    def _assert_contract(self, df, oof, n_classes):
        # every labeled participant segment got an out-of-fold prediction, in range
        for sid in _labeled_participant_ids(df):
            self.assertIn(sid, oof)
        self.assertTrue(all(0 <= int(v) < n_classes for v in oof.values()))
        # anchors are never predicted/scored, and VAAMR never crosses to therapists
        self.assertFalse(any(str(k).startswith('anchor:') for k in oof))
        for tsid in _therapist_ids(df):
            self.assertNotIn(tsid, oof)

    def test_runs_5_class(self):
        df, oof = self._run(5, framework=None)
        self._assert_contract(df, oof, 5)

    def test_runs_6_class_with_nocode_and_framework(self):
        # pass the real VAAMR framework object (forwarded to build_graph for parity)
        from constructs.registry import load as _load
        df, oof = self._run(6, framework=_load('vaamr'))
        self._assert_contract(df, oof, 6)
        # 6-class arm can emit the No-code class (5) for the unlabeled participant turns
        no_code_ids = [str(s) for s in
                       df.loc[(df['speaker'] == 'participant') & df['final_label'].isna(),
                              'segment_id']]
        self.assertTrue(all(sid in oof for sid in no_code_ids))


if __name__ == '__main__':
    unittest.main()
