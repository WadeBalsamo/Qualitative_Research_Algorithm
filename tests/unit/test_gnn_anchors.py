"""
tests/unit/test_gnn_anchors.py
------------------------------
Unit tests for gnn_layer/anchors.py (Path B / G2) and the human-axis
anchor-contribution ablation in gnn_layer/ablation.py.

Hermetic: framework definitions load from the markdown registry (no network);
the anchor embedder is monkeypatched so no 16GB model is downloaded.
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

from tests.testhelpers import synthetic_df, embedding_patch
from tests.testhelpers.marks import slow_test
from gnn_layer.config import GnnLayerConfig
from gnn_layer.classifier import anchors as anc


def _patch_anchor_embedder(dim=16, seed=11):
    """Deterministic anchor embedder; returns a restore callable."""
    import gnn_layer.embeddings as emb
    orig = emb.embed_anchor_texts
    rng = np.random.default_rng(seed)

    def _fake(texts, config):
        return rng.standard_normal((len(texts), dim)).astype('float32')

    emb.embed_anchor_texts = _fake
    return orig


class TestAnchorSpecs(unittest.TestCase):
    def test_vaamr_and_purer_by_default(self):
        specs = anc.anchor_specs(GnnLayerConfig())
        ids = [a for a, _ in specs]
        self.assertEqual(sum(1 for i in ids if i.startswith('anchor:vaamr:')), 5)
        self.assertEqual(sum(1 for i in ids if i.startswith('anchor:purer:')), 5)
        self.assertFalse(any(i.startswith('anchor:vce:') for i in ids))
        # definition text is non-empty
        self.assertTrue(all(txt.strip() for _, txt in specs))

    def test_vce_included_when_flagged(self):
        c = GnnLayerConfig(include_vce_nodes=True)
        ids = [a for a, _ in anc.anchor_specs(c)]
        self.assertGreater(sum(1 for i in ids if i.startswith('anchor:vce:')), 0)

    def test_families_can_be_disabled(self):
        c = GnnLayerConfig(include_vaamr_nodes=False, include_purer_nodes=True)
        ids = [a for a, _ in anc.anchor_specs(c)]
        self.assertFalse(any(i.startswith('anchor:vaamr:') for i in ids))
        self.assertTrue(any(i.startswith('anchor:purer:') for i in ids))


class TestAnchorSimilarityEdges(unittest.TestCase):
    def test_label_free_topm_edges(self):
        rng = np.random.default_rng(3)
        seg_emb = {f's{i}': rng.standard_normal(8).astype('float32') for i in range(20)}
        feats = {'anchor:vaamr:0': rng.standard_normal(8).astype('float32'),
                 'anchor:purer:1': rng.standard_normal(8).astype('float32')}
        edges = anc.anchor_similarity_edges(feats, seg_emb, top_m=3)
        # 2 anchors * 3 segments each
        self.assertEqual(len(edges), 6)
        for aid, sid, w in edges:
            self.assertIn(aid, feats)
            self.assertIn(sid, seg_emb)
            self.assertGreaterEqual(w, 0.0)

    def test_topm_capped_at_n_segments(self):
        rng = np.random.default_rng(4)
        seg_emb = {f's{i}': rng.standard_normal(8).astype('float32') for i in range(2)}
        feats = {'anchor:vaamr:0': rng.standard_normal(8).astype('float32')}
        edges = anc.anchor_similarity_edges(feats, seg_emb, top_m=10)
        self.assertEqual(len(edges), 2)

    def test_empty_inputs(self):
        self.assertEqual(anc.anchor_similarity_edges({}, {'s': [1, 2]}, 3), [])
        self.assertEqual(anc.anchor_similarity_edges({'a': [1, 2]}, {}, 3), [])


class TestCrossFrameworkAnchorEdges(unittest.TestCase):
    def test_none_when_vce_off(self):
        df = pd.DataFrame({'segment_id': ['a'], 'final_label': [0]})
        self.assertEqual(anc.cross_framework_anchor_edges(df, GnnLayerConfig(include_vce_nodes=False)), [])

    def test_maps_lift_to_anchor_ids_when_vce_on(self):
        # final_label (VAAMR) + codebook_labels_ensemble (VCE codes) drive the lift.
        df = pd.DataFrame({
            'segment_id': [f's{i}' for i in range(8)],
            'speaker': ['participant'] * 8,
            'final_label': [0, 0, 0, 0, 1, 1, 1, 1],
            'codebook_labels_ensemble': [['x'], ['x'], ['x'], ['x'], ['y'], ['y'], ['y'], []],
        })
        c = GnnLayerConfig(include_vce_nodes=True, cross_framework_min_lift=1.0)
        edges = anc.cross_framework_anchor_edges(df, c)
        # may be empty if no pair clears the threshold; if present, ids are well-formed
        for src, dst, w in edges:
            self.assertTrue(src.startswith('anchor:vaamr:'))
            self.assertTrue(dst.startswith('anchor:vce:'))
            self.assertIsInstance(w, float)


class TestBuildAnchors(unittest.TestCase):
    def setUp(self):
        self._restore = _patch_anchor_embedder(dim=16)

    def tearDown(self):
        import gnn_layer.embeddings as emb
        emb.embed_anchor_texts = self._restore

    def test_build_features_and_edges(self):
        rng = np.random.default_rng(5)
        seg_emb = {f's{i}': rng.standard_normal(16).astype('float32') for i in range(12)}
        df = pd.DataFrame({'segment_id': list(seg_emb), 'speaker': ['participant'] * 12})
        feats, edges = anc.build_anchors(df, seg_emb, GnnLayerConfig(anchor_knn_m=4))
        self.assertEqual(len(feats), 10)  # 5 VAAMR + 5 PURER
        # 10 anchors * 4 similarity edges each (no VCE cross edges by default)
        self.assertEqual(len(edges), 40)

    def test_empty_when_no_families(self):
        c = GnnLayerConfig(include_vaamr_nodes=False, include_purer_nodes=False,
                           include_vce_nodes=False)
        feats, edges = anc.build_anchors(pd.DataFrame({'segment_id': []}), {}, c)
        self.assertEqual(feats, {})
        self.assertEqual(edges, [])


class TestAnchorContributionVerdict(unittest.TestCase):
    """The decisive axis is human; with no human subset the verdict is inconclusive."""

    @slow_test
    def test_inconclusive_without_human_subset(self):
        self._restore = _patch_anchor_embedder(dim=16)
        try:
            df = synthetic_df()  # no in_human_coded_subset column
            cfg = GnnLayerConfig(enabled=True, gnn_classifier_enabled=True, hidden_dim=16, n_layers=2, knn_k=3,
                                 epochs=15, cache_embeddings=False, seed=1,
                                 validation_folds=2, anchor_knn_m=3)
            from gnn_layer.classifier import ablation as abl
            with embedding_patch(dim=16):
                from gnn_layer import embeddings as emb
                seg_emb = emb.load_or_build_segment_embeddings(df, cfg, cache_path=None)
                res = abl.anchor_contribution(df, seg_emb, cfg)
            self.assertEqual(res['verdict'], 'inconclusive')
            self.assertFalse(res['recommend_anchors'])
            self.assertIn('n_anchors', res)
        finally:
            import gnn_layer.embeddings as emb
            emb.embed_anchor_texts = self._restore

    def test_report_writes_and_frames_human_axis_decisive(self):
        tmp = tempfile.mkdtemp()
        try:
            from gnn_layer.classifier import ablation as abl
            res = {
                'n_anchors': 10, 'n_anchor_edges': 40, 'human_n': 0, 'anchor_min_human': 10,
                'human_kappa_without_anchors': None, 'human_kappa_with_anchors': None,
                'delta_kappa_human': None,
                'llm_kappa_without_anchors': 0.61, 'llm_kappa_with_anchors': 0.78,
                'delta_kappa_llm': 0.17, 'verdict': 'inconclusive', 'recommend_anchors': False,
            }
            path = abl.write_anchor_contribution_report(res, tmp)
            text = open(path, encoding='utf-8').read()
            self.assertTrue(path.endswith('anchor_contribution.txt'))
            self.assertIn('DECISIVE', text)
            self.assertIn('GNN vs HUMAN', text)
            self.assertIn('INFLATED', text)
            self.assertIn('RECOMMEND ANCHORS ON: NO', text)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestRunnerAnchorWiring(unittest.TestCase):
    """run_gnn_analysis with anchors ON in the main graph + the ablation report."""

    @slow_test
    def test_anchors_in_main_graph_and_ablation_report(self):
        self._restore = _patch_anchor_embedder(dim=16)
        try:
            from gnn_layer import runner
            from process import output_paths as _paths
            df = synthetic_df()
            cfg = GnnLayerConfig(enabled=True, gnn_classifier_enabled=True, hidden_dim=16, n_layers=2, knn_k=3, epochs=15,
                                 n_motif_clusters=3, cache_embeddings=False, seed=1,
                                 interpret_against_cf_ic=False, validation_folds=2,
                                 use_anchor_nodes=True, run_anchor_ablation=True, anchor_knn_m=3)
            out_dir = tempfile.mkdtemp()
            with embedding_patch(dim=16):
                res = runner.run_gnn_analysis(df, out_dir, config=cfg, verbose=False)
            self.assertEqual(res['status'], 'ok')
            names = {os.path.basename(f) for f in res['files_written']}
            self.assertIn('anchor_contribution.txt', names)
            # The persisted main graph must actually contain anchor nodes.
            from gnn_layer.classifier import graph_builder as gb
            g = gb.load_graph(_paths.gnn_model_dir(out_dir))
            self.assertGreater(g.meta.get('n_anchors', 0), 0)
            self.assertTrue(any(t == 'anchor' for t in g.node_types))
            shutil.rmtree(out_dir, ignore_errors=True)
        finally:
            import gnn_layer.embeddings as emb
            emb.embed_anchor_texts = self._restore


if __name__ == '__main__':
    unittest.main()
