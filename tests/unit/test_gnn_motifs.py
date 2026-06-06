"""
tests/unit/test_gnn_motifs.py
------------------------------
Unit tests for gnn_layer/motifs.py:
  - cluster_cue_motifs
  - score_motif_influence
  - annotate_label_purity
  - flag_emergent_motifs
  - select_motif_exemplars
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
from gnn_layer import motifs as _mot


def _block_rows(n=10, n_stages=5):
    """Minimal cue-block row list for testing."""
    rows = []
    for i in range(n):
        fs = i % n_stages
        ts = (fs + 1) % n_stages
        rows.append({
            'from_seg_id': f'p_{i}',
            'to_seg_id': f'p_{i + 1}',
            'from_stage': fs,
            'to_stage': ts,
            'session_id': f'c1s{(i // 5) + 1}',
            'therapist_seg_ids': [f't_{i}'],
        })
    return rows


class TestClusterCueMotifs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = GnnLayerConfig(n_motif_clusters=3, seed=42)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((12, 8)).astype(np.float32)
        ids = _mot.cluster_cue_motifs(X, self.cfg)
        self.assertEqual(ids.shape, (12,))

    def test_ids_are_integers(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((9, 8)).astype(np.float32)
        ids = _mot.cluster_cue_motifs(X, self.cfg)
        self.assertTrue(np.issubdtype(ids.dtype, np.integer))

    def test_n_unique_at_most_k(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((15, 8)).astype(np.float32)
        ids = _mot.cluster_cue_motifs(X, self.cfg)
        self.assertLessEqual(len(np.unique(ids)), self.cfg.n_motif_clusters)

    def test_empty_input(self):
        X = np.zeros((0, 8), dtype=np.float32)
        ids = _mot.cluster_cue_motifs(X, self.cfg)
        self.assertEqual(len(ids), 0)

    def test_single_row_returns_zero(self):
        X = np.ones((1, 8), dtype=np.float32)
        ids = _mot.cluster_cue_motifs(X, self.cfg)
        self.assertEqual(ids[0], 0)

    def test_n_clusters_capped_at_n(self):
        """If n < k, KMeans should get n clusters not k."""
        cfg = GnnLayerConfig(n_motif_clusters=10, seed=0)
        X = np.random.default_rng(5).standard_normal((3, 8)).astype(np.float32)
        ids = _mot.cluster_cue_motifs(X, cfg)
        self.assertLessEqual(len(np.unique(ids)), 3)


class TestScoreMotifInfluence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = GnnLayerConfig(seed=0)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_inputs(self, n=12, dim=8):
        rng = np.random.default_rng(99)
        X = rng.standard_normal((n, dim)).astype(np.float32)
        from_stages = [i % 5 for i in range(n)]
        forward = [1 if i % 2 == 0 else 0 for i in range(n)]
        motif_ids = np.array([i % 3 for i in range(n)])
        return X, from_stages, forward, motif_ids

    def test_returns_dict_keyed_by_motif(self):
        X, fs, fwd, ids = self._make_inputs()
        out = _mot.score_motif_influence(X, fs, fwd, ids, self.cfg)
        self.assertIsInstance(out, dict)
        for m in np.unique(ids):
            self.assertIn(int(m), out)

    def test_each_entry_has_required_keys(self):
        X, fs, fwd, ids = self._make_inputs()
        out = _mot.score_motif_influence(X, fs, fwd, ids, self.cfg)
        for m, d in out.items():
            self.assertIn('influence', d)
            self.assertIn('mean_pred_forward', d)
            self.assertIn('n_blocks', d)

    def test_n_blocks_sums_to_total(self):
        X, fs, fwd, ids = self._make_inputs(12)
        out = _mot.score_motif_influence(X, fs, fwd, ids, self.cfg)
        total = sum(d['n_blocks'] for d in out.values())
        self.assertEqual(total, 12)

    def test_empty_returns_empty(self):
        X = np.zeros((0, 8), dtype=np.float32)
        out = _mot.score_motif_influence(X, [], [], np.array([]), self.cfg)
        self.assertEqual(out, {})

    def test_influence_is_non_negative(self):
        X, fs, fwd, ids = self._make_inputs()
        out = _mot.score_motif_influence(X, fs, fwd, ids, self.cfg)
        for d in out.values():
            self.assertGreaterEqual(d['influence'], 0.0)

    def test_all_forward_gives_elevated_influence(self):
        """All-forward outcome → mean_pred_forward should be > 0."""
        X, fs, _, ids = self._make_inputs()
        fwd = [1] * len(fs)
        out = _mot.score_motif_influence(X, fs, fwd, ids, self.cfg)
        for d in out.values():
            self.assertGreater(d['mean_pred_forward'], 0.0)


class TestAnnotateLabelPurity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dominant_purer_is_mode(self):
        motif_ids = np.array([0, 0, 0, 0, 1, 1])
        purer_labels = [2, 2, 2, 1, 3, 3]
        out = _mot.annotate_label_purity(motif_ids, purer_labels)
        self.assertEqual(out[0]['dominant_purer'], 2)
        self.assertEqual(out[1]['dominant_purer'], 3)

    def test_purity_is_fraction(self):
        motif_ids = np.array([0, 0, 0, 0])
        purer_labels = [1, 1, 2, 2]  # 50% for each
        out = _mot.annotate_label_purity(motif_ids, purer_labels)
        self.assertAlmostEqual(out[0]['purer_purity'], 0.5, places=3)

    def test_pure_motif_has_purity_1(self):
        motif_ids = np.array([0, 0, 0])
        purer_labels = [4, 4, 4]
        out = _mot.annotate_label_purity(motif_ids, purer_labels)
        self.assertAlmostEqual(out[0]['purer_purity'], 1.0, places=5)

    def test_none_labels_excluded_gracefully(self):
        motif_ids = np.array([0, 0, 0])
        purer_labels = [None, 1, 1]
        out = _mot.annotate_label_purity(motif_ids, purer_labels)
        # Two non-None labels, both 1 → purity 1.0
        self.assertAlmostEqual(out[0]['purer_purity'], 1.0, places=5)

    def test_all_none_gives_zero_purity(self):
        motif_ids = np.array([0, 0])
        purer_labels = [None, None]
        out = _mot.annotate_label_purity(motif_ids, purer_labels)
        self.assertEqual(out[0]['purer_purity'], 0.0)
        self.assertIsNone(out[0]['dominant_purer'])


class TestFlagEmergentMotifs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = GnnLayerConfig(min_motif_influence=1.2, motif_min_block_count=3)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_flags_high_influence_low_purity(self):
        stats = {
            0: {'n_blocks': 5, 'influence': 1.5, 'mean_pred_forward': 0.7},
            1: {'n_blocks': 5, 'influence': 0.8, 'mean_pred_forward': 0.3},
        }
        purity = {
            0: {'dominant_purer': 1, 'purer_purity': 0.3},
            1: {'dominant_purer': 2, 'purer_purity': 0.9},
        }
        flagged = _mot.flag_emergent_motifs(stats, purity, self.cfg)
        self.assertIn(0, flagged)
        self.assertNotIn(1, flagged)

    def test_below_min_blocks_excluded(self):
        stats = {0: {'n_blocks': 1, 'influence': 2.0, 'mean_pred_forward': 0.9}}
        purity = {0: {'dominant_purer': 1, 'purer_purity': 0.1}}
        flagged = _mot.flag_emergent_motifs(stats, purity, self.cfg)
        self.assertEqual(flagged, [])

    def test_below_influence_threshold_excluded(self):
        stats = {0: {'n_blocks': 10, 'influence': 1.0, 'mean_pred_forward': 0.5}}
        purity = {0: {'dominant_purer': 1, 'purer_purity': 0.1}}
        flagged = _mot.flag_emergent_motifs(stats, purity, self.cfg)
        self.assertEqual(flagged, [])

    def test_high_purity_excluded(self):
        stats = {0: {'n_blocks': 10, 'influence': 2.0, 'mean_pred_forward': 0.9}}
        purity = {0: {'dominant_purer': 1, 'purer_purity': 0.8}}
        flagged = _mot.flag_emergent_motifs(stats, purity, self.cfg)
        self.assertEqual(flagged, [])

    def test_returns_sorted_list(self):
        stats = {
            3: {'n_blocks': 5, 'influence': 1.5, 'mean_pred_forward': 0.7},
            1: {'n_blocks': 5, 'influence': 1.5, 'mean_pred_forward': 0.7},
        }
        purity = {
            3: {'dominant_purer': 1, 'purer_purity': 0.2},
            1: {'dominant_purer': 2, 'purer_purity': 0.2},
        }
        flagged = _mot.flag_emergent_motifs(stats, purity, self.cfg)
        self.assertEqual(flagged, sorted(flagged))


class TestSelectMotifExemplars(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_at_most_k_per_motif(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 8)).astype(np.float32)
        motif_ids = np.array([0] * 5 + [1] * 5)
        rows = _block_rows(10)
        out = _mot.select_motif_exemplars(motif_ids, X, rows, k=3)
        for exs in out.values():
            self.assertLessEqual(len(exs), 3)

    def test_exemplar_dicts_have_required_keys(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((6, 8)).astype(np.float32)
        motif_ids = np.array([0, 0, 0, 1, 1, 1])
        rows = _block_rows(6)
        out = _mot.select_motif_exemplars(motif_ids, X, rows, k=2)
        for exs in out.values():
            for e in exs:
                self.assertIn('from_seg_id', e)
                self.assertIn('to_seg_id', e)
                self.assertIn('from_stage', e)
                self.assertIn('to_stage', e)
                self.assertIn('session_id', e)

    def test_all_motifs_covered(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((9, 4)).astype(np.float32)
        motif_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        rows = _block_rows(9)
        out = _mot.select_motif_exemplars(motif_ids, X, rows, k=2)
        self.assertIn(0, out)
        self.assertIn(1, out)
        self.assertIn(2, out)


if __name__ == '__main__':
    unittest.main()
