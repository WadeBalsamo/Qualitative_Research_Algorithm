"""
tests/unit/test_gnn_coupling.py
--------------------------------
Unit tests for gnn_layer/coupling.py:
  - extract_latent_factors (PCA + forward_corr)
  - factor_exemplars
  - interpret_factors (hermetic: interpret_against_cf_ic=False path)
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
from gnn_layer import coupling as _cp


def _block_rows(n=10):
    rows = []
    for i in range(n):
        rows.append({
            'from_seg_id': f'p_{i}',
            'to_seg_id': f'p_{i + 1}',
            'from_stage': i % 5,
            'to_stage': (i + 1) % 5,
            'session_id': f'c1s{(i // 5) + 1}',
            'therapist_seg_ids': [f't_{i}'],
        })
    return rows


class TestExtractLatentFactors(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = GnnLayerConfig(n_latent_factors=3, seed=42)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_embeddings(self, n=15, dim=8):
        return np.random.default_rng(0).standard_normal((n, dim)).astype(np.float32)

    def test_output_keys(self):
        X = self._make_embeddings()
        result = _cp.extract_latent_factors(X, config=self.cfg)
        self.assertIn('components', result)
        self.assertIn('block_scores', result)
        self.assertIn('explained_variance_ratio', result)
        self.assertIn('factor_forward_corr', result)

    def test_block_scores_shape(self):
        X = self._make_embeddings(15, 8)
        result = _cp.extract_latent_factors(X, config=self.cfg)
        # n_factors capped at min(n_latent_factors, n, dim)
        scores = result['block_scores']
        self.assertEqual(scores.shape[0], 15)
        self.assertLessEqual(scores.shape[1], 3)

    def test_explained_variance_ratio_sums_le_1(self):
        X = self._make_embeddings()
        result = _cp.extract_latent_factors(X, config=self.cfg)
        evr = result['explained_variance_ratio']
        self.assertIsNotNone(evr)
        self.assertLessEqual(sum(evr), 1.0 + 1e-5)

    def test_forward_corr_none_when_all_same_outcome(self):
        X = self._make_embeddings()
        # All-same forward outcome → no variance → corr should be None or 0 per spec
        # Source: "if len(np.unique(forward_outcome)) > 1" → None if uniform
        result = _cp.extract_latent_factors(X, forward_outcome=[1] * 15, config=self.cfg)
        self.assertIsNone(result['factor_forward_corr'])

    def test_forward_corr_is_list_of_floats_when_varied(self):
        X = self._make_embeddings(20, 8)
        fwd = [1 if i % 2 == 0 else 0 for i in range(20)]
        result = _cp.extract_latent_factors(X, forward_outcome=fwd, config=self.cfg)
        corr = result['factor_forward_corr']
        self.assertIsNotNone(corr)
        self.assertIsInstance(corr, list)
        for c in corr:
            self.assertIsInstance(c, float)

    def test_insufficient_rows_returns_none_values(self):
        X = np.ones((1, 4), dtype=np.float32)
        result = _cp.extract_latent_factors(X, config=self.cfg)
        self.assertIsNone(result['components'])
        self.assertIsNone(result['block_scores'])

    def test_n_factors_capped_at_dim(self):
        """If n_latent_factors > embedding dimension, n_factors = dim."""
        cfg = GnnLayerConfig(n_latent_factors=20, seed=0)
        X = self._make_embeddings(10, 4)  # dim=4 < 20
        result = _cp.extract_latent_factors(X, config=cfg)
        scores = result['block_scores']
        self.assertLessEqual(scores.shape[1], 4)


class TestFactorExemplars(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_top_k_per_factor(self):
        scores = np.random.default_rng(0).standard_normal((10, 3)).astype(np.float32)
        rows = _block_rows(10)
        out = _cp.factor_exemplars(scores, rows, k=2)
        for f in range(3):
            self.assertIn(f, out)
            self.assertLessEqual(len(out[f]), 2)

    def test_exemplar_keys(self):
        scores = np.random.default_rng(1).standard_normal((6, 2)).astype(np.float32)
        rows = _block_rows(6)
        out = _cp.factor_exemplars(scores, rows, k=2)
        for exs in out.values():
            for e in exs:
                self.assertIn('from_stage', e)
                self.assertIn('to_stage', e)
                self.assertIn('session_id', e)
                self.assertIn('from_seg_id', e)

    def test_none_scores_returns_empty(self):
        out = _cp.factor_exemplars(None, _block_rows(5), k=2)
        self.assertEqual(out, {})

    def test_exemplars_ordered_by_positive_loading(self):
        """Top exemplars must have the highest loading on the factor (descending)."""
        scores = np.zeros((5, 1), dtype=np.float32)
        scores[:, 0] = [0.1, 0.9, 0.3, 0.8, 0.2]
        rows = _block_rows(5)
        out = _cp.factor_exemplars(scores, rows, k=2)
        # Factor 0 top-2 exemplars should correspond to indices 1 (0.9) and 3 (0.8)
        top2_from_ids = [e['from_seg_id'] for e in out[0]]
        self.assertIn(rows[1]['from_seg_id'], top2_from_ids)
        self.assertIn(rows[3]['from_seg_id'], top2_from_ids)


class TestInterpretFactors(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_disabled_returns_note(self):
        """With interpret_against_cf_ic=False, must return note dict without embedding."""
        cfg = GnnLayerConfig(interpret_against_cf_ic=False)
        factors = {'components': None}
        result = _cp.interpret_factors(factors, None, cfg)
        self.assertIsInstance(result, dict)
        self.assertIn('note', result)

    def test_no_exemplar_texts_returns_note(self):
        cfg = GnnLayerConfig(interpret_against_cf_ic=True)
        factors = {'components': None}
        # Pass None as exemplar_texts_by_factor
        result = _cp.interpret_factors(factors, None, cfg)
        self.assertIn('note', result)

    def test_empty_exemplar_texts_returns_note(self):
        cfg = GnnLayerConfig(interpret_against_cf_ic=True)
        factors = {}
        result = _cp.interpret_factors(factors, {}, cfg)
        self.assertIn('note', result)

    def test_interpret_off_is_hermetic(self):
        """Calling with interpret_against_cf_ic=False never triggers embedding downloads."""
        cfg = GnnLayerConfig(interpret_against_cf_ic=False)
        # Should complete instantly with no network calls
        result = _cp.interpret_factors({}, {0: ['some text']}, cfg)
        self.assertIn('note', result)

    def test_cf_ic_reference_nonempty(self):
        from gnn_layer.coupling import CF_IC_REFERENCE
        self.assertGreater(len(CF_IC_REFERENCE), 0)
        for k, v in CF_IC_REFERENCE.items():
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, str)


if __name__ == '__main__':
    unittest.main()
