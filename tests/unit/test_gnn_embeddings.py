"""
tests/unit/test_gnn_embeddings.py
-----------------------------------
Unit tests for gnn_layer/embeddings.py:
  - load_or_build_segment_embeddings:
      * cache hit (build once, second call loads from .npz cache)
      * stale-by-text-hash invalidation (changed text triggers re-encode)
      * cache_embeddings=False skips reading/writing cache
All hermetic: monkeypatches embed_segment_texts so no model loads.
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
import gnn_layer.embeddings as _emb


def _simple_df(n=6):
    rows = [{'segment_id': f's{i}', 'text': f'utterance {i}'} for i in range(n)]
    return pd.DataFrame(rows)


class TestLoadOrBuildSegmentEmbeddingsCacheHit(unittest.TestCase):
    """Build once → cache written; second call loads from cache (no re-encode)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = GnnLayerConfig(cache_embeddings=True)
        self.cache_path = os.path.join(self.tmp, 'embeddings.npz')
        self.call_count = 0

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _patched_embed(self, n, dim=16):
        """Return a new deterministic patch that counts calls."""
        rng = np.random.default_rng(0)
        call_tracker = {'n': 0}

        def _fake(texts, config):
            call_tracker['n'] += 1
            return rng.standard_normal((len(texts), dim)).astype('float32')

        return _fake, call_tracker

    def test_cache_file_created(self):
        df = _simple_df(4)
        orig = _emb.embed_segment_texts
        rng = np.random.default_rng(1)
        _emb.embed_segment_texts = lambda t, c: rng.standard_normal((len(t), 16)).astype('float32')
        try:
            _emb.load_or_build_segment_embeddings(df, self.cfg, cache_path=self.cache_path)
        finally:
            _emb.embed_segment_texts = orig
        self.assertTrue(os.path.isfile(self.cache_path))

    def test_second_call_does_not_re_encode(self):
        """After cache is built, a second call with same df should not call embed again."""
        df = _simple_df(4)
        orig = _emb.embed_segment_texts
        call_counter = {'n': 0}
        rng = np.random.default_rng(2)

        def _fake(texts, config):
            call_counter['n'] += 1
            return rng.standard_normal((len(texts), 16)).astype('float32')

        _emb.embed_segment_texts = _fake
        try:
            # First call — builds cache
            _emb.load_or_build_segment_embeddings(df, self.cfg, cache_path=self.cache_path)
            n_after_first = call_counter['n']
            # Second call — should hit cache entirely
            _emb.load_or_build_segment_embeddings(df, self.cfg, cache_path=self.cache_path)
            n_after_second = call_counter['n']
        finally:
            _emb.embed_segment_texts = orig

        # First call must have encoded something
        self.assertGreater(n_after_first, 0)
        # Second call must not have encoded anything new
        self.assertEqual(n_after_second, n_after_first)

    def test_returns_dict_with_segment_ids(self):
        df = _simple_df(5)
        orig = _emb.embed_segment_texts
        rng = np.random.default_rng(3)
        _emb.embed_segment_texts = lambda t, c: rng.standard_normal((len(t), 16)).astype('float32')
        try:
            result = _emb.load_or_build_segment_embeddings(df, self.cfg, cache_path=self.cache_path)
        finally:
            _emb.embed_segment_texts = orig
        for sid in df['segment_id']:
            self.assertIn(str(sid), result)

    def test_embedding_shape(self):
        df = _simple_df(4)
        orig = _emb.embed_segment_texts
        dim = 8
        rng = np.random.default_rng(4)
        _emb.embed_segment_texts = lambda t, c: rng.standard_normal((len(t), dim)).astype('float32')
        try:
            result = _emb.load_or_build_segment_embeddings(df, self.cfg, cache_path=self.cache_path)
        finally:
            _emb.embed_segment_texts = orig
        for vec in result.values():
            self.assertEqual(vec.shape, (dim,))


class TestLoadOrBuildStaleByTextHash(unittest.TestCase):
    """Changed text for an existing segment_id forces re-encode (stale hash)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = GnnLayerConfig(cache_embeddings=True)
        self.cache_path = os.path.join(self.tmp, 'embeddings.npz')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_text_change_triggers_reencode(self):
        df_v1 = pd.DataFrame([
            {'segment_id': 's0', 'text': 'original text'},
            {'segment_id': 's1', 'text': 'another text'},
        ])
        df_v2 = pd.DataFrame([
            {'segment_id': 's0', 'text': 'CHANGED text'},  # hash differs
            {'segment_id': 's1', 'text': 'another text'},
        ])

        orig = _emb.embed_segment_texts
        call_counter = {'n': 0}
        rng = np.random.default_rng(5)

        def _fake(texts, config):
            call_counter['n'] += len(texts)
            return rng.standard_normal((len(texts), 16)).astype('float32')

        _emb.embed_segment_texts = _fake
        try:
            # Build cache for v1 (encodes 2 segments)
            _emb.load_or_build_segment_embeddings(df_v1, self.cfg, cache_path=self.cache_path)
            n1 = call_counter['n']
            # v2: s0 text changed → should re-encode at least 1 segment
            _emb.load_or_build_segment_embeddings(df_v2, self.cfg, cache_path=self.cache_path)
            n2 = call_counter['n']
        finally:
            _emb.embed_segment_texts = orig

        self.assertEqual(n1, 2)
        # At least one re-encode
        self.assertGreater(n2, n1)

    def test_unchanged_text_no_reencode(self):
        df = _simple_df(3)
        orig = _emb.embed_segment_texts
        call_counter = {'n': 0}
        rng = np.random.default_rng(6)

        def _fake(texts, config):
            call_counter['n'] += 1
            return rng.standard_normal((len(texts), 16)).astype('float32')

        _emb.embed_segment_texts = _fake
        try:
            _emb.load_or_build_segment_embeddings(df, self.cfg, cache_path=self.cache_path)
            n1 = call_counter['n']
            _emb.load_or_build_segment_embeddings(df, self.cfg, cache_path=self.cache_path)
            n2 = call_counter['n']
        finally:
            _emb.embed_segment_texts = orig

        self.assertEqual(n1, n2, "Unchanged texts must not trigger re-encoding")


class TestLoadOrBuildCacheDisabled(unittest.TestCase):
    """When cache_embeddings=False, no cache file is written or read."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = GnnLayerConfig(cache_embeddings=False)
        self.cache_path = os.path.join(self.tmp, 'embeddings.npz')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_cache_file_written(self):
        df = _simple_df(4)
        orig = _emb.embed_segment_texts
        rng = np.random.default_rng(7)
        _emb.embed_segment_texts = lambda t, c: rng.standard_normal((len(t), 16)).astype('float32')
        try:
            _emb.load_or_build_segment_embeddings(df, self.cfg, cache_path=self.cache_path)
        finally:
            _emb.embed_segment_texts = orig
        self.assertFalse(os.path.isfile(self.cache_path))

    def test_always_encodes_when_cache_disabled(self):
        """Each call encodes all segments (no cache bypass)."""
        df = _simple_df(3)
        orig = _emb.embed_segment_texts
        call_counter = {'n': 0}
        rng = np.random.default_rng(8)

        def _fake(texts, config):
            call_counter['n'] += 1
            return rng.standard_normal((len(texts), 16)).astype('float32')

        _emb.embed_segment_texts = _fake
        try:
            _emb.load_or_build_segment_embeddings(df, self.cfg, cache_path=self.cache_path)
            n1 = call_counter['n']
            _emb.load_or_build_segment_embeddings(df, self.cfg, cache_path=self.cache_path)
            n2 = call_counter['n']
        finally:
            _emb.embed_segment_texts = orig

        self.assertEqual(n1, 1)
        self.assertEqual(n2, 2)


class TestTextHash(unittest.TestCase):
    """Internal _text_hash function contract."""

    def test_same_text_same_hash(self):
        h1 = _emb._text_hash("hello world")
        h2 = _emb._text_hash("hello world")
        self.assertEqual(h1, h2)

    def test_different_texts_different_hashes(self):
        h1 = _emb._text_hash("hello world")
        h2 = _emb._text_hash("goodbye world")
        self.assertNotEqual(h1, h2)

    def test_empty_string_handled(self):
        h = _emb._text_hash("")
        self.assertIsNotNone(h)
        self.assertIsInstance(h, str)

    def test_none_handled(self):
        h = _emb._text_hash(None)
        self.assertIsNotNone(h)


if __name__ == '__main__':
    unittest.main()
