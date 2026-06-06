"""
tests/unit/test_gnn_embeddings_remote.py
----------------------------------------
Unit tests for gnn_layer/embeddings_remote.py (the OpenAI-compatible
/v1/embeddings backend) and the embedding_backend dispatch in
gnn_layer/embeddings.py.

All hermetic: requests.post is monkeypatched so NO real network is used. The one
live-endpoint smoke test is guarded by @slow_test (set QRA_RUN_SLOW=1 to run).
"""

import json
import os
import sys
import unittest
from unittest import mock

import numpy as np

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from tests.testhelpers.marks import slow_test
from gnn_layer.config import GnnLayerConfig
import gnn_layer.embeddings as _emb
import gnn_layer.embeddings_remote as _rem


# ---------------------------------------------------------------------------
# HTTP fakes
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _make_fake_post(dim=4096, recorder=None, status=200):
    """Return a fake requests.post that records calls and returns `dim`-d vectors."""
    def _post(url, headers=None, json=None, timeout=None):
        if recorder is not None:
            recorder.append({'url': url, 'headers': headers, 'json': json, 'timeout': timeout})
        inputs = json['input']
        data = [{'index': i, 'embedding': [float(i)] * dim} for i in range(len(inputs))]
        return _FakeResp({'data': data}, status=status)
    return _post


def _cfg(**kw):
    base = dict(
        embedding_backend='openai',
        embedding_base_url='http://10.0.0.58:1234/v1',
        embedding_model='text-embedding-qwen3-embedding-8b',
        embedding_api_key=None,
        embedding_batch_size=8,
        use_query_prefix=True,
    )
    base.update(kw)
    return GnnLayerConfig(**base)


# ---------------------------------------------------------------------------
# Request shape: URL / model / body
# ---------------------------------------------------------------------------

class TestRequestShape(unittest.TestCase):
    def test_url_model_and_passage_body(self):
        rec = []
        with mock.patch('requests.post', _make_fake_post(dim=4, recorder=rec)):
            _rem.embed_texts_remote(['alpha', 'beta'], _cfg(), is_query=False)
        self.assertEqual(len(rec), 1)
        call = rec[0]
        self.assertEqual(call['url'], 'http://10.0.0.58:1234/v1/embeddings')
        self.assertEqual(call['json']['model'], 'text-embedding-qwen3-embedding-8b')
        # is_query=False → passages, raw text, no prefix
        self.assertEqual(call['json']['input'], ['alpha', 'beta'])

    def test_base_url_trailing_slash_normalized(self):
        rec = []
        with mock.patch('requests.post', _make_fake_post(dim=4, recorder=rec)):
            _rem.embed_texts_remote(['x'], _cfg(embedding_base_url='http://h:1234/v1/'),
                                    is_query=False)
        self.assertEqual(rec[0]['url'], 'http://h:1234/v1/embeddings')

    def test_missing_base_url_raises(self):
        with self.assertRaises(ValueError):
            _rem.embed_texts_remote(['x'], _cfg(embedding_base_url=None), is_query=False)


# ---------------------------------------------------------------------------
# Query / passage prefix parity
# ---------------------------------------------------------------------------

class TestQueryPrefix(unittest.TestCase):
    def test_prefix_applied_for_query(self):
        rec = []
        with mock.patch('requests.post', _make_fake_post(dim=4, recorder=rec)):
            _rem.embed_texts_remote(['noticing pain'], _cfg(), is_query=True)
        sent = rec[0]['json']['input'][0]
        # served id has no '/', so the constant is used; exact concat, no separator
        self.assertEqual(sent, _rem.QWEN3_QUERY_PROMPT + 'noticing pain')
        self.assertTrue(sent.startswith('Instruct: '))
        self.assertTrue(sent.endswith('Query:noticing pain'))

    def test_no_prefix_for_passage(self):
        rec = []
        with mock.patch('requests.post', _make_fake_post(dim=4, recorder=rec)):
            _rem.embed_texts_remote(['a definition'], _cfg(), is_query=False)
        self.assertEqual(rec[0]['json']['input'], ['a definition'])

    def test_no_prefix_when_use_query_prefix_false(self):
        rec = []
        with mock.patch('requests.post', _make_fake_post(dim=4, recorder=rec)):
            _rem.embed_texts_remote(['seg text'], _cfg(use_query_prefix=False), is_query=True)
        # query but prefixing disabled → raw text
        self.assertEqual(rec[0]['json']['input'], ['seg text'])

    def test_resolve_prompt_from_cached_hf_repo_id(self):
        """An HF repo id present in the offline cache resolves to its declared 'query' prompt."""
        fake_cfg = {'prompts': {'query': 'PFX:', 'document': ''}}
        m = mock.mock_open(read_data=json.dumps(fake_cfg))
        with mock.patch('huggingface_hub.try_to_load_from_cache', return_value='/tmp/x.json'), \
             mock.patch('os.path.isfile', return_value=True), \
             mock.patch('builtins.open', m):
            got = _rem._resolve_query_prompt('Qwen/Qwen3-Embedding-8B')
        self.assertEqual(got, 'PFX:')

    def test_resolve_prompt_falls_back_to_constant(self):
        # served id (no slash) never consults the cache
        self.assertEqual(_rem._resolve_query_prompt('text-embedding-qwen3-embedding-8b'),
                         _rem.QWEN3_QUERY_PROMPT)


# ---------------------------------------------------------------------------
# Batching + shape/dtype
# ---------------------------------------------------------------------------

class TestBatchingAndShape(unittest.TestCase):
    def test_batches_split_across_requests(self):
        rec = []
        texts = [f't{i}' for i in range(5)]
        with mock.patch('requests.post', _make_fake_post(dim=4, recorder=rec)):
            out = _rem.embed_texts_remote(texts, _cfg(embedding_batch_size=2), is_query=False)
        # 5 texts / batch 2 → 3 requests of sizes 2,2,1
        self.assertEqual(len(rec), 3)
        self.assertEqual([len(c['json']['input']) for c in rec], [2, 2, 1])
        self.assertEqual(out.shape, (5, 4))

    def test_shape_and_dtype(self):
        with mock.patch('requests.post', _make_fake_post(dim=16)):
            out = _rem.embed_texts_remote(['a', 'b', 'c'], _cfg(), is_query=False)
        self.assertEqual(out.shape, (3, 16))
        self.assertEqual(out.dtype, np.float32)

    def test_default_batch_size_used_when_unset(self):
        rec = []
        texts = [f't{i}' for i in range(20)]
        with mock.patch('requests.post', _make_fake_post(dim=2, recorder=rec)):
            _rem.embed_texts_remote(texts, _cfg(embedding_batch_size=0), is_query=False)
        # DEFAULT_BATCH_SIZE == 16 → 20 texts → 2 requests (16, 4)
        self.assertEqual([len(c['json']['input']) for c in rec], [_rem.DEFAULT_BATCH_SIZE, 4])

    def test_empty_input_returns_empty(self):
        rec = []
        with mock.patch('requests.post', _make_fake_post(dim=4, recorder=rec)):
            out = _rem.embed_texts_remote([], _cfg(), is_query=False)
        self.assertEqual(out.shape[0], 0)
        self.assertEqual(rec, [])  # no HTTP call for empty input

    def test_row_order_follows_index_field(self):
        """Out-of-order 'index' fields are re-sorted to input order."""
        def _shuffled_post(url, headers=None, json=None, timeout=None):
            n = len(json['input'])
            data = [{'index': i, 'embedding': [float(i)]} for i in reversed(range(n))]
            return _FakeResp({'data': data})
        with mock.patch('requests.post', _shuffled_post):
            out = _rem.embed_texts_remote(['a', 'b', 'c'], _cfg(), is_query=False)
        self.assertEqual(out.reshape(-1).tolist(), [0.0, 1.0, 2.0])


# ---------------------------------------------------------------------------
# Auth header
# ---------------------------------------------------------------------------

class TestAuthHeader(unittest.TestCase):
    def test_no_auth_header_without_key(self):
        rec = []
        with mock.patch('requests.post', _make_fake_post(dim=2, recorder=rec)):
            _rem.embed_texts_remote(['x'], _cfg(embedding_api_key=None), is_query=False)
        self.assertNotIn('Authorization', rec[0]['headers'])

    def test_auth_header_with_key(self):
        rec = []
        with mock.patch('requests.post', _make_fake_post(dim=2, recorder=rec)):
            _rem.embed_texts_remote(['x'], _cfg(embedding_api_key='secret'), is_query=False)
        self.assertEqual(rec[0]['headers']['Authorization'], 'Bearer secret')


# ---------------------------------------------------------------------------
# Retry / backoff
# ---------------------------------------------------------------------------

class TestRetry(unittest.TestCase):
    def test_retry_on_connection_error_then_success(self):
        import requests
        calls = {'n': 0}

        def _flaky(url, headers=None, json=None, timeout=None):
            calls['n'] += 1
            if calls['n'] == 1:
                raise requests.exceptions.ConnectionError('reset')
            data = [{'index': i, 'embedding': [1.0, 2.0]} for i in range(len(json['input']))]
            return _FakeResp({'data': data})

        with mock.patch('requests.post', _flaky):
            out = _rem.embed_texts_remote(['x'], _cfg(), is_query=False, retry_base_delay=0.0)
        self.assertEqual(calls['n'], 2)
        self.assertEqual(out.shape, (1, 2))

    def test_retry_on_transient_5xx_then_success(self):
        calls = {'n': 0}

        def _flaky(url, headers=None, json=None, timeout=None):
            calls['n'] += 1
            if calls['n'] == 1:
                return _FakeResp({'error': 'overloaded'}, status=503)
            data = [{'index': i, 'embedding': [1.0]} for i in range(len(json['input']))]
            return _FakeResp({'data': data})

        with mock.patch('requests.post', _flaky):
            out = _rem.embed_texts_remote(['x'], _cfg(), is_query=False, retry_base_delay=0.0)
        self.assertEqual(calls['n'], 2)
        self.assertEqual(out.shape, (1, 1))

    def test_exhausted_retries_raise(self):
        import requests

        def _always_fail(url, headers=None, json=None, timeout=None):
            raise requests.exceptions.Timeout('nope')

        with mock.patch('requests.post', _always_fail):
            with self.assertRaises(RuntimeError):
                _rem.embed_texts_remote(['x'], _cfg(), is_query=False,
                                        max_retries=3, retry_base_delay=0.0)

    def test_non_retryable_4xx_raises_immediately(self):
        calls = {'n': 0}

        def _bad_request(url, headers=None, json=None, timeout=None):
            calls['n'] += 1
            return _FakeResp({'error': 'bad model'}, status=400)

        with mock.patch('requests.post', _bad_request):
            with self.assertRaises(RuntimeError):
                _rem.embed_texts_remote(['x'], _cfg(), is_query=False,
                                        max_retries=3, retry_base_delay=0.0)
        self.assertEqual(calls['n'], 1)  # not retried


# ---------------------------------------------------------------------------
# Backend dispatch in embeddings.py
# ---------------------------------------------------------------------------

class TestDispatch(unittest.TestCase):
    def test_openai_routes_segments_and_anchors_to_remote(self):
        rec = []

        def _fake_remote(texts, config, is_query):
            rec.append(is_query)
            return np.zeros((len(texts), 3), dtype=np.float32)

        cfg = _cfg()
        with mock.patch.object(_rem, 'embed_texts_remote', _fake_remote):
            _emb.embed_segment_texts(['a'], cfg)
            _emb.embed_anchor_texts(['b'], cfg)
        self.assertEqual(rec, [True, False])  # segments=query, anchors=passage

    def test_local_does_not_route_to_remote(self):
        cfg = GnnLayerConfig(embedding_backend='local', embedding_model='all-MiniLM-L6-v2')

        class _FakeEmbedder:
            def __init__(self):
                self.q, self.p = 0, 0

            def _embed_queries(self, texts):
                self.q += 1
                return np.zeros((len(texts), 3), dtype=np.float32)

            def _embed(self, texts):
                self.p += 1
                return np.zeros((len(texts), 3), dtype=np.float32)

        fake = _FakeEmbedder()
        remote_called = {'n': 0}

        def _boom(*a, **k):
            remote_called['n'] += 1
            raise AssertionError('remote must not be called on local backend')

        with mock.patch.object(_emb, '_make_embedder', return_value=fake), \
             mock.patch.object(_rem, 'embed_texts_remote', _boom):
            _emb.embed_segment_texts(['a'], cfg)
            _emb.embed_anchor_texts(['b'], cfg)
        self.assertEqual(remote_called['n'], 0)
        self.assertEqual((fake.q, fake.p), (1, 1))


# ---------------------------------------------------------------------------
# Live endpoint smoke check (slow; real network)
# ---------------------------------------------------------------------------

@slow_test
class TestLiveSmoke(unittest.TestCase):
    def test_live_qwen3_embeddings(self):
        cfg = _cfg()  # backend='openai', base_url=10.0.0.58:1234/v1, served qwen3 id
        texts = [
            'noticing pain in my back',
            'I just try to push it away',
            'I watched my mind getting anxious',
        ]
        out = _rem.embed_texts_remote(texts, cfg, is_query=True)
        self.assertEqual(out.shape, (3, 4096))
        self.assertEqual(out.dtype, np.float32)
        norm0 = float(np.linalg.norm(out[0]))
        print(f"\n[live smoke] shape={out.shape} dtype={out.dtype}")
        print(f"[live smoke] vec0[:4]={out[0][:4].tolist()}")
        print(f"[live smoke] ||vec0||2={norm0:.4f} (≈1.0 ⇒ endpoint normalizes)")


if __name__ == '__main__':
    unittest.main()
