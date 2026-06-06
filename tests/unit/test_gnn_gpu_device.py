"""
tests/unit/test_gnn_gpu_device.py
---------------------------------
GPU/device plumbing for the GNN layer (the layer must use CUDA when available and
honor the configured device, falling back to CPU cleanly).

Covers (hermetic — EmbeddingCodebookClassifier construction is lazy, no model download):
  * EmbeddingClassifierConfig exposes a `device` field (default None → auto)
  * _make_embedder caches one embedder per run, forwards `device`, rebuilds on change,
    and release_embedder() clears the cache
  * _device(config) resolves None → cuda when available else cpu, and honors an explicit value
  * a trained checkpoint reloads onto the configured device (load_checkpoint .to(dev))
  * set_seed seeds without error (incl. CUDA path)
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.testhelpers import synthetic_df
from gnn_layer.config import GnnLayerConfig
from gnn_layer import embeddings as E


def _seg_emb(df, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    return {sid: rng.standard_normal(dim).astype('float32') for sid in df['segment_id']}


class TestEmbeddingDeviceConfig(unittest.TestCase):

    def test_embedding_config_has_device_field(self):
        from codebook.config import EmbeddingClassifierConfig
        self.assertIsNone(EmbeddingClassifierConfig().device)
        self.assertEqual(EmbeddingClassifierConfig(device='cuda:1').device, 'cuda:1')

    def tearDown(self):
        E.release_embedder()

    def test_make_embedder_caches_and_forwards_device(self):
        cfg = GnnLayerConfig(embedding_model='all-MiniLM-L6-v2', device='cpu')
        e1 = E._make_embedder(cfg)
        e2 = E._make_embedder(cfg)
        self.assertIs(e1, e2)                       # one model per run
        self.assertEqual(e1.config.device, 'cpu')   # device forwarded to the embedder

    def test_release_embedder_clears_cache(self):
        cfg = GnnLayerConfig(embedding_model='all-MiniLM-L6-v2', device='cpu')
        e1 = E._make_embedder(cfg)
        E.release_embedder()
        e2 = E._make_embedder(cfg)
        self.assertIsNot(e1, e2)

    def test_make_embedder_rebuilds_on_device_change(self):
        e1 = E._make_embedder(GnnLayerConfig(embedding_model='all-MiniLM-L6-v2', device='cpu'))
        e2 = E._make_embedder(GnnLayerConfig(embedding_model='all-MiniLM-L6-v2', device='cuda'))
        self.assertIsNot(e1, e2)
        self.assertEqual(e2.config.device, 'cuda')

    def test_release_embedder_safe_when_empty(self):
        E.release_embedder()
        E.release_embedder()  # idempotent / no crash


class TestDeviceResolution(unittest.TestCase):

    def test_device_none_resolves_to_available(self):
        import torch
        from gnn_layer.train import _device
        dev = _device(GnnLayerConfig())
        expected = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.assertEqual(dev.type, expected)

    def test_explicit_device_honored(self):
        from gnn_layer.train import _device
        self.assertEqual(_device(GnnLayerConfig(device='cpu')).type, 'cpu')

    def test_set_seed_no_crash(self):
        from gnn_layer.train import set_seed
        set_seed(123)  # must seed CPU (and CUDA when available) without error


class TestCheckpointDevice(unittest.TestCase):

    def test_load_checkpoint_on_configured_device(self):
        from gnn_layer import graph_builder as gb, train as tr
        from gnn_layer.soft_labels import build_soft_targets
        cfg = GnnLayerConfig(hidden_dim=16, n_layers=2, knn_k=3, epochs=4,
                             cache_embeddings=False, seed=1,
                             objectives=['soft_vaamr', 'progression', 'purer'])
        df = synthetic_df(n_sessions=2)
        g = gb.build_graph(df, _seg_emb(df), cfg)
        tgts = tr.assemble_targets(g, build_soft_targets(df, 'weak'), cfg, df_all=df)
        model, _ = tr.train_model(g, tgts, cfg)
        d = tempfile.mkdtemp()
        tr.export_checkpoint(model, cfg, d)
        gb.save_graph(g, d)
        loaded = tr.load_checkpoint(d, g, cfg)
        want = tr._device(cfg)
        dev = next(loaded.parameters()).device
        self.assertEqual(dev.type, want.type)


if __name__ == '__main__':
    unittest.main()
