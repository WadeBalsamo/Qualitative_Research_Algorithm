"""
Integration: real tiny embedding model (all-MiniLM-L6-v2).

Exercises the SAME embedding entry point the GNN/codebook layers use
(gnn_layer.embeddings.embed_segment_texts) with a real, small sentence-
transformer — no mocking. Downloads ~80MB on first run.
"""
import os
import sys
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from tests.testhelpers import TINY_EMBED, build_tiny_config, integration_test


@integration_test
class TestRealEmbeddings(unittest.TestCase):
    def setUp(self):
        self.cfg = build_tiny_config(os.path.join(
            os.path.dirname(__file__), os.pardir, "testrun-outputs", "emb"))

    def test_embed_shape_and_determinism(self):
        import numpy as np
        from gnn_layer import embeddings as emb
        texts = [
            "I can't stop thinking about the pain.",
            "I stayed with the sensation and kept bringing my attention back.",
            "When I really looked, the pain was many different feelings.",
        ]
        try:
            m1 = emb.embed_segment_texts(texts, self.cfg.gnn_layer)
        except Exception as e:  # model could not be loaded in this environment
            raise unittest.SkipTest(f"could not load {TINY_EMBED}: {e}")
        self.assertEqual(m1.shape[0], 3)
        self.assertGreater(m1.shape[1], 0)
        self.assertEqual(str(m1.dtype), "float32")
        # Deterministic at temperature 0 / eval mode.
        m2 = emb.embed_segment_texts(texts, self.cfg.gnn_layer)
        self.assertTrue(np.allclose(m1, m2, atol=1e-4))
        # Distinct sentences get distinct vectors.
        self.assertFalse(np.allclose(m1[0], m1[1], atol=1e-3))


if __name__ == "__main__":
    unittest.main()
