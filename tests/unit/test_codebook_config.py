"""
tests/unit/test_codebook_config.py
------------------------------------
Unit tests for codebook/config.py.

Verifies the exact default values for all three dataclasses:
  - EmbeddingClassifierConfig
  - LLMCodebookConfig
  - EnsembleConfig

Defaults are cross-checked against the source to pin regressions.
"""

import os
import sys
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from codebook.config import EmbeddingClassifierConfig, LLMCodebookConfig, EnsembleConfig


class TestEmbeddingClassifierConfigDefaults(unittest.TestCase):
    """EmbeddingClassifierConfig default field values."""

    def setUp(self):
        self.cfg = EmbeddingClassifierConfig()

    def test_default_embedding_model(self):
        self.assertEqual(self.cfg.embedding_model, 'Qwen/Qwen3-Embedding-8B')

    def test_default_use_query_prefix(self):
        self.assertTrue(self.cfg.use_query_prefix)

    def test_default_embedding_batch_size(self):
        self.assertEqual(self.cfg.embedding_batch_size, 8)

    def test_default_similarity_threshold(self):
        self.assertAlmostEqual(self.cfg.similarity_threshold, 1.375)

    def test_default_max_codes_per_sentence_is_none(self):
        self.assertIsNone(self.cfg.max_codes_per_sentence)

    def test_default_criteria_weight(self):
        self.assertAlmostEqual(self.cfg.criteria_weight, 0.5)

    def test_default_exemplar_weight(self):
        self.assertAlmostEqual(self.cfg.exemplar_weight, 0.5)

    def test_default_exemplar_import_path_none(self):
        self.assertIsNone(self.cfg.exemplar_import_path)

    def test_default_exemplar_export_path_none(self):
        self.assertIsNone(self.cfg.exemplar_export_path)

    def test_default_max_exemplar_tokens(self):
        self.assertEqual(self.cfg.max_exemplar_tokens, 512)

    def test_default_exemplar_confidence_threshold(self):
        self.assertAlmostEqual(self.cfg.exemplar_confidence_threshold, 0.8)

    def test_default_two_pass(self):
        self.assertTrue(self.cfg.two_pass)

    # ------------------------------------------------------------------
    # Field override (dataclass mutability)
    # ------------------------------------------------------------------

    def test_override_embedding_model(self):
        cfg = EmbeddingClassifierConfig(embedding_model='all-MiniLM-L6-v2')
        self.assertEqual(cfg.embedding_model, 'all-MiniLM-L6-v2')

    def test_override_similarity_threshold(self):
        cfg = EmbeddingClassifierConfig(similarity_threshold=2.0)
        self.assertAlmostEqual(cfg.similarity_threshold, 2.0)

    def test_override_two_pass_false(self):
        cfg = EmbeddingClassifierConfig(two_pass=False)
        self.assertFalse(cfg.two_pass)

    def test_override_exemplar_import_path(self):
        cfg = EmbeddingClassifierConfig(exemplar_import_path='/tmp/ex.json')
        self.assertEqual(cfg.exemplar_import_path, '/tmp/ex.json')

    def test_override_use_query_prefix_false(self):
        cfg = EmbeddingClassifierConfig(use_query_prefix=False)
        self.assertFalse(cfg.use_query_prefix)

    # ------------------------------------------------------------------
    # Type checks
    # ------------------------------------------------------------------

    def test_embedding_model_is_str(self):
        self.assertIsInstance(self.cfg.embedding_model, str)

    def test_similarity_threshold_is_float(self):
        self.assertIsInstance(self.cfg.similarity_threshold, float)

    def test_two_pass_is_bool(self):
        self.assertIsInstance(self.cfg.two_pass, bool)

    def test_batch_size_is_int(self):
        self.assertIsInstance(self.cfg.embedding_batch_size, int)


class TestLLMCodebookConfigDefaults(unittest.TestCase):
    """LLMCodebookConfig default field values."""

    def setUp(self):
        self.cfg = LLMCodebookConfig()

    def test_default_n_runs(self):
        self.assertEqual(self.cfg.n_runs, 1)

    def test_default_max_codes_per_segment(self):
        self.assertEqual(self.cfg.max_codes_per_segment, 5)

    def test_default_confidence_threshold(self):
        self.assertAlmostEqual(self.cfg.confidence_threshold, 0.5)

    def test_default_randomize_codebook(self):
        self.assertTrue(self.cfg.randomize_codebook)

    def test_default_save_interval(self):
        self.assertEqual(self.cfg.save_interval, 20)

    def test_default_output_dir(self):
        self.assertEqual(self.cfg.output_dir, '')

    # ------------------------------------------------------------------
    # Field override
    # ------------------------------------------------------------------

    def test_override_n_runs(self):
        cfg = LLMCodebookConfig(n_runs=3)
        self.assertEqual(cfg.n_runs, 3)

    def test_override_confidence_threshold(self):
        cfg = LLMCodebookConfig(confidence_threshold=0.7)
        self.assertAlmostEqual(cfg.confidence_threshold, 0.7)

    def test_override_output_dir(self):
        cfg = LLMCodebookConfig(output_dir='/tmp/out')
        self.assertEqual(cfg.output_dir, '/tmp/out')


class TestEnsembleConfigDefaults(unittest.TestCase):
    """EnsembleConfig default field values."""

    def setUp(self):
        self.cfg = EnsembleConfig()

    def test_default_require_agreement(self):
        self.assertFalse(self.cfg.require_agreement)

    def test_default_flag_disagreements(self):
        self.assertTrue(self.cfg.flag_disagreements)

    def test_default_preferred_method(self):
        # Default is 'llm' (not 'both') — the LLM-preferred mode
        self.assertEqual(self.cfg.preferred_method, 'llm')

    # ------------------------------------------------------------------
    # Field override
    # ------------------------------------------------------------------

    def test_override_require_agreement(self):
        cfg = EnsembleConfig(require_agreement=True)
        self.assertTrue(cfg.require_agreement)

    def test_override_preferred_method_both(self):
        cfg = EnsembleConfig(preferred_method='both')
        self.assertEqual(cfg.preferred_method, 'both')

    def test_override_preferred_method_embedding(self):
        cfg = EnsembleConfig(preferred_method='embedding')
        self.assertEqual(cfg.preferred_method, 'embedding')

    def test_override_flag_disagreements_false(self):
        cfg = EnsembleConfig(flag_disagreements=False)
        self.assertFalse(cfg.flag_disagreements)

    # ------------------------------------------------------------------
    # Type checks
    # ------------------------------------------------------------------

    def test_require_agreement_is_bool(self):
        self.assertIsInstance(self.cfg.require_agreement, bool)

    def test_flag_disagreements_is_bool(self):
        self.assertIsInstance(self.cfg.flag_disagreements, bool)

    def test_preferred_method_is_str(self):
        self.assertIsInstance(self.cfg.preferred_method, str)


class TestConfigIsolation(unittest.TestCase):
    """Confirm independent instances don't share mutable state."""

    def test_embedding_configs_independent(self):
        a = EmbeddingClassifierConfig()
        b = EmbeddingClassifierConfig(embedding_model='other-model')
        self.assertNotEqual(a.embedding_model, b.embedding_model)
        # a's default is still the original
        self.assertEqual(a.embedding_model, 'Qwen/Qwen3-Embedding-8B')

    def test_ensemble_configs_independent(self):
        a = EnsembleConfig()
        b = EnsembleConfig(preferred_method='both')
        self.assertEqual(a.preferred_method, 'llm')
        self.assertEqual(b.preferred_method, 'both')


if __name__ == '__main__':
    unittest.main()
