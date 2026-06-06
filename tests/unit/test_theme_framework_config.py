"""
tests/unit/test_theme_framework_config.py
------------------------------------------
Unit tests for theme_framework/config.py.

Covers ThemeClassificationConfig:
  - All default field values
  - Field types
  - Mutable list defaults are independent instances
  - Override via constructor arguments
  - api_key reads from environment (OPENROUTER_API_KEY) when set
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from theme_framework.config import ThemeClassificationConfig


class TestThemeClassificationConfigDefaults(unittest.TestCase):
    """Assert every documented default value in ThemeClassificationConfig."""

    def setUp(self):
        # Ensure OPENROUTER_API_KEY is absent so api_key default is ''
        self._orig_key = os.environ.pop('OPENROUTER_API_KEY', None)
        self.cfg = ThemeClassificationConfig()

    def tearDown(self):
        if self._orig_key is not None:
            os.environ['OPENROUTER_API_KEY'] = self._orig_key
        elif 'OPENROUTER_API_KEY' in os.environ:
            del os.environ['OPENROUTER_API_KEY']

    # --- Model fields ---

    def test_default_model(self):
        self.assertEqual(self.cfg.model, 'nvidia/nemotron-3-super')

    def test_default_summarization_model(self):
        self.assertEqual(self.cfg.summarization_model, 'nvidia/nemotron-3-nano-4b')

    def test_default_models_empty_list(self):
        self.assertEqual(self.cfg.models, [])

    def test_default_per_run_models_empty_list(self):
        self.assertEqual(self.cfg.per_run_models, [])

    # --- Sampling / run control ---

    def test_default_temperature(self):
        self.assertAlmostEqual(self.cfg.temperature, 0.0)

    def test_default_n_runs(self):
        self.assertEqual(self.cfg.n_runs, 1)

    def test_default_randomize_codebook(self):
        self.assertTrue(self.cfg.randomize_codebook)

    # --- Backend / connection ---

    def test_default_backend(self):
        self.assertEqual(self.cfg.backend, 'lmstudio')

    def test_default_lmstudio_base_url(self):
        self.assertEqual(self.cfg.lmstudio_base_url, 'http://127.0.0.1:1234/v1')

    def test_default_ollama_host(self):
        self.assertEqual(self.cfg.ollama_host, '0.0.0.0')

    def test_default_ollama_port(self):
        self.assertEqual(self.cfg.ollama_port, 11434)

    def test_default_api_key_empty_when_env_absent(self):
        self.assertEqual(self.cfg.api_key, '')

    # --- Token / I/O ---

    def test_default_max_new_tokens(self):
        self.assertEqual(self.cfg.max_new_tokens, 512)

    def test_default_output_dir(self):
        self.assertEqual(self.cfg.output_dir, './data/output/llm_labels/')

    def test_default_save_interval(self):
        self.assertEqual(self.cfg.save_interval, 20)

    # --- Context window ---

    def test_default_context_window_segments(self):
        self.assertEqual(self.cfg.context_window_segments, 2)

    # --- Prompt exemplar control ---

    def test_default_zero_shot_prompt(self):
        self.assertFalse(self.cfg.zero_shot_prompt)

    def test_default_prompt_n_exemplars_none(self):
        self.assertIsNone(self.cfg.prompt_n_exemplars)

    def test_default_prompt_include_subtle(self):
        self.assertTrue(self.cfg.prompt_include_subtle)

    def test_default_prompt_n_subtle_none(self):
        self.assertIsNone(self.cfg.prompt_n_subtle)

    def test_default_prompt_include_adversarial(self):
        self.assertTrue(self.cfg.prompt_include_adversarial)

    def test_default_prompt_n_adversarial_none(self):
        self.assertIsNone(self.cfg.prompt_n_adversarial)

    # --- Evidence-based reconciliation ---

    def test_default_evidence_secondary_weight(self):
        self.assertAlmostEqual(self.cfg.evidence_secondary_weight, 0.6)

    def test_default_evidence_presence_threshold(self):
        self.assertAlmostEqual(self.cfg.evidence_presence_threshold, 0.5)

    # --- Merging ---

    def test_default_min_classifiable_words(self):
        self.assertEqual(self.cfg.min_classifiable_words, 10)


class TestThemeClassificationConfigTypes(unittest.TestCase):
    """Verify field types for the default instance."""

    def setUp(self):
        self._orig_key = os.environ.pop('OPENROUTER_API_KEY', None)
        self.cfg = ThemeClassificationConfig()

    def tearDown(self):
        if self._orig_key is not None:
            os.environ['OPENROUTER_API_KEY'] = self._orig_key
        elif 'OPENROUTER_API_KEY' in os.environ:
            del os.environ['OPENROUTER_API_KEY']

    def test_model_is_str(self):
        self.assertIsInstance(self.cfg.model, str)

    def test_temperature_is_float(self):
        self.assertIsInstance(self.cfg.temperature, float)

    def test_n_runs_is_int(self):
        self.assertIsInstance(self.cfg.n_runs, int)

    def test_models_is_list(self):
        self.assertIsInstance(self.cfg.models, list)

    def test_per_run_models_is_list(self):
        self.assertIsInstance(self.cfg.per_run_models, list)

    def test_zero_shot_prompt_is_bool(self):
        self.assertIsInstance(self.cfg.zero_shot_prompt, bool)

    def test_randomize_codebook_is_bool(self):
        self.assertIsInstance(self.cfg.randomize_codebook, bool)

    def test_evidence_secondary_weight_is_float(self):
        self.assertIsInstance(self.cfg.evidence_secondary_weight, float)


class TestThemeClassificationConfigMutableDefaults(unittest.TestCase):
    """Mutable list fields must be independent across instances."""

    def setUp(self):
        os.environ.pop('OPENROUTER_API_KEY', None)

    def test_models_lists_are_independent(self):
        cfg1 = ThemeClassificationConfig()
        cfg2 = ThemeClassificationConfig()
        cfg1.models.append('model_x')
        self.assertNotIn('model_x', cfg2.models)

    def test_per_run_models_lists_are_independent(self):
        cfg1 = ThemeClassificationConfig()
        cfg2 = ThemeClassificationConfig()
        cfg1.per_run_models.append('model_y')
        self.assertNotIn('model_y', cfg2.per_run_models)


class TestThemeClassificationConfigOverrides(unittest.TestCase):
    """Constructor overrides work for all key fields."""

    def test_override_model(self):
        cfg = ThemeClassificationConfig(model='gpt-4o')
        self.assertEqual(cfg.model, 'gpt-4o')

    def test_override_temperature(self):
        cfg = ThemeClassificationConfig(temperature=0.7)
        self.assertAlmostEqual(cfg.temperature, 0.7)

    def test_override_n_runs(self):
        cfg = ThemeClassificationConfig(n_runs=3)
        self.assertEqual(cfg.n_runs, 3)

    def test_override_backend(self):
        cfg = ThemeClassificationConfig(backend='openrouter')
        self.assertEqual(cfg.backend, 'openrouter')

    def test_override_zero_shot(self):
        cfg = ThemeClassificationConfig(zero_shot_prompt=True)
        self.assertTrue(cfg.zero_shot_prompt)

    def test_override_context_window(self):
        cfg = ThemeClassificationConfig(context_window_segments=6)
        self.assertEqual(cfg.context_window_segments, 6)

    def test_override_min_classifiable_words(self):
        cfg = ThemeClassificationConfig(min_classifiable_words=0)
        self.assertEqual(cfg.min_classifiable_words, 0)

    def test_override_n_exemplars(self):
        cfg = ThemeClassificationConfig(prompt_n_exemplars=2)
        self.assertEqual(cfg.prompt_n_exemplars, 2)

    def test_override_models_list(self):
        cfg = ThemeClassificationConfig(models=['a', 'b'])
        self.assertEqual(cfg.models, ['a', 'b'])

    def test_override_per_run_models(self):
        cfg = ThemeClassificationConfig(per_run_models=['m1', 'm2', 'm3'])
        self.assertEqual(cfg.per_run_models, ['m1', 'm2', 'm3'])


class TestThemeClassificationConfigApiKeyEnv(unittest.TestCase):
    """api_key default reads OPENROUTER_API_KEY from the environment."""

    def test_api_key_from_env(self):
        os.environ['OPENROUTER_API_KEY'] = 'test-secret-key'
        try:
            cfg = ThemeClassificationConfig()
            self.assertEqual(cfg.api_key, 'test-secret-key')
        finally:
            del os.environ['OPENROUTER_API_KEY']

    def test_api_key_empty_without_env(self):
        os.environ.pop('OPENROUTER_API_KEY', None)
        cfg = ThemeClassificationConfig()
        self.assertEqual(cfg.api_key, '')

    def test_api_key_override_ignores_env(self):
        os.environ['OPENROUTER_API_KEY'] = 'env-key'
        try:
            cfg = ThemeClassificationConfig(api_key='override-key')
            self.assertEqual(cfg.api_key, 'override-key')
        finally:
            del os.environ['OPENROUTER_API_KEY']


if __name__ == "__main__":
    unittest.main()
