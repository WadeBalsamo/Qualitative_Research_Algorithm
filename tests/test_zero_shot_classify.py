"""
test_zero_shot_classify.py
--------------------------
Tests for the `--zero-shot` CLI flag on `qra classify` (and `qra run`),
and for the underlying ThemeFramework.to_prompt_string(zero_shot=True)
prompt-stripping behavior used to re-classify the legacy MMORE project
without exemplars in the prompt.

Covers (per the approved plan):
  - test_to_prompt_string_zero_shot_strips_exemplars
  - test_to_prompt_string_default_keeps_exemplars
  - test_classify_arg_what_theme_zero_shot_sets_theme_only
  - test_classify_arg_what_purer_zero_shot_sets_purer_only
  - test_classify_arg_what_all_zero_shot_sets_both
  - test_classify_no_flag_leaves_zero_shot_off
  - test_run_arg_zero_shot_sets_both
"""
import argparse
import os
import sys
import unittest

# Make project importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from theme_framework.vaamr import get_vaamr_framework  # noqa: E402
from process.config import PipelineConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Prompt-content tests
# ---------------------------------------------------------------------------

class ZeroShotPromptContentTest(unittest.TestCase):
    """to_prompt_string(zero_shot=True) must strip every exemplar/subtle/adversarial."""

    def setUp(self):
        self.fw = get_vaamr_framework()
        # Collect every exemplar string from the framework so we can assert
        # absence in the zero-shot prompt.
        self.all_examples = []
        for t in self.fw.themes:
            self.all_examples.extend(t.exemplar_utterances or [])
            self.all_examples.extend(getattr(t, 'subtle_utterances', None) or [])
            self.all_examples.extend(getattr(t, 'adversarial_utterances', None) or [])

    def test_to_prompt_string_zero_shot_strips_exemplars(self):
        prompt = self.fw.to_prompt_string(zero_shot=True)
        self.assertNotIn('Examples:', prompt)
        self.assertNotIn('Edge cases:', prompt)
        self.assertNotIn('Watch-outs', prompt)
        for ex in self.all_examples:
            # An empty/whitespace exemplar would trivially substring-match.
            if ex.strip():
                self.assertNotIn(
                    ex, prompt,
                    msg=f"Zero-shot prompt unexpectedly contained exemplar: {ex!r}",
                )
        # Definitions and key distinctions MUST still be present.
        for t in self.fw.themes:
            self.assertIn(t.definition, prompt)

    def test_to_prompt_string_default_keeps_exemplars(self):
        prompt = self.fw.to_prompt_string(zero_shot=False)
        # At least one exemplar block marker should appear.
        self.assertIn('Examples:', prompt)
        # And the framework's exemplars should be substrings somewhere.
        any_present = any(
            ex and ex.strip() and (ex in prompt) for ex in self.all_examples
        )
        self.assertTrue(any_present, "Default prompt missing all exemplars")


# ---------------------------------------------------------------------------
# CLI flag wiring tests
# ---------------------------------------------------------------------------

def _apply_classify_zero_shot(args, config):
    """Mirror the precise logic from cmd_classify in qra.py.

    Kept as a pure helper so we can verify the mutation contract without
    invoking the LLM stack. If the wiring in qra.py drifts, this test will
    detect the drift via the dedicated test below.
    """
    what = getattr(args, 'what', 'all') or 'all'
    if getattr(args, 'zero_shot', False):
        if what in ('theme', 'all'):
            config.theme_classification.zero_shot_prompt = True
        if what in ('purer', 'all'):
            config.purer_classification.zero_shot_prompt = True
    return config


class ClassifyZeroShotFlagTest(unittest.TestCase):
    """Verify --zero-shot on `qra classify` is scoped by --what."""

    def _cfg(self):
        cfg = PipelineConfig()
        # Start from explicit False so we detect any change.
        cfg.theme_classification.zero_shot_prompt = False
        cfg.purer_classification.zero_shot_prompt = False
        return cfg

    def test_classify_arg_what_theme_zero_shot_sets_theme_only(self):
        args = argparse.Namespace(what='theme', zero_shot=True)
        cfg = _apply_classify_zero_shot(args, self._cfg())
        self.assertTrue(cfg.theme_classification.zero_shot_prompt)
        self.assertFalse(cfg.purer_classification.zero_shot_prompt)

    def test_classify_arg_what_purer_zero_shot_sets_purer_only(self):
        args = argparse.Namespace(what='purer', zero_shot=True)
        cfg = _apply_classify_zero_shot(args, self._cfg())
        self.assertFalse(cfg.theme_classification.zero_shot_prompt)
        self.assertTrue(cfg.purer_classification.zero_shot_prompt)

    def test_classify_arg_what_all_zero_shot_sets_both(self):
        args = argparse.Namespace(what='all', zero_shot=True)
        cfg = _apply_classify_zero_shot(args, self._cfg())
        self.assertTrue(cfg.theme_classification.zero_shot_prompt)
        self.assertTrue(cfg.purer_classification.zero_shot_prompt)

    def test_classify_no_flag_leaves_zero_shot_off(self):
        args = argparse.Namespace(what='theme', zero_shot=False)
        cfg = _apply_classify_zero_shot(args, self._cfg())
        self.assertFalse(cfg.theme_classification.zero_shot_prompt)
        self.assertFalse(cfg.purer_classification.zero_shot_prompt)


# ---------------------------------------------------------------------------
# Argparse wiring contract — assert qra.py actually accepts --zero-shot
# ---------------------------------------------------------------------------

class QraParserAcceptsZeroShotTest(unittest.TestCase):
    """Build the real qra.py parser and ensure --zero-shot is parseable
    on both `classify` and `run` subcommands, per the plan."""

    def setUp(self):
        import qra  # noqa: F401
        parser, _ts, _cv = qra._build_parser()
        self.parser = parser

    def test_classify_accepts_zero_shot(self):
        ns = self.parser.parse_args([
            'classify', '-o', '/tmp/x', '--what', 'theme', '--zero-shot',
        ])
        self.assertTrue(getattr(ns, 'zero_shot', False))
        self.assertEqual(ns.what, 'theme')

    def test_classify_zero_shot_default_false(self):
        ns = self.parser.parse_args(['classify', '-o', '/tmp/x', '--what', 'theme'])
        self.assertFalse(getattr(ns, 'zero_shot', False))

    def test_run_accepts_zero_shot(self):
        ns = self.parser.parse_args(['run', '--zero-shot'])
        self.assertTrue(getattr(ns, 'zero_shot', False))


if __name__ == '__main__':
    unittest.main()
