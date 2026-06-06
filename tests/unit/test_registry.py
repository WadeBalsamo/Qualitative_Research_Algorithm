"""
tests/unit/test_registry.py
----------------------------
Unit tests for theme_framework/registry.py.

Covers:
  load(None)          → None
  load('vaamr')       → 5-theme ThemeFramework, canonical ids 0-4, canonical names
  load('purer')       → 5-theme ThemeFramework, canonical ids 0-4, canonical names
  load('bogus')       → raises KeyError
  lru_cache           → same object returned on repeat calls

SOURCE BUG DOCUMENTED:
  theme_framework/purer.py get_purer_framework() uses parents[2] which resolves to
  /root/QRA instead of the repo root (/root/QRA/Qualitative_Research_Algorithm), so
  it looks for PURER_FRAMEWORK.md at /root/QRA/PURER_FRAMEWORK.md which does not
  exist → FileNotFoundError.  get_vaamr_framework() uses parents[1] correctly.
  registry.load('purer') uses parents[1] in registry.py and therefore works correctly.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from theme_framework.registry import load


# ---------------------------------------------------------------------------
# Canonical name constants (from the real FRAMEWORK.md files)
# ---------------------------------------------------------------------------

_VAAMR_CANONICAL_NAMES = [
    "Pain Vigilance",
    "Experiential Avoidance",
    "Attention Regulation",
    "Metacognitive Awareness",
    "Cognitive and Sensory Reappraisal",
]

_PURER_CANONICAL_NAMES = [
    "Phenomenological",
    "Utilization",
    "Reframing",
    "Educate/Expectancy",
    "Reinforcement",
]


# ---------------------------------------------------------------------------
# load(None)
# ---------------------------------------------------------------------------

class TestLoadNone(unittest.TestCase):

    def test_load_none_returns_none(self):
        self.assertIsNone(load(None))

    def test_load_none_idempotent(self):
        self.assertIsNone(load(None))
        self.assertIsNone(load(None))


# ---------------------------------------------------------------------------
# load('vaamr')
# ---------------------------------------------------------------------------

class TestLoadVaamr(unittest.TestCase):

    def setUp(self):
        self.fw = load('vaamr')

    def test_returns_a_theme_framework(self):
        from theme_framework.theme_schema import ThemeFramework
        self.assertIsInstance(self.fw, ThemeFramework)

    def test_framework_name_is_vaamr(self):
        self.assertEqual(self.fw.name, "VAAMR")

    def test_five_themes(self):
        self.assertEqual(self.fw.num_themes, 5)

    def test_theme_ids_zero_through_four(self):
        ids = sorted(t.theme_id for t in self.fw.themes)
        self.assertEqual(ids, list(range(5)))

    def test_canonical_names(self):
        actual_names = {t.theme_id: t.name for t in self.fw.themes}
        for i, expected in enumerate(_VAAMR_CANONICAL_NAMES):
            self.assertEqual(
                actual_names[i], expected,
                msg=f"VAAMR theme {i}: expected {expected!r}, got {actual_names[i]!r}"
            )

    def test_all_themes_have_definitions(self):
        for t in self.fw.themes:
            self.assertTrue(t.definition, msg=f"Empty definition for theme_id={t.theme_id}")


# ---------------------------------------------------------------------------
# load('purer')
# ---------------------------------------------------------------------------

class TestLoadPurer(unittest.TestCase):

    def setUp(self):
        self.fw = load('purer')

    def test_returns_a_theme_framework(self):
        from theme_framework.theme_schema import ThemeFramework
        self.assertIsInstance(self.fw, ThemeFramework)

    def test_framework_name_is_purer(self):
        self.assertEqual(self.fw.name, "PURER")

    def test_five_moves(self):
        self.assertEqual(self.fw.num_themes, 5)

    def test_theme_ids_zero_through_four(self):
        ids = sorted(t.theme_id for t in self.fw.themes)
        self.assertEqual(ids, list(range(5)))

    def test_canonical_names(self):
        actual_names = {t.theme_id: t.name for t in self.fw.themes}
        for i, expected in enumerate(_PURER_CANONICAL_NAMES):
            self.assertEqual(
                actual_names[i], expected,
                msg=f"PURER move {i}: expected {expected!r}, got {actual_names[i]!r}"
            )

    def test_all_moves_have_definitions(self):
        for t in self.fw.themes:
            self.assertTrue(t.definition, msg=f"Empty definition for theme_id={t.theme_id}")


# ---------------------------------------------------------------------------
# load('bogus') → KeyError
# ---------------------------------------------------------------------------

class TestLoadBogus(unittest.TestCase):

    def test_unknown_name_raises_key_error(self):
        with self.assertRaises(KeyError):
            load('bogus')

    def test_case_sensitive_raises_key_error(self):
        """Registry keys are lowercase; mixed-case should be unknown."""
        with self.assertRaises(KeyError):
            load('VAAMR')

    def test_empty_string_raises_key_error(self):
        with self.assertRaises(KeyError):
            load('')


# ---------------------------------------------------------------------------
# lru_cache — same object on repeat calls
# ---------------------------------------------------------------------------

class TestLruCache(unittest.TestCase):

    def test_vaamr_same_object(self):
        fw1 = load('vaamr')
        fw2 = load('vaamr')
        self.assertIs(fw1, fw2, "lru_cache should return the same ThemeFramework object")

    def test_purer_same_object(self):
        fw1 = load('purer')
        fw2 = load('purer')
        self.assertIs(fw1, fw2, "lru_cache should return the same ThemeFramework object")

    def test_vaamr_and_purer_are_different_objects(self):
        vaamr = load('vaamr')
        purer = load('purer')
        self.assertIsNot(vaamr, purer)


# ---------------------------------------------------------------------------
# SOURCE BUG: purer.py get_purer_framework() uses parents[2]
# ---------------------------------------------------------------------------

class TestPurerModulePathBug(unittest.TestCase):
    """
    Guards the markdown path in theme_framework/purer.py.

    get_purer_framework() computes its markdown path as:
        Path(__file__).resolve().parents[1] / "PURER_FRAMEWORK.md"

    parents[1] of  .../theme_framework/purer.py  resolves to the repo root
    (/root/QRA/Qualitative_Research_Algorithm), where PURER_FRAMEWORK.md lives.
    A previous parents[2] bug pointed outside the repo and raised
    FileNotFoundError; this test pins the corrected behavior so it can't regress.
    registry.load('purer') and get_vaamr_framework() use the same parents[1] path.
    """

    def test_get_purer_framework_loads_correctly(self):
        """get_purer_framework() uses parents[1] (correct) and succeeds."""
        from theme_framework.purer import get_purer_framework
        fw = get_purer_framework()
        self.assertEqual(fw.num_themes, 5)

    def test_get_vaamr_framework_works_correctly(self):
        """get_vaamr_framework() uses parents[1] (correct) and succeeds."""
        from theme_framework.vaamr import get_vaamr_framework
        fw = get_vaamr_framework()
        self.assertEqual(fw.num_themes, 5)
        self.assertEqual(fw.name, "VAAMR")

    def test_registry_load_purer_works(self):
        """registry.load('purer') uses the correct parents[1] path and succeeds."""
        fw = load('purer')
        self.assertIsNotNone(fw)
        self.assertEqual(fw.num_themes, 5)
        self.assertEqual(fw.name, "PURER")


if __name__ == "__main__":
    unittest.main()
