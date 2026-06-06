"""
tests/unit/test_theme_schema.py
--------------------------------
Unit tests for theme_framework/theme_schema.py.

Covers:
  ThemeDefinition  — construction, field defaults
  ThemeFramework   — build_name_to_id_map (name/short_name/prompt_name/key/aliases,
                       all lowercased; collision last-writer-wins)
                   — build_id_to_short_map
                   — get_theme_by_id (hit + miss -> None)
                   — num_themes property
                   — stages property (alias for themes)
                   — to_json (structure, sorted by theme_id)
                   — to_prompt_string (zero_shot omits examples; non-zero_shot
                       includes exemplars/edge-cases/watch-outs; n_* caps;
                       randomize toggle)
"""
import os
import sys
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from theme_framework.theme_schema import ThemeDefinition, ThemeFramework
from tests.testhelpers import tiny_vaamr_framework


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_theme(theme_id, name, short_name=None, prompt_name=None, key=None, aliases=None,
                exemplars=None, subtle=None, adversarial=None):
    """Minimal ThemeDefinition factory for ad-hoc tests."""
    return ThemeDefinition(
        theme_id=theme_id,
        key=key or name.lower().replace(' ', '_'),
        name=name,
        short_name=short_name or name[:3].upper(),
        prompt_name=prompt_name or name,
        definition=f"Definition of {name}.",
        prototypical_features=[f"{name} feature"],
        distinguishing_criteria=f"{name} distinction.",
        exemplar_utterances=exemplars or [f"Exemplar for {name}."],
        subtle_utterances=subtle or [f"Subtle {name}."],
        adversarial_utterances=adversarial or [f"Near-miss for {name}."],
        aliases=aliases or [],
    )


def _two_theme_framework():
    """A minimal ThemeFramework with two themes for collision / basic tests."""
    td0 = _make_theme(0, "Alpha", short_name="ALF", prompt_name="Alpha Prompt",
                      key="alpha_key", aliases=["a1", "alpha_alias"])
    td1 = _make_theme(1, "Beta", short_name="BET", prompt_name="Beta Prompt",
                      key="beta_key", aliases=["b1"])
    return ThemeFramework(name="TestFW", version="1.0", description="test", themes=[td0, td1])


# ---------------------------------------------------------------------------
# ThemeDefinition tests
# ---------------------------------------------------------------------------

class TestThemeDefinition(unittest.TestCase):

    def test_required_fields_stored(self):
        td = _make_theme(3, "Metacognition", short_name="MET",
                         prompt_name="Metacognition Prompt", key="meta")
        self.assertEqual(td.theme_id, 3)
        self.assertEqual(td.name, "Metacognition")
        self.assertEqual(td.short_name, "MET")
        self.assertEqual(td.prompt_name, "Metacognition Prompt")
        self.assertEqual(td.key, "meta")

    def test_default_list_fields_are_empty(self):
        td = ThemeDefinition(
            theme_id=0, key="k", name="N", short_name="SN", prompt_name="PN",
            definition="d", prototypical_features=[], distinguishing_criteria="c",
            exemplar_utterances=[],
        )
        self.assertEqual(td.subtle_utterances, [])
        self.assertEqual(td.adversarial_utterances, [])
        self.assertEqual(td.word_prototypes, [])
        self.assertEqual(td.aliases, [])

    def test_aliases_stored(self):
        td = _make_theme(0, "Vigilance", aliases=["pain_focus", "hypervigilance"])
        self.assertIn("pain_focus", td.aliases)
        self.assertIn("hypervigilance", td.aliases)


# ---------------------------------------------------------------------------
# ThemeFramework — properties
# ---------------------------------------------------------------------------

class TestThemeFrameworkProperties(unittest.TestCase):

    def setUp(self):
        self.fw = tiny_vaamr_framework()

    def test_num_themes(self):
        self.assertEqual(self.fw.num_themes, 5)

    def test_stages_is_themes(self):
        # stages is a property alias for themes
        self.assertIs(self.fw.stages, self.fw.themes)

    def test_get_theme_by_id_hit(self):
        t = self.fw.get_theme_by_id(0)
        self.assertIsNotNone(t)
        self.assertEqual(t.name, "Vigilance")

    def test_get_theme_by_id_all_canonical_ids(self):
        for tid in range(5):
            self.assertIsNotNone(self.fw.get_theme_by_id(tid),
                                 msg=f"Expected theme for id={tid}")

    def test_get_theme_by_id_miss_returns_none(self):
        self.assertIsNone(self.fw.get_theme_by_id(99))
        self.assertIsNone(self.fw.get_theme_by_id(-1))

    def test_num_themes_empty_framework(self):
        fw = ThemeFramework(name="Empty", version="0", description="", themes=[])
        self.assertEqual(fw.num_themes, 0)


# ---------------------------------------------------------------------------
# build_name_to_id_map
# ---------------------------------------------------------------------------

class TestBuildNameToIdMap(unittest.TestCase):

    def setUp(self):
        self.fw = _two_theme_framework()
        self.m = self.fw.build_name_to_id_map()

    def test_name_lowercased(self):
        self.assertEqual(self.m.get("alpha"), 0)
        self.assertEqual(self.m.get("beta"), 1)

    def test_short_name_lowercased(self):
        self.assertEqual(self.m.get("alf"), 0)
        self.assertEqual(self.m.get("bet"), 1)

    def test_prompt_name_lowercased(self):
        self.assertEqual(self.m.get("alpha prompt"), 0)
        self.assertEqual(self.m.get("beta prompt"), 1)

    def test_key_lowercased(self):
        self.assertEqual(self.m.get("alpha_key"), 0)
        self.assertEqual(self.m.get("beta_key"), 1)

    def test_aliases_included(self):
        self.assertEqual(self.m.get("a1"), 0)
        self.assertEqual(self.m.get("alpha_alias"), 0)
        self.assertEqual(self.m.get("b1"), 1)

    def test_values_are_ints(self):
        for v in self.m.values():
            self.assertIsInstance(v, int)

    def test_keys_are_lowercase_strings(self):
        for k in self.m.keys():
            self.assertEqual(k, k.lower())

    def test_multi_word_name(self):
        fw = tiny_vaamr_framework()
        m = fw.build_name_to_id_map()
        # "Attention Regulation" → id 2
        self.assertEqual(m.get("attention regulation"), 2)

    def test_collision_last_writer_wins(self):
        """When two themes share an alias key the last theme in the list wins."""
        td0 = _make_theme(0, "Alpha", aliases=["shared"])
        td1 = _make_theme(1, "Beta", aliases=["shared"])
        fw = ThemeFramework(name="FW", version="1", description="", themes=[td0, td1])
        m = fw.build_name_to_id_map()
        self.assertEqual(m["shared"], 1)  # td1 processed last → overwrites td0

    def test_empty_framework_returns_empty_map(self):
        fw = ThemeFramework(name="FW", version="1", description="", themes=[])
        self.assertEqual(fw.build_name_to_id_map(), {})

    def test_theme_without_aliases_has_no_alias_keys(self):
        td = _make_theme(0, "Solo", aliases=[])
        fw = ThemeFramework(name="FW", version="1", description="", themes=[td])
        m = fw.build_name_to_id_map()
        # name/prompt_name/key all lowercase to "solo"; short_name → "sol"
        self.assertEqual(len(m), 2)


# ---------------------------------------------------------------------------
# build_id_to_short_map
# ---------------------------------------------------------------------------

class TestBuildIdToShortMap(unittest.TestCase):

    def test_maps_id_to_short_name(self):
        fw = tiny_vaamr_framework()
        s = fw.build_id_to_short_map()
        self.assertEqual(s[0], "VIG")
        self.assertEqual(s[1], "AVD")
        self.assertEqual(s[2], "ATT")
        self.assertEqual(s[3], "MET")
        self.assertEqual(s[4], "REA")

    def test_keys_are_ints(self):
        fw = tiny_vaamr_framework()
        s = fw.build_id_to_short_map()
        for k in s.keys():
            self.assertIsInstance(k, int)

    def test_empty_framework(self):
        fw = ThemeFramework(name="FW", version="1", description="", themes=[])
        self.assertEqual(fw.build_id_to_short_map(), {})


# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------

class TestToJson(unittest.TestCase):

    def setUp(self):
        self.fw = tiny_vaamr_framework()
        self.j = self.fw.to_json()

    def test_top_level_keys(self):
        self.assertIn("framework", self.j)
        self.assertIn("version", self.j)
        self.assertIn("themes", self.j)

    def test_framework_name(self):
        self.assertEqual(self.j["framework"], "VAAMR")

    def test_version(self):
        self.assertEqual(self.j["version"], "test")

    def test_themes_count(self):
        self.assertEqual(len(self.j["themes"]), 5)

    def test_themes_sorted_by_theme_id(self):
        ids = [t["theme_id"] for t in self.j["themes"]]
        self.assertEqual(ids, sorted(ids))

    def test_theme_entry_keys(self):
        expected_keys = {
            "theme_id", "key", "name", "short_name", "definition",
            "prototypical_features", "distinguishing_criteria",
            "exemplar_utterances", "subtle_utterances", "adversarial_utterances",
        }
        for t in self.j["themes"]:
            self.assertTrue(expected_keys.issubset(t.keys()),
                            msg=f"Missing keys in theme entry: {expected_keys - set(t.keys())}")

    def test_reverse_order_themes_still_sorted(self):
        """to_json sorts by theme_id regardless of insertion order."""
        td4 = _make_theme(4, "D4")
        td0 = _make_theme(0, "D0")
        td2 = _make_theme(2, "D2")
        fw = ThemeFramework(name="FW", version="1", description="", themes=[td4, td0, td2])
        j = fw.to_json()
        ids = [t["theme_id"] for t in j["themes"]]
        self.assertEqual(ids, [0, 2, 4])


# ---------------------------------------------------------------------------
# to_prompt_string
# ---------------------------------------------------------------------------

class TestToPromptString(unittest.TestCase):

    def _make_fw_with_all_utterances(self):
        """A framework where each theme has exemplar, subtle, and adversarial."""
        themes = [
            _make_theme(
                i, f"Theme{i}",
                exemplars=[f"Ex{i}_1", f"Ex{i}_2", f"Ex{i}_3"],
                subtle=[f"Sub{i}_1", f"Sub{i}_2"],
                adversarial=[f"Adv{i}_1", f"Adv{i}_2"],
            )
            for i in range(3)
        ]
        return ThemeFramework(name="FW", version="1", description="", themes=themes)

    def test_zero_shot_omits_examples(self):
        fw = self._make_fw_with_all_utterances()
        s = fw.to_prompt_string(zero_shot=True)
        self.assertNotIn("Examples:", s)
        self.assertNotIn("Edge cases:", s)
        self.assertNotIn("Watch-outs", s)

    def test_zero_shot_includes_definition_features_distinction(self):
        fw = self._make_fw_with_all_utterances()
        s = fw.to_prompt_string(zero_shot=True)
        self.assertIn("Prototypical features:", s)
        self.assertIn("Key distinction:", s)

    def test_non_zero_shot_includes_exemplars(self):
        fw = self._make_fw_with_all_utterances()
        s = fw.to_prompt_string(zero_shot=False)
        self.assertIn("Examples:", s)

    def test_non_zero_shot_includes_edge_cases(self):
        fw = self._make_fw_with_all_utterances()
        s = fw.to_prompt_string(zero_shot=False)
        self.assertIn("Edge cases:", s)

    def test_non_zero_shot_includes_watch_outs(self):
        fw = self._make_fw_with_all_utterances()
        s = fw.to_prompt_string(zero_shot=False)
        self.assertIn("Watch-outs", s)

    def test_n_exemplars_zero_omits_examples(self):
        fw = self._make_fw_with_all_utterances()
        s = fw.to_prompt_string(zero_shot=False, n_exemplars=0)
        self.assertNotIn("Examples:", s)

    def test_n_exemplars_cap(self):
        fw = self._make_fw_with_all_utterances()
        # Each theme has 3 exemplars; cap at 1
        s = fw.to_prompt_string(zero_shot=False, n_exemplars=1)
        # Ex0_2 should not appear (it's the 2nd exemplar for theme 0)
        self.assertNotIn("Ex0_2", s)
        self.assertIn("Ex0_1", s)

    def test_n_subtle_zero_omits_edge_cases(self):
        fw = self._make_fw_with_all_utterances()
        s = fw.to_prompt_string(zero_shot=False, n_subtle=0)
        self.assertNotIn("Edge cases:", s)

    def test_n_adversarial_zero_omits_watch_outs(self):
        fw = self._make_fw_with_all_utterances()
        s = fw.to_prompt_string(zero_shot=False, n_adversarial=0)
        self.assertNotIn("Watch-outs", s)

    def test_include_subtle_false(self):
        fw = self._make_fw_with_all_utterances()
        s = fw.to_prompt_string(zero_shot=False, include_subtle=False)
        self.assertNotIn("Edge cases:", s)

    def test_include_adversarial_false(self):
        fw = self._make_fw_with_all_utterances()
        s = fw.to_prompt_string(zero_shot=False, include_adversarial=False)
        self.assertNotIn("Watch-outs", s)

    def test_themes_separated_by_double_newline(self):
        fw = self._make_fw_with_all_utterances()
        s = fw.to_prompt_string(zero_shot=True)
        self.assertIn("\n\n", s)

    def test_randomize_does_not_lose_themes(self):
        """All theme names still present after shuffling."""
        import random
        random.seed(0)
        fw = self._make_fw_with_all_utterances()
        s = fw.to_prompt_string(randomize=True, zero_shot=True)
        for i in range(3):
            self.assertIn(f"Theme{i}", s)

    def test_empty_utterance_lists_no_section_header(self):
        """A theme with no exemplars should not emit 'Examples:' in non-zero_shot mode."""
        td = ThemeDefinition(
            theme_id=0, key="k", name="NoEx", short_name="NX", prompt_name="NoEx",
            definition="d", prototypical_features=["f"], distinguishing_criteria="c",
            exemplar_utterances=[],  # empty
            subtle_utterances=[],
            adversarial_utterances=[],
        )
        fw = ThemeFramework(name="FW", version="1", description="", themes=[td])
        s = fw.to_prompt_string(zero_shot=False)
        self.assertNotIn("Examples:", s)

    def test_prompt_name_capitalized_in_output(self):
        """to_prompt_string capitalizes the prompt_name."""
        td = _make_theme(0, "vigilance", prompt_name="vigilance theme")
        fw = ThemeFramework(name="FW", version="1", description="", themes=[td])
        s = fw.to_prompt_string(zero_shot=True)
        self.assertIn("Vigilance theme:", s)


if __name__ == "__main__":
    unittest.main()
