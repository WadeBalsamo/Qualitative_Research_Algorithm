"""
tests/test_refactor.py
-----------------------
Expanded test suite for the Markdown-Defined Frameworks refactor.

Covers:
  - Parser internals (edge cases, HTML stripping, multi-line blocks)
  - Registry (caching, None handling, unknown name errors)
  - ThemeFramework / ThemeDefinition behaviour (to_prompt_string, to_json, lookups)
  - Codebook behaviour (to_prompt_string, to_embedding_targets, domain grouping)
  - PipelineConfig (new framework fields, JSON round-trip)
  - Schema invariants (no color, no subcodes, word_prototypes populated)
  - Phase 4 wiring (registry resolves config framework names)

Run:
    PYTHONPATH=src python -m unittest src.tests.test_refactor
    pytest src/tests/test_refactor.py
"""

import dataclasses
import json
import re
import textwrap
import unittest
from pathlib import Path

# Path bootstrap is handled by src/tests/__init__.py
REPO_ROOT = Path(__file__).resolve().parents[2]
VAAMR_MD = REPO_ROOT / 'VAAMR_FRAMEWORK.md'
PURER_MD = REPO_ROOT / 'PURER_FRAMEWORK.md'
CODEBOOK_MD = REPO_ROOT / 'PHENOMENOLOGY_CODEBOOK.md'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()


# ---------------------------------------------------------------------------
# Parser internals
# ---------------------------------------------------------------------------

class TestMarkdownLoaderHelpers(unittest.TestCase):
    """Unit tests for the low-level parser helpers."""

    def setUp(self):
        from theme_framework.markdown_loader import (
            _strip_html_comments,
            _parse_bullets,
            _parse_blockquotes,
            _parse_word_prototypes,
            _normalize_prose,
        )
        self.strip = _strip_html_comments
        self.bullets = _parse_bullets
        self.blockquotes = _parse_blockquotes
        self.word_protos = _parse_word_prototypes
        self.normalize = _normalize_prose

    # --- HTML comment stripping ---

    def test_strip_single_line_comment(self):
        result = self.strip("before <!-- comment --> after")
        self.assertNotIn('comment', result)
        self.assertIn('before', result)
        self.assertIn('after', result)

    def test_strip_multiline_comment(self):
        text = "a\n<!-- line1\nline2\nline3 -->\nb"
        result = self.strip(text)
        self.assertNotIn('line1', result)
        self.assertNotIn('line2', result)
        self.assertIn('a', result)
        self.assertIn('b', result)

    def test_strip_multiple_comments(self):
        text = "<!-- A -->\ntext\n<!-- B -->\nmore"
        result = self.strip(text)
        self.assertNotIn('A', result)
        self.assertNotIn('B', result)
        self.assertIn('text', result)
        self.assertIn('more', result)

    def test_strip_comment_with_gt_inside(self):
        """HTML comments containing '>' chars should still be stripped."""
        text = "<!-- if x > 0 -->\nresult"
        result = self.strip(text)
        self.assertNotIn('if x', result)
        self.assertIn('result', result)

    # --- Bullet parsing ---

    def test_bullets_basic(self):
        text = "- item one\n- item two\n- item three"
        self.assertEqual(self.bullets(text), ['item one', 'item two', 'item three'])

    def test_bullets_skips_non_bullet_lines(self):
        text = "header\n- item\nfooter"
        self.assertEqual(self.bullets(text), ['item'])

    def test_bullets_empty(self):
        self.assertEqual(self.bullets(""), [])

    def test_bullets_strips_dash_prefix(self):
        text = "  - spaced item"
        result = self.bullets(text)
        self.assertEqual(result, ['spaced item'])

    # --- Blockquote parsing ---

    def test_blockquotes_single(self):
        text = "> single utterance"
        self.assertEqual(self.blockquotes(text), ["single utterance"])

    def test_blockquotes_multiple_separated(self):
        text = "> first\n\n> second\n\n> third"
        self.assertEqual(self.blockquotes(text), ["first", "second", "third"])

    def test_blockquotes_multi_line_joined(self):
        text = "> line one\n> line two"
        self.assertEqual(self.blockquotes(text), ["line one line two"])

    def test_blockquotes_skips_non_quote_lines(self):
        text = "prefix\n> utterance\npostfix"
        self.assertEqual(self.blockquotes(text), ["utterance"])

    def test_blockquotes_filters_empty(self):
        self.assertEqual(self.blockquotes(""), [])

    def test_blockquotes_after_comment_stripped(self):
        """After HTML comment stripping, blockquotes should parse cleanly."""
        from theme_framework.markdown_loader import _strip_html_comments
        text = "<!-- note -->\n> actual utterance"
        clean = _strip_html_comments(text)
        result = self.blockquotes(clean)
        self.assertEqual(result, ["actual utterance"])

    # --- Word prototype parsing ---

    def test_word_protos_basic(self):
        text = "alpha, beta, gamma"
        self.assertEqual(self.word_protos(text), ["alpha", "beta", "gamma"])

    def test_word_protos_ignores_trailing_hr(self):
        text = "word one, word two\n\n---\n"
        result = self.word_protos(text)
        self.assertEqual(result, ["word one", "word two"])

    def test_word_protos_empty(self):
        self.assertEqual(self.word_protos(""), [])

    def test_word_protos_multi_word_entries(self):
        text = "mind wanders, can't focus, butterfly effect"
        result = self.word_protos(text)
        self.assertIn("mind wanders", result)
        self.assertIn("can't focus", result)
        self.assertIn("butterfly effect", result)

    # --- Prose normalization ---

    def test_normalize_collapses_whitespace(self):
        self.assertEqual(self.normalize("a  b   c"), "a b c")

    def test_normalize_strips_newlines(self):
        self.assertEqual(self.normalize("a\nb\nc"), "a b c")

    def test_normalize_strips_horizontal_rule(self):
        self.assertEqual(self.normalize("text\n\n---\n"), "text")

    def test_normalize_strips_leading_trailing(self):
        self.assertEqual(self.normalize("  hello  "), "hello")


# ---------------------------------------------------------------------------
# Parser: load_framework_md (integration with real .md files)
# ---------------------------------------------------------------------------

class TestFrameworkMarkdownParserIntegration(unittest.TestCase):
    """Integration tests: parse real .md files and validate output structure."""

    @classmethod
    def setUpClass(cls):
        from theme_framework.markdown_loader import load_framework_md
        cls.vaamr = load_framework_md(VAAMR_MD)
        cls.purer = load_framework_md(PURER_MD)

    def test_vaamr_theme_ids_are_0_to_4(self):
        ids = [t.theme_id for t in self.vaamr.themes]
        self.assertEqual(sorted(ids), [0, 1, 2, 3, 4])

    def test_purer_theme_ids_are_0_to_4(self):
        ids = [t.theme_id for t in self.purer.themes]
        self.assertEqual(sorted(ids), [0, 1, 2, 3, 4])

    def test_every_vaamr_theme_has_definition(self):
        for t in self.vaamr.themes:
            with self.subTest(key=t.key):
                self.assertTrue(len(t.definition) > 50,
                                f"{t.key} definition too short: {t.definition[:40]!r}")

    def test_every_vaamr_theme_has_exemplars(self):
        for t in self.vaamr.themes:
            with self.subTest(key=t.key):
                self.assertGreater(len(t.exemplar_utterances), 0)

    def test_every_vaamr_theme_has_word_prototypes(self):
        for t in self.vaamr.themes:
            with self.subTest(key=t.key):
                self.assertGreater(len(t.word_prototypes), 0)

    def test_every_vaamr_theme_has_adversarial_utterances(self):
        for t in self.vaamr.themes:
            with self.subTest(key=t.key):
                self.assertGreater(len(t.adversarial_utterances), 0)

    def test_every_purer_theme_has_definition(self):
        for t in self.purer.themes:
            with self.subTest(key=t.key):
                self.assertTrue(len(t.definition) > 50)

    def test_no_utterance_contains_html_comment_remnant(self):
        """HTML comments must be fully stripped — no '<!--' or '-->' in output."""
        for fw in (self.vaamr, self.purer):
            for t in fw.themes:
                for field_name in ('exemplar_utterances', 'subtle_utterances', 'adversarial_utterances'):
                    for u in getattr(t, field_name):
                        with self.subTest(fw=fw.name, key=t.key, field=field_name):
                            self.assertNotIn('<!--', u)
                            self.assertNotIn('-->', u)

    def test_no_definition_contains_html_comment_remnant(self):
        for fw in (self.vaamr, self.purer):
            for t in fw.themes:
                with self.subTest(fw=fw.name, key=t.key):
                    self.assertNotIn('<!--', t.definition)
                    self.assertNotIn('<!--', t.distinguishing_criteria)

    def test_vaamr_categories_present(self):
        self.assertIsNotNone(self.vaamr.categories)
        self.assertIn('AttentionDysregulation', self.vaamr.categories)

    def test_purer_categories_absent(self):
        self.assertIsNone(self.purer.categories)

    def test_keys_are_unique_per_framework(self):
        for fw in (self.vaamr, self.purer):
            keys = [t.key for t in fw.themes]
            self.assertEqual(len(keys), len(set(keys)), f"{fw.name} has duplicate keys")

    def test_aliases_are_lists(self):
        for fw in (self.vaamr, self.purer):
            for t in fw.themes:
                with self.subTest(fw=fw.name, key=t.key):
                    self.assertIsInstance(t.aliases, list)

    def test_vaamr_reappraisal_has_metacognition_boundary_exemplar(self):
        """Canonical load-bearing exemplar must parse correctly."""
        reap = next(t for t in self.vaamr.themes if t.key == 'reappraisal')
        combined = ' '.join(reap.exemplar_utterances)
        self.assertIn("noticing the way that I'm noticing pain", combined)

    def test_purer_reinforcement_has_co_occurrence_annotation(self):
        """(also reframes) annotations must survive from markdown to parsed data."""
        reinforce = next(t for t in self.purer.themes if t.key == 'R2')
        combined = ' '.join(reinforce.exemplar_utterances)
        self.assertIn('(also reframes)', combined)


# ---------------------------------------------------------------------------
# Parser: load_codebook_md (integration)
# ---------------------------------------------------------------------------

class TestCodebookMarkdownParserIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from codebook.markdown_loader import load_codebook_md
        cls.cb = load_codebook_md(CODEBOOK_MD)

    def test_code_count(self):
        self.assertGreaterEqual(len(self.cb.codes), 50)

    def test_all_codes_have_code_id(self):
        for c in self.cb.codes:
            with self.subTest(category=c.category):
                self.assertTrue(c.code_id, "code_id must not be empty")

    def test_all_codes_have_description(self):
        for c in self.cb.codes:
            with self.subTest(code_id=c.code_id):
                self.assertTrue(len(c.description) > 20)

    def test_all_codes_have_inclusive_criteria(self):
        for c in self.cb.codes:
            with self.subTest(code_id=c.code_id):
                self.assertTrue(len(c.inclusive_criteria) > 10)

    def test_all_codes_have_exclusive_criteria(self):
        for c in self.cb.codes:
            with self.subTest(code_id=c.code_id):
                self.assertTrue(len(c.exclusive_criteria) > 10)

    def test_domain_names_match_expected(self):
        expected = {'Affective', 'Cognitive', 'Perceptual', 'Sense of Self', 'Social', 'Somatic'}
        actual = set(self.cb.domain_names)
        self.assertEqual(actual, expected)

    def test_domains_dict_covers_all_codes(self):
        all_ids_in_domains = {cid for ids in self.cb.domains.values() for cid in ids}
        all_ids = {c.code_id for c in self.cb.codes}
        self.assertEqual(all_ids_in_domains, all_ids)

    def test_no_html_comment_remnants_in_criteria(self):
        for c in self.cb.codes:
            with self.subTest(code_id=c.code_id):
                self.assertNotIn('<!--', c.inclusive_criteria)
                self.assertNotIn('<!--', c.exclusive_criteria)

    def test_exclusive_criteria_no_trailing_hr(self):
        """'---' separators between code blocks must not leak into criteria text."""
        for c in self.cb.codes:
            with self.subTest(code_id=c.code_id):
                self.assertNotIn('---', c.exclusive_criteria)


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------

class TestSchemaInvariants(unittest.TestCase):

    def test_theme_definition_has_no_color(self):
        from theme_framework.theme_schema import ThemeDefinition
        names = {f.name for f in dataclasses.fields(ThemeDefinition)}
        self.assertNotIn('color', names)

    def test_code_definition_has_no_subcodes(self):
        from codebook.codebook_schema import CodeDefinition
        names = {f.name for f in dataclasses.fields(CodeDefinition)}
        self.assertNotIn('subcodes', names)

    def test_theme_definition_required_fields_present(self):
        from theme_framework.theme_schema import ThemeDefinition
        required = {
            'theme_id', 'key', 'name', 'short_name', 'prompt_name',
            'definition', 'prototypical_features', 'distinguishing_criteria',
            'exemplar_utterances', 'subtle_utterances', 'adversarial_utterances',
            'word_prototypes', 'aliases',
        }
        actual = {f.name for f in dataclasses.fields(ThemeDefinition)}
        self.assertTrue(required.issubset(actual), f"Missing: {required - actual}")

    def test_code_definition_required_fields_present(self):
        from codebook.codebook_schema import CodeDefinition
        required = {'code_id', 'category', 'domain', 'description',
                    'inclusive_criteria', 'exclusive_criteria', 'exemplar_utterances'}
        actual = {f.name for f in dataclasses.fields(CodeDefinition)}
        self.assertTrue(required.issubset(actual), f"Missing: {required - actual}")


# ---------------------------------------------------------------------------
# ThemeFramework methods
# ---------------------------------------------------------------------------

class TestThemeFrameworkMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from theme_framework.registry import load
        cls.vaamr = load('vaamr')
        cls.purer = load('purer')

    def test_get_theme_by_id_returns_correct_theme(self):
        t = self.vaamr.get_theme_by_id(0)
        self.assertIsNotNone(t)
        self.assertEqual(t.key, 'vigilance')

    def test_get_theme_by_id_returns_none_for_unknown(self):
        self.assertIsNone(self.vaamr.get_theme_by_id(99))

    def test_build_name_to_id_map_contains_all_keys(self):
        mapping = self.vaamr.build_name_to_id_map()
        for t in self.vaamr.themes:
            self.assertIn(t.key.lower(), mapping)
            self.assertIn(t.short_name.lower(), mapping)
            self.assertIn(t.prompt_name.lower(), mapping)

    def test_build_name_to_id_map_aliases_resolve(self):
        mapping = self.vaamr.build_name_to_id_map()
        for t in self.vaamr.themes:
            for alias in t.aliases:
                self.assertIn(alias.lower(), mapping,
                              f"Alias {alias!r} missing for theme {t.key}")

    def test_build_id_to_short_map(self):
        mapping = self.vaamr.build_id_to_short_map()
        self.assertEqual(mapping[0], 'Vigilance')
        self.assertEqual(mapping[4], 'Reappraisal')

    def test_to_json_structure(self):
        j = self.vaamr.to_json()
        self.assertEqual(j['framework'], 'VAAMR')
        self.assertEqual(j['version'], '4.0')
        self.assertIsInstance(j['themes'], list)
        self.assertEqual(len(j['themes']), 5)
        first = j['themes'][0]
        for key in ('theme_id', 'key', 'name', 'definition',
                    'prototypical_features', 'exemplar_utterances'):
            self.assertIn(key, first)
        self.assertNotIn('color', first)

    def test_to_json_themes_sorted_by_id(self):
        j = self.vaamr.to_json()
        ids = [t['theme_id'] for t in j['themes']]
        self.assertEqual(ids, sorted(ids))

    def test_to_prompt_string_contains_all_themes(self):
        prompt = self.vaamr.to_prompt_string()
        for t in self.vaamr.themes:
            self.assertIn(t.prompt_name, prompt.lower() + prompt)

    def test_to_prompt_string_zero_shot_excludes_examples(self):
        prompt_full = self.vaamr.to_prompt_string(zero_shot=False)
        prompt_zero = self.vaamr.to_prompt_string(zero_shot=True)
        self.assertGreater(len(prompt_full), len(prompt_zero))
        # Exemplar content present in full but not zero-shot
        first_exemplar = self.vaamr.themes[0].exemplar_utterances[0][:30]
        self.assertIn(first_exemplar, prompt_full)
        self.assertNotIn(first_exemplar, prompt_zero)

    def test_to_prompt_string_n_exemplars_limits_count(self):
        prompt = self.vaamr.to_prompt_string(n_exemplars=1, include_subtle=False,
                                              include_adversarial=False)
        # A theme with 12 exemplars should only show 1
        vigilance = self.vaamr.themes[0]
        count = sum(1 for u in vigilance.exemplar_utterances
                    if u[:20] in prompt)
        self.assertLessEqual(count, 1)

    def test_stages_property_alias(self):
        self.assertEqual(self.vaamr.stages, self.vaamr.themes)

    def test_num_themes(self):
        self.assertEqual(self.vaamr.num_themes, 5)
        self.assertEqual(self.purer.num_themes, 5)

    def test_purer_name_to_id_includes_move_keys(self):
        mapping = self.purer.build_name_to_id_map()
        self.assertIn('p', mapping)
        self.assertIn('u', mapping)
        self.assertIn('r', mapping)
        self.assertIn('e', mapping)
        self.assertIn('r2', mapping)


# ---------------------------------------------------------------------------
# Codebook methods
# ---------------------------------------------------------------------------

class TestCodebookMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from codebook.phenomenology_codebook import get_phenomenology_codebook
        cls.cb = get_phenomenology_codebook()

    def test_get_codes_by_domain(self):
        affective = self.cb.get_codes_by_domain('Affective')
        self.assertGreater(len(affective), 0)
        for c in affective:
            self.assertEqual(c.domain, 'Affective')

    def test_build_name_to_id_map(self):
        mapping = self.cb.build_name_to_id_map()
        for c in self.cb.codes:
            self.assertIn(c.code_id.lower(), mapping)
            self.assertIn(c.category.lower(), mapping)

    def test_domain_names_sorted(self):
        names = self.cb.domain_names
        self.assertEqual(names, sorted(names))

    def test_to_prompt_string_no_subcodes_label(self):
        prompt = self.cb.to_prompt_string()
        self.assertNotIn('Subcodes:', prompt)

    def test_to_prompt_string_contains_include_exclude(self):
        prompt = self.cb.to_prompt_string()
        self.assertIn('Include when:', prompt)
        self.assertIn('Exclude when:', prompt)

    def test_to_prompt_string_contains_all_categories(self):
        prompt = self.cb.to_prompt_string()
        for c in self.cb.codes:
            self.assertIn(c.category, prompt,
                          f"Category {c.category!r} missing from prompt string")

    def test_to_embedding_targets_structure(self):
        targets = self.cb.to_embedding_targets()
        self.assertEqual(len(targets), len(self.cb.codes))
        for t in targets:
            for key in ('code_id', 'category', 'definition', 'criteria', 'exemplars'):
                self.assertIn(key, t)
            self.assertNotIn('subcodes', t)

    def test_to_embedding_targets_definition_contains_category(self):
        targets = self.cb.to_embedding_targets()
        for t in targets:
            self.assertIn(t['category'], t['definition'])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestFrameworkRegistry(unittest.TestCase):

    def test_load_vaamr_returns_framework(self):
        from theme_framework.registry import load
        fw = load('vaamr')
        self.assertEqual(fw.name, 'VAAMR')

    def test_load_purer_returns_framework(self):
        from theme_framework.registry import load
        fw = load('purer')
        self.assertEqual(fw.name, 'PURER')

    def test_load_none_returns_none(self):
        from theme_framework.registry import load
        self.assertIsNone(load(None))

    def test_load_unknown_raises_key_error(self):
        from theme_framework.registry import load
        with self.assertRaises(KeyError):
            load('not_a_framework')

    def test_repeated_load_returns_same_object(self):
        """lru_cache must ensure the same ThemeFramework object is returned."""
        from theme_framework.registry import load
        fw1 = load('vaamr')
        fw2 = load('vaamr')
        self.assertIs(fw1, fw2)

    def test_frameworks_dict_contains_expected_keys(self):
        from theme_framework.registry import FRAMEWORKS
        self.assertIn('vaamr', FRAMEWORKS)
        self.assertIn('purer', FRAMEWORKS)

    def test_framework_paths_exist(self):
        from theme_framework.registry import FRAMEWORKS
        for name, path in FRAMEWORKS.items():
            with self.subTest(name=name):
                self.assertTrue(path.exists(), f"{name}: {path} does not exist")


# ---------------------------------------------------------------------------
# PipelineConfig — Phase 4 fields
# ---------------------------------------------------------------------------

class TestPipelineConfigFrameworkFields(unittest.TestCase):

    def test_participant_framework_field_exists(self):
        from process.config import PipelineConfig
        names = {f.name for f in dataclasses.fields(PipelineConfig)}
        self.assertIn('participant_framework', names)

    def test_therapist_framework_field_exists(self):
        from process.config import PipelineConfig
        names = {f.name for f in dataclasses.fields(PipelineConfig)}
        self.assertIn('therapist_framework', names)

    def test_defaults(self):
        from process.config import PipelineConfig
        cfg = PipelineConfig()
        self.assertEqual(cfg.participant_framework, 'vaamr')
        self.assertEqual(cfg.therapist_framework, 'purer')

    def test_therapist_framework_accepts_none(self):
        from process.config import PipelineConfig
        cfg = PipelineConfig(therapist_framework=None)
        self.assertIsNone(cfg.therapist_framework)

    def test_to_json_includes_framework_fields(self):
        from process.config import PipelineConfig
        cfg = PipelineConfig()
        j = cfg.to_json()
        self.assertIn('participant_framework', j)
        self.assertIn('therapist_framework', j)
        self.assertEqual(j['participant_framework'], 'vaamr')
        self.assertEqual(j['therapist_framework'], 'purer')

    def test_to_json_round_trip_framework_fields(self):
        """Fields survive JSON serialisation and can be read back."""
        from process.config import PipelineConfig
        cfg = PipelineConfig(participant_framework='vaamr', therapist_framework=None)
        j = cfg.to_json()
        self.assertEqual(j['participant_framework'], 'vaamr')
        self.assertIsNone(j['therapist_framework'])

    def test_registry_resolves_participant_framework(self):
        """Registry must be able to load whatever participant_framework is set to."""
        from process.config import PipelineConfig
        from theme_framework.registry import load
        cfg = PipelineConfig()
        fw = load(cfg.participant_framework)
        self.assertIsNotNone(fw)
        self.assertEqual(fw.name, 'VAAMR')

    def test_registry_resolves_therapist_framework(self):
        from process.config import PipelineConfig
        from theme_framework.registry import load
        cfg = PipelineConfig()
        fw = load(cfg.therapist_framework)
        self.assertIsNotNone(fw)
        self.assertEqual(fw.name, 'PURER')

    def test_registry_handles_none_therapist_framework(self):
        from process.config import PipelineConfig
        from theme_framework.registry import load
        cfg = PipelineConfig(therapist_framework=None)
        self.assertIsNone(load(cfg.therapist_framework))


# ---------------------------------------------------------------------------
# VAAMR-specific content invariants
# ---------------------------------------------------------------------------

class TestVAAMRContentInvariants(unittest.TestCase):
    """Spot-check that key clinical distinctions survived the markdown round-trip."""

    @classmethod
    def setUpClass(cls):
        from theme_framework.registry import load
        cls.fw = load('vaamr')
        cls.by_key = {t.key: t for t in cls.fw.themes}

    def test_vigilance_is_theme_id_0(self):
        self.assertEqual(self.by_key['vigilance'].theme_id, 0)

    def test_reappraisal_is_highest_id(self):
        max_id = max(t.theme_id for t in self.fw.themes)
        self.assertEqual(self.by_key['reappraisal'].theme_id, max_id)

    def test_avoidance_definition_mentions_escape(self):
        defn = self.by_key['avoidance'].definition.lower()
        self.assertIn('escape', defn)

    def test_attention_regulation_definition_mentions_volitional(self):
        defn = self.by_key['attention'].definition.lower()
        self.assertIn('volitional', defn)

    def test_metacognition_distinguishing_criteria_mentions_reappraisal_boundary(self):
        criteria = self.by_key['metacognition'].distinguishing_criteria.lower()
        self.assertIn('reappraisal', criteria)

    def test_reappraisal_exemplars_include_sensory_decomposition(self):
        exemplars = ' '.join(self.by_key['reappraisal'].exemplar_utterances).lower()
        self.assertIn('jagged', exemplars)

    def test_vigilance_features_count(self):
        self.assertGreaterEqual(len(self.by_key['vigilance'].prototypical_features), 10)

    def test_avoidance_aliases_include_kinesiophobic(self):
        aliases_lower = [a.lower() for a in self.by_key['avoidance'].aliases]
        self.assertIn('kinesiophobic avoidance', aliases_lower)

    def test_metacognition_word_prototypes_include_noticing(self):
        self.assertIn('noticing', self.by_key['metacognition'].word_prototypes)

    def test_categories_attentiondysregulation_contains_vigilance_and_avoidance(self):
        cats = self.fw.categories
        self.assertIn('Vigilance', cats['AttentionDysregulation'])
        self.assertIn('Avoidance', cats['AttentionDysregulation'])


# ---------------------------------------------------------------------------
# PURER-specific content invariants
# ---------------------------------------------------------------------------

class TestPURERContentInvariants(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from theme_framework.registry import load
        cls.fw = load('purer')
        cls.by_key = {t.key: t for t in cls.fw.themes}

    def test_phenomenological_key_is_P(self):
        self.assertIn('P', self.by_key)

    def test_reinforcement_key_is_R2(self):
        self.assertIn('R2', self.by_key)

    def test_phenomenological_definition_mentions_breakdown(self):
        defn = self.by_key['P'].definition.lower()
        # "breakdown" or "decomposable" captures the phenomenological inquiry move
        self.assertTrue('breakdown' in defn or 'decomposable' in defn)

    def test_utilization_distinguishing_criteria_mentions_future(self):
        criteria = self.by_key['U'].distinguishing_criteria.lower()
        self.assertIn('future', criteria)

    def test_reinforcement_exemplar_with_reframes_annotation(self):
        exemplars = ' '.join(self.by_key['R2'].exemplar_utterances)
        self.assertIn('(also reframes)', exemplars)

    def test_reframing_exemplar_with_reinforces_annotation(self):
        exemplars = ' '.join(self.by_key['R'].exemplar_utterances)
        self.assertIn('(also reinforces)', exemplars)

    def test_all_purer_themes_have_word_prototypes(self):
        for t in self.fw.themes:
            with self.subTest(key=t.key):
                self.assertGreater(len(t.word_prototypes), 0)

    def test_education_word_prototypes_include_brain(self):
        self.assertIn('brain', self.by_key['E'].word_prototypes)


# ---------------------------------------------------------------------------
# Parser synthetic edge-case tests (in-memory markdown)
# ---------------------------------------------------------------------------

class TestParserSyntheticMarkdown(unittest.TestCase):
    """Parse minimal hand-written markdown snippets to test edge cases."""

    def _make_minimal_framework_md(self, framework_name: str, heading_style: str) -> str:
        """Return a minimal but valid framework markdown with one theme."""
        if heading_style == 'stage':
            theme_heading = '## Stage 0 — TestTheme'
        else:
            theme_heading = '## Move 0 — T — TestTheme'

        return textwrap.dedent(f"""\
            ---
            framework: {framework_name}
            version: "1.0"
            framework_description: >-
              Test framework description.
            ---

            {theme_heading}

            ```yaml
            theme_id: 0
            key: test
            name: Test Theme
            short_name: TestTheme
            prompt_name: test theme
            aliases:
              - alias_one
            ```

            ### Definition

            This is the definition of the test theme.

            ### Prototypical Features

            - feature one
            - feature two

            ### Distinguishing Criteria

            Distinguished from nothing.

            ### Exemplar Utterances

            > First exemplar utterance.

            > Second exemplar utterance.

            ### Subtle Utterances

            > A subtle utterance.

            ### Adversarial Utterances

            <!-- note about this adversarial case -->
            > Adversarial utterance one.

            ### Word Prototypes

            word one, word two, word three
        """)

    def _parse(self, text: str, path_suffix: str = 'test.md'):
        import tempfile
        from theme_framework.markdown_loader import load_framework_md
        with tempfile.NamedTemporaryFile(mode='w', suffix=path_suffix,
                                         delete=False, encoding='utf-8') as f:
            f.write(text)
            tmp = Path(f.name)
        try:
            return load_framework_md(tmp)
        finally:
            tmp.unlink()

    def test_vaamr_style_parses_one_theme(self):
        md = self._make_minimal_framework_md('VAAMR', 'stage')
        fw = self._parse(md)
        self.assertEqual(len(fw.themes), 1)
        self.assertEqual(fw.themes[0].key, 'test')

    def test_purer_style_parses_one_theme(self):
        md = self._make_minimal_framework_md('PURER', 'move')
        fw = self._parse(md)
        self.assertEqual(len(fw.themes), 1)
        self.assertEqual(fw.themes[0].short_name, 'TestTheme')

    def test_exemplars_parsed_correctly(self):
        md = self._make_minimal_framework_md('VAAMR', 'stage')
        fw = self._parse(md)
        t = fw.themes[0]
        self.assertEqual(len(t.exemplar_utterances), 2)
        self.assertIn('First exemplar', t.exemplar_utterances[0])

    def test_adversarial_comment_stripped(self):
        md = self._make_minimal_framework_md('VAAMR', 'stage')
        fw = self._parse(md)
        adv = fw.themes[0].adversarial_utterances
        self.assertEqual(len(adv), 1)
        self.assertNotIn('<!--', adv[0])
        self.assertIn('Adversarial utterance one', adv[0])

    def test_word_prototypes_parsed(self):
        md = self._make_minimal_framework_md('VAAMR', 'stage')
        fw = self._parse(md)
        self.assertEqual(fw.themes[0].word_prototypes,
                         ['word one', 'word two', 'word three'])

    def test_aliases_parsed(self):
        md = self._make_minimal_framework_md('VAAMR', 'stage')
        fw = self._parse(md)
        self.assertEqual(fw.themes[0].aliases, ['alias_one'])

    def test_no_color_field_in_parsed_theme(self):
        md = self._make_minimal_framework_md('VAAMR', 'stage')
        fw = self._parse(md)
        self.assertFalse(hasattr(fw.themes[0], 'color'))

    def test_framework_with_no_frontmatter_raises(self):
        from theme_framework.markdown_loader import load_framework_md
        import tempfile
        text = "# Just a heading\nNo frontmatter here."
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md',
                                         delete=False, encoding='utf-8') as f:
            f.write(text)
            tmp = Path(f.name)
        try:
            with self.assertRaises(ValueError):
                load_framework_md(tmp)
        finally:
            tmp.unlink()

    def test_multiline_blockquote_joined(self):
        """Consecutive '> ' lines should be joined into one utterance."""
        md = textwrap.dedent("""\
            ---
            framework: VAAMR
            version: "1.0"
            framework_description: >-
              desc
            ---
            ## Stage 0 — X
            ```yaml
            theme_id: 0
            key: x
            name: X
            short_name: X
            prompt_name: x
            ```
            ### Definition
            Def.
            ### Prototypical Features
            - feat
            ### Distinguishing Criteria
            Crit.
            ### Exemplar Utterances
            > Line one
            > line two continued
            ### Subtle Utterances
            ### Adversarial Utterances
            ### Word Prototypes
            w
        """)
        fw = self._parse(md)
        exemplars = fw.themes[0].exemplar_utterances
        self.assertEqual(len(exemplars), 1)
        self.assertIn('Line one', exemplars[0])
        self.assertIn('line two continued', exemplars[0])


if __name__ == '__main__':
    unittest.main()
