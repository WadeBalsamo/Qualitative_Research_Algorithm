"""
tests/test_markdown_loader.py
------------------------------
TDD parity tests for the markdown framework loaders (Phase 1 of the
frameworks refactor).

These tests FAIL until constructs/markdown_loader.py is implemented.
Parity gate: each loader must produce output deep-equal to the existing
Python factory, except for the `color` field (dropped in Phase 2).

Run:
    python -m unittest tests.test_markdown_loader
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # tests/unit/ → tests/ → repo root
VAAMR_MD = REPO_ROOT / 'frameworks' / 'VAAMR_FRAMEWORK.md'
PURER_MD = REPO_ROOT / 'frameworks' / 'PURER_FRAMEWORK.md'


def _normalize(s: str) -> str:
    """Collapse internal whitespace; strip leading/trailing. Used for prose comparisons."""
    return re.sub(r'\s+', ' ', s).strip()


def _theme_fields(t, *, exclude=('color',)):
    """Return a dict of ThemeDefinition fields, excluding visualization-only ones."""
    return {
        k: v for k, v in vars(t).items() if k not in exclude
    }


# ---------------------------------------------------------------------------
# VAAMR
# ---------------------------------------------------------------------------

class TestVAAMRMarkdownLoader(unittest.TestCase):
    """Parse VAAMR.md and assert structural + content parity with get_vaamr_framework()."""

    @classmethod
    def setUpClass(cls):
        from constructs.markdown_loader import load_framework_md
        from constructs.vaamr import get_vaamr_framework
        cls.md_fw = load_framework_md(VAAMR_MD)
        cls.py_fw = get_vaamr_framework()

    # --- ThemeFramework scalars ---

    def test_framework_name(self):
        self.assertEqual(self.md_fw.name, self.py_fw.name)

    def test_framework_version(self):
        self.assertEqual(self.md_fw.version, self.py_fw.version)

    def test_framework_description_normalized(self):
        self.assertEqual(
            _normalize(self.md_fw.description),
            _normalize(self.py_fw.description),
        )

    def test_categories(self):
        self.assertEqual(self.md_fw.categories, self.py_fw.categories)

    def test_theme_count(self):
        self.assertEqual(len(self.md_fw.themes), len(self.py_fw.themes))

    # --- Per-theme identity scalars (exact) ---

    def test_theme_ids(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(md_t.theme_id, py_t.theme_id)

    def test_theme_keys(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(md_t.key, py_t.key)

    def test_theme_names(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(md_t.name, py_t.name)

    def test_short_names(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(md_t.short_name, py_t.short_name)

    def test_prompt_names(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(md_t.prompt_name, py_t.prompt_name)

    def test_aliases(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(md_t.aliases, py_t.aliases)

    # --- Per-theme prose (normalized) ---

    def test_definitions(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(_normalize(md_t.definition), _normalize(py_t.definition))

    def test_distinguishing_criteria(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(
                    _normalize(md_t.distinguishing_criteria),
                    _normalize(py_t.distinguishing_criteria),
                )

    # --- Per-theme lists (exact) ---

    def test_prototypical_features(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(md_t.prototypical_features, py_t.prototypical_features)

    def test_exemplar_utterance_count(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(
                    len(md_t.exemplar_utterances),
                    len(py_t.exemplar_utterances),
                )

    def test_exemplar_utterances(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            for i, (md_u, py_u) in enumerate(
                zip(md_t.exemplar_utterances, py_t.exemplar_utterances)
            ):
                with self.subTest(theme_id=py_t.theme_id, idx=i):
                    self.assertEqual(md_u, py_u)

    def test_subtle_utterance_count(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(
                    len(md_t.subtle_utterances),
                    len(py_t.subtle_utterances),
                )

    def test_subtle_utterances(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            for i, (md_u, py_u) in enumerate(
                zip(md_t.subtle_utterances, py_t.subtle_utterances)
            ):
                with self.subTest(theme_id=py_t.theme_id, idx=i):
                    self.assertEqual(md_u, py_u)

    def test_adversarial_utterance_count(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(
                    len(md_t.adversarial_utterances),
                    len(py_t.adversarial_utterances),
                )

    def test_adversarial_utterances(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            for i, (md_u, py_u) in enumerate(
                zip(md_t.adversarial_utterances, py_t.adversarial_utterances)
            ):
                with self.subTest(theme_id=py_t.theme_id, idx=i):
                    self.assertEqual(md_u, py_u)

    def test_word_prototypes(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(md_t.word_prototypes, py_t.word_prototypes)

    # --- Phase 2: color removed from dataclass ---

    def test_color_field_absent(self):
        import dataclasses
        from constructs.theme_schema import ThemeDefinition
        field_names = {f.name for f in dataclasses.fields(ThemeDefinition)}
        self.assertNotIn('color', field_names)


# ---------------------------------------------------------------------------
# PURER
# ---------------------------------------------------------------------------

class TestPURERMarkdownLoader(unittest.TestCase):
    """Parse PURER.md and assert structural + content parity with get_purer_framework()."""

    @classmethod
    def setUpClass(cls):
        from constructs.markdown_loader import load_framework_md
        from constructs.purer import get_purer_framework
        cls.md_fw = load_framework_md(PURER_MD)
        cls.py_fw = get_purer_framework()

    def test_framework_name(self):
        self.assertEqual(self.md_fw.name, self.py_fw.name)

    def test_framework_version(self):
        self.assertEqual(self.md_fw.version, self.py_fw.version)

    def test_framework_description_normalized(self):
        self.assertEqual(
            _normalize(self.md_fw.description),
            _normalize(self.py_fw.description),
        )

    def test_no_categories(self):
        self.assertIsNone(self.md_fw.categories)

    def test_theme_count(self):
        self.assertEqual(len(self.md_fw.themes), len(self.py_fw.themes))

    def test_theme_ids(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(md_t.theme_id, py_t.theme_id)

    def test_theme_keys(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(md_t.key, py_t.key)

    def test_theme_names(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(md_t.name, py_t.name)

    def test_short_names(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(md_t.short_name, py_t.short_name)

    def test_prompt_names(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(md_t.prompt_name, py_t.prompt_name)

    def test_aliases(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(md_t.aliases, py_t.aliases)

    def test_definitions(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(_normalize(md_t.definition), _normalize(py_t.definition))

    def test_distinguishing_criteria(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(
                    _normalize(md_t.distinguishing_criteria),
                    _normalize(py_t.distinguishing_criteria),
                )

    def test_prototypical_features(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(md_t.prototypical_features, py_t.prototypical_features)

    def test_exemplar_utterance_count(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(
                    len(md_t.exemplar_utterances),
                    len(py_t.exemplar_utterances),
                )

    def test_exemplar_utterances(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            for i, (md_u, py_u) in enumerate(
                zip(md_t.exemplar_utterances, py_t.exemplar_utterances)
            ):
                with self.subTest(theme_id=py_t.theme_id, idx=i):
                    self.assertEqual(md_u, py_u)

    def test_subtle_utterances(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            for i, (md_u, py_u) in enumerate(
                zip(md_t.subtle_utterances, py_t.subtle_utterances)
            ):
                with self.subTest(theme_id=py_t.theme_id, idx=i):
                    self.assertEqual(md_u, py_u)

    def test_adversarial_utterances(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            for i, (md_u, py_u) in enumerate(
                zip(md_t.adversarial_utterances, py_t.adversarial_utterances)
            ):
                with self.subTest(theme_id=py_t.theme_id, idx=i):
                    self.assertEqual(md_u, py_u)

    def test_word_prototypes(self):
        for md_t, py_t in zip(self.md_fw.themes, self.py_fw.themes):
            with self.subTest(theme_id=py_t.theme_id):
                self.assertEqual(md_t.word_prototypes, py_t.word_prototypes)

    def test_color_field_absent(self):
        import dataclasses
        from constructs.theme_schema import ThemeDefinition
        field_names = {f.name for f in dataclasses.fields(ThemeDefinition)}
        self.assertNotIn('color', field_names)


# ---------------------------------------------------------------------------
# Phase 2 — drop dead fields
# ---------------------------------------------------------------------------

CODEBOOK_MD = REPO_ROOT / 'frameworks' / 'PHENOMENOLOGY_CODEBOOK.md'


class TestCodebookMarkdownLoader(unittest.TestCase):
    """Parse CODEBOOK.md and assert structural + content parity with get_phenomenology_codebook()."""

    @classmethod
    def setUpClass(cls):
        from constructs.codebook.markdown_loader import load_codebook_md
        from constructs.codebook.phenomenology_codebook import get_phenomenology_codebook
        cls.md_cb = load_codebook_md(CODEBOOK_MD)
        cls.py_cb = get_phenomenology_codebook()

    def test_codebook_name(self):
        self.assertEqual(self.md_cb.name, self.py_cb.name)

    def test_codebook_version(self):
        self.assertEqual(self.md_cb.version, self.py_cb.version)

    def test_codebook_description_normalized(self):
        self.assertEqual(
            _normalize(self.md_cb.description),
            _normalize(self.py_cb.description),
        )

    def test_code_count(self):
        self.assertEqual(len(self.md_cb.codes), len(self.py_cb.codes))

    def test_code_ids(self):
        for md_c, py_c in zip(self.md_cb.codes, self.py_cb.codes):
            with self.subTest(code_id=py_c.code_id):
                self.assertEqual(md_c.code_id, py_c.code_id)

    def test_categories(self):
        for md_c, py_c in zip(self.md_cb.codes, self.py_cb.codes):
            with self.subTest(code_id=py_c.code_id):
                self.assertEqual(md_c.category, py_c.category)

    def test_domains(self):
        for md_c, py_c in zip(self.md_cb.codes, self.py_cb.codes):
            with self.subTest(code_id=py_c.code_id):
                self.assertEqual(md_c.domain, py_c.domain)

    def test_descriptions_normalized(self):
        for md_c, py_c in zip(self.md_cb.codes, self.py_cb.codes):
            with self.subTest(code_id=py_c.code_id):
                self.assertEqual(
                    _normalize(md_c.description),
                    _normalize(py_c.description),
                )

    def test_inclusive_criteria_normalized(self):
        for md_c, py_c in zip(self.md_cb.codes, self.py_cb.codes):
            with self.subTest(code_id=py_c.code_id):
                self.assertEqual(
                    _normalize(md_c.inclusive_criteria),
                    _normalize(py_c.inclusive_criteria),
                )

    def test_exclusive_criteria_normalized(self):
        for md_c, py_c in zip(self.md_cb.codes, self.py_cb.codes):
            with self.subTest(code_id=py_c.code_id):
                self.assertEqual(
                    _normalize(md_c.exclusive_criteria),
                    _normalize(py_c.exclusive_criteria),
                )

    def test_domains_dict(self):
        self.assertEqual(set(self.md_cb.domains.keys()), set(self.py_cb.domains.keys()))


# ---------------------------------------------------------------------------
# Phase 2 — drop dead fields
# ---------------------------------------------------------------------------

class TestPhase2DroppedFields(unittest.TestCase):
    """
    Phase 2 gate: color must be absent from ThemeDefinition; subcodes must
    be absent from CodeDefinition.  These tests are RED until the fields
    are removed from the dataclasses.
    """

    def test_theme_definition_has_no_color_field(self):
        import dataclasses
        from constructs.theme_schema import ThemeDefinition
        field_names = {f.name for f in dataclasses.fields(ThemeDefinition)}
        self.assertNotIn(
            'color', field_names,
            "ThemeDefinition.color must be removed (Phase 2)"
        )

    def test_code_definition_has_no_subcodes_field(self):
        import dataclasses
        from constructs.codebook.codebook_schema import CodeDefinition
        field_names = {f.name for f in dataclasses.fields(CodeDefinition)}
        self.assertNotIn(
            'subcodes', field_names,
            "CodeDefinition.subcodes must be removed (Phase 2)"
        )

    def test_codebook_prompt_string_has_no_subcodes(self):
        from constructs.codebook.phenomenology_codebook import get_phenomenology_codebook
        cb = get_phenomenology_codebook()
        prompt = cb.to_prompt_string()
        self.assertNotIn('Subcodes:', prompt)
        self.assertNotIn('subcodes', prompt.lower())

    def test_codebook_embedding_targets_have_no_subcodes(self):
        from constructs.codebook.phenomenology_codebook import get_phenomenology_codebook
        cb = get_phenomenology_codebook()
        targets = cb.to_embedding_targets()
        for t in targets:
            self.assertNotIn('subcodes', t)


# ---------------------------------------------------------------------------
# Phase 3 — factories backed by markdown
# ---------------------------------------------------------------------------

class TestPhase4FrameworkRegistry(unittest.TestCase):
    """
    Phase 4 gate: PipelineConfig must have participant_framework /
    therapist_framework fields; constructs/registry.py must exist and
    resolve framework names to ThemeFramework objects.
    """

    def test_pipeline_config_has_participant_framework(self):
        import dataclasses
        from process.config import PipelineConfig
        field_names = {f.name for f in dataclasses.fields(PipelineConfig)}
        self.assertIn('participant_framework', field_names)

    def test_pipeline_config_has_therapist_framework(self):
        import dataclasses
        from process.config import PipelineConfig
        field_names = {f.name for f in dataclasses.fields(PipelineConfig)}
        self.assertIn('therapist_framework', field_names)

    def test_participant_framework_defaults_to_vaamr(self):
        from process.config import PipelineConfig
        cfg = PipelineConfig()
        self.assertEqual(cfg.participant_framework, 'vaamr')

    def test_therapist_framework_defaults_to_purer(self):
        from process.config import PipelineConfig
        cfg = PipelineConfig()
        self.assertEqual(cfg.therapist_framework, 'purer')

    def test_registry_module_exists(self):
        from constructs import registry  # noqa: F401

    def test_registry_load_vaamr(self):
        from constructs.registry import load
        fw = load('vaamr')
        self.assertEqual(fw.name, 'VAAMR')
        self.assertEqual(len(fw.themes), 5)

    def test_registry_load_purer(self):
        from constructs.registry import load
        fw = load('purer')
        self.assertEqual(fw.name, 'PURER')
        self.assertEqual(len(fw.themes), 5)

    def test_registry_load_none_returns_none(self):
        from constructs.registry import load
        self.assertIsNone(load(None))

    def test_registry_unknown_name_raises(self):
        from constructs.registry import load
        with self.assertRaises(KeyError):
            load('unknown_framework')


class TestPhase3FactoriesBackedByMarkdown(unittest.TestCase):
    """
    Phase 3 gate: get_vaamr_framework(), get_purer_framework(), and
    get_phenomenology_codebook() must return objects parsed from the
    markdown files, not from inline Python data.

    Verified by confirming the factories delegate to the markdown loaders
    (checked via __module__ of the returned objects' class — the dataclass
    definition lives in theme_schema / codebook_schema, which is unchanged,
    so we verify behaviour by checking the factory source file is the shim).
    """

    def test_vaamr_factory_uses_markdown(self):
        """get_vaamr_framework must load from VAAMR.md (not inline Python data)."""
        import inspect
        import constructs.vaamr as mod
        src = inspect.getsource(mod.get_vaamr_framework)
        self.assertIn('load_framework_md', src,
                      "get_vaamr_framework() must delegate to load_framework_md")
        self.assertNotIn('ThemeDefinition(', src,
                         "get_vaamr_framework() must not inline ThemeDefinition construction")

    def test_purer_factory_uses_markdown(self):
        """get_purer_framework must load from PURER.md (not inline Python data)."""
        import inspect
        import constructs.purer as mod
        src = inspect.getsource(mod.get_purer_framework)
        self.assertIn('load_framework_md', src,
                      "get_purer_framework() must delegate to load_framework_md")
        self.assertNotIn('ThemeDefinition(', src,
                         "get_purer_framework() must not inline ThemeDefinition construction")

    def test_codebook_factory_uses_markdown(self):
        """get_phenomenology_codebook must load from CODEBOOK.md."""
        import inspect
        import constructs.codebook.phenomenology_codebook as mod
        src = inspect.getsource(mod.get_phenomenology_codebook)
        self.assertIn('load_codebook_md', src,
                      "get_phenomenology_codebook() must delegate to load_codebook_md")
        self.assertNotIn('CodeDefinition(', src,
                         "get_phenomenology_codebook() must not inline CodeDefinition construction")


if __name__ == '__main__':
    unittest.main()
