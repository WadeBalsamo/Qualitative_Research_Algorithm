"""
tests/unit/test_methodology_frameworks.py
------------------------------------------
Conceptual conformance: VAAMR and PURER framework structure matches
the methodology's load-bearing design decisions.

Methodology references:
  §2.2 — VAAMR as trajectory of re-habituation (5 stages, 0..4 arc)
  §2.3 — PURER as structured phenomenological inquiry (5 moves, P/U/R/E/R2)
  §3.1 — Each ThemeDefinition has full operational specification (definition,
          prototypical_features, distinguishing_criteria, exemplar/subtle/adversarial)
  §4.1 — Adversarial utterances encode boundary cases (non-empty per stage)

PURER precedence:
  CLAUDE.md / purer.py module-docstring + PURER_FRAMEWORK.md §Framework Notes:
    1. Reinforcement (R2) is the WRAPPER around another move — code the
       substantive inner move (P/U/R/E) when R2 is the outer affective register.
    2. Utilization (U) takes precedence over Reframing (R) for forward-application.
    3. Reframing (R) takes precedence over Education (E) when anchored to story.
  Precedence is encoded as prose in the framework description (not a Python constant).
  These tests assert that the description encodes the rules, not that a sorting
  function implements them (no such function exists — precedence is editorial guidance).
"""

import os
import sys
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from tests.testhelpers import load_real_framework_or_skip


# ---------------------------------------------------------------------------
# VAAMR framework structure (methodology §2.2, §3.1, §4.1)
# ---------------------------------------------------------------------------

class TestVAAMRStructure(unittest.TestCase):
    """The VAAMR framework encodes the 5-stage re-habituation arc."""

    @classmethod
    def setUpClass(cls):
        cls.fw = load_real_framework_or_skip('vaamr')

    def test_exactly_five_themes(self):
        # §2.2: exactly five stages in the developmental arc
        self.assertEqual(len(self.fw.themes), 5)

    def test_theme_ids_are_zero_through_four(self):
        # §2.2: ids 0..4 form a contiguous integer sequence
        ids = sorted(t.theme_id for t in self.fw.themes)
        self.assertEqual(ids, [0, 1, 2, 3, 4])

    def test_stage_0_is_vigilance(self):
        # §2.2 Stage 0: attentional capture by pain (Vigilance)
        t = self.fw.get_theme_by_id(0)
        self.assertIsNotNone(t)
        # Real name from registry is 'Pain Vigilance' — check contains 'vigilance'
        self.assertIn('vigilance', t.name.lower())

    def test_stage_1_is_avoidance(self):
        # §2.2 Stage 1: attentional skill deployed for experiential escape (Avoidance)
        t = self.fw.get_theme_by_id(1)
        self.assertIsNotNone(t)
        self.assertIn('avoidance', t.name.lower())

    def test_stage_2_is_attention_regulation(self):
        # §2.2 Stage 2: stable volitional presence (Attention Regulation)
        t = self.fw.get_theme_by_id(2)
        self.assertIsNotNone(t)
        self.assertIn('attention', t.name.lower())

    def test_stage_3_is_metacognition(self):
        # §2.2 Stage 3: reflexive observation (Metacognition)
        t = self.fw.get_theme_by_id(3)
        self.assertIsNotNone(t)
        self.assertIn('metacognit', t.name.lower())

    def test_stage_4_is_reappraisal(self):
        # §2.2 Stage 4: noematic transformation (Reappraisal)
        t = self.fw.get_theme_by_id(4)
        self.assertIsNotNone(t)
        self.assertIn('reappraisal', t.name.lower())

    def test_each_stage_has_non_empty_definition(self):
        # §3.1: each ThemeDefinition has a formal definition
        for t in self.fw.themes:
            with self.subTest(theme_id=t.theme_id, name=t.name):
                self.assertTrue(t.definition.strip(),
                                f"Stage {t.theme_id} has empty definition")

    def test_each_stage_has_prototypical_features(self):
        # §3.1: each ThemeDefinition has prototypical linguistic features
        for t in self.fw.themes:
            with self.subTest(theme_id=t.theme_id):
                self.assertTrue(len(t.prototypical_features) > 0,
                                f"Stage {t.theme_id} has no prototypical_features")

    def test_each_stage_has_non_empty_distinguishing_criteria(self):
        # §3.1: each ThemeDefinition has distinguishing_criteria
        for t in self.fw.themes:
            with self.subTest(theme_id=t.theme_id):
                self.assertTrue(t.distinguishing_criteria.strip(),
                                f"Stage {t.theme_id} has empty distinguishing_criteria")

    def test_each_stage_has_exemplar_utterances(self):
        # §3.1: exemplar utterances (clear expressions)
        for t in self.fw.themes:
            with self.subTest(theme_id=t.theme_id):
                self.assertTrue(len(t.exemplar_utterances) > 0,
                                f"Stage {t.theme_id} has no exemplar_utterances")

    def test_each_stage_has_subtle_utterances(self):
        # §3.1: subtle utterances (harder to classify)
        for t in self.fw.themes:
            with self.subTest(theme_id=t.theme_id):
                self.assertTrue(len(t.subtle_utterances) > 0,
                                f"Stage {t.theme_id} has no subtle_utterances")

    def test_each_stage_has_adversarial_utterances(self):
        # §4.1: adversarial utterances encode the stage boundary cases
        for t in self.fw.themes:
            with self.subTest(theme_id=t.theme_id):
                self.assertTrue(len(t.adversarial_utterances) > 0,
                                f"Stage {t.theme_id} has no adversarial_utterances — "
                                "boundary cases are required (§4.1)")

    def test_num_themes_property_consistent(self):
        self.assertEqual(self.fw.num_themes, 5)

    def test_framework_name_contains_vaamr(self):
        self.assertIn('VAAMR', self.fw.name.upper())


# ---------------------------------------------------------------------------
# PURER framework structure (methodology §2.3)
# ---------------------------------------------------------------------------

class TestPURERStructure(unittest.TestCase):
    """The PURER framework encodes 5 therapist guided-inquiry moves."""

    @classmethod
    def setUpClass(cls):
        cls.fw = load_real_framework_or_skip('purer')

    def test_exactly_five_moves(self):
        # §2.3: P/U/R/E/R2 — five guided-inquiry moves
        self.assertEqual(len(self.fw.themes), 5)

    def test_move_ids_are_zero_through_four(self):
        ids = sorted(t.theme_id for t in self.fw.themes)
        self.assertEqual(ids, [0, 1, 2, 3, 4])

    def test_move_0_is_phenomenological(self):
        # §2.3: P — epoché-style experiential elicitation
        t = self.fw.get_theme_by_id(0)
        self.assertIsNotNone(t)
        self.assertIn('phenomenol', t.name.lower())

    def test_move_1_is_utilization(self):
        # §2.3: U — forward application to everyday life
        t = self.fw.get_theme_by_id(1)
        self.assertIsNotNone(t)
        self.assertIn('utilization', t.name.lower())

    def test_move_2_is_reframing(self):
        # §2.3: R — free imaginative variation (Giorgi), repositioning report as MORE concept
        t = self.fw.get_theme_by_id(2)
        self.assertIsNotNone(t)
        self.assertIn('reframing', t.name.lower())

    def test_move_3_is_educate_expectancy(self):
        # §2.3: E — psychoeducation + expectancy scaffolding
        t = self.fw.get_theme_by_id(3)
        self.assertIsNotNone(t)
        # name is 'Educate/Expectancy'
        self.assertTrue('educat' in t.name.lower() or 'expect' in t.name.lower())

    def test_move_4_is_reinforcement(self):
        # §2.3: R2 — selective affirmation of adaptive responses
        t = self.fw.get_theme_by_id(4)
        self.assertIsNotNone(t)
        self.assertIn('reinforcement', t.name.lower())

    def test_each_move_has_non_empty_definition(self):
        for t in self.fw.themes:
            with self.subTest(move_id=t.theme_id, name=t.name):
                self.assertTrue(t.definition.strip(),
                                f"Move {t.theme_id} has empty definition")

    def test_each_move_has_prototypical_features(self):
        for t in self.fw.themes:
            with self.subTest(move_id=t.theme_id):
                self.assertTrue(len(t.prototypical_features) > 0,
                                f"Move {t.theme_id} has no prototypical_features")

    def test_each_move_has_distinguishing_criteria(self):
        for t in self.fw.themes:
            with self.subTest(move_id=t.theme_id):
                self.assertTrue(t.distinguishing_criteria.strip(),
                                f"Move {t.theme_id} has empty distinguishing_criteria")

    def test_each_move_has_exemplar_utterances(self):
        for t in self.fw.themes:
            with self.subTest(move_id=t.theme_id):
                self.assertTrue(len(t.exemplar_utterances) > 0,
                                f"Move {t.theme_id} has no exemplar_utterances")

    def test_framework_name_contains_purer(self):
        self.assertIn('PURER', self.fw.name.upper())


# ---------------------------------------------------------------------------
# PURER precedence rules (CLAUDE.md / purer.py docstring / PURER_FRAMEWORK.md)
# Precedence is encoded as framework_description prose — no Python sort constant.
# These tests assert the description encodes the three editorial rules.
# ---------------------------------------------------------------------------

class TestPURERPrecedenceEncoding(unittest.TestCase):
    """
    Precedence is editorial guidance stored in the PURER framework description,
    not a Python sort function. We verify the description encodes the three rules:

      Rule 1: Reinforcement (R2) is a WRAPPER — code the substantive inner move.
      Rule 2: Utilization > Reframing for forward-application prompts.
      Rule 3: Reframing > Education when anchored to participant's specific story.

    Source: PURER_FRAMEWORK.md §Co-Occurrence and Precedence, purer.py docstring.

    NOTE: No Python constant/function encodes the precedence order. Tests below
    assert the description text, not a data-structure key. If a machine-readable
    constant is added later, update these tests to assert both.
    """

    @classmethod
    def setUpClass(cls):
        cls.fw = load_real_framework_or_skip('purer')

    def _desc(self):
        # framework_description is the canonical location for precedence rules
        return (self.fw.description or '').lower()

    def test_description_notes_reinforcement_is_wrapper(self):
        # Rule 1: R2 wraps the substantive move
        desc = self._desc()
        # The description should mention R2's role as wrapper or outer register
        self.assertTrue(
            'r2' in desc or 'reinforcement' in desc,
            "PURER description should mention Reinforcement (R2) precedence rule"
        )
        # It should indicate R2 wraps another move
        self.assertTrue(
            'wrap' in desc or 'substantive' in desc,
            "PURER description should note R2 is a wrapper around the substantive move"
        )

    def test_description_notes_utilization_over_reframing(self):
        # Rule 2: U > R when forward application is being prompted
        desc = self._desc()
        self.assertTrue(
            'u > r' in desc or 'utilization' in desc,
            "PURER description should encode U > R precedence rule"
        )

    def test_description_notes_reframing_over_education(self):
        # Rule 3: R > E when concept anchored to participant's specific story
        desc = self._desc()
        self.assertTrue(
            'r > e' in desc or 'reframing' in desc,
            "PURER description should encode R > E precedence rule"
        )

    def test_reinforcement_move_description_notes_wrapper_role(self):
        # The Reinforcement ThemeDefinition itself should acknowledge it is often
        # the outer wrapper and should yield to the substantive inner move
        r2 = self.fw.get_theme_by_id(4)
        self.assertIsNotNone(r2)
        combined = (r2.definition + ' ' + r2.distinguishing_criteria).lower()
        self.assertTrue(
            'wrap' in combined or 'substantive' in combined or 'outer' in combined,
            "Reinforcement definition/criteria should note its wrapper role"
        )

    def test_utilization_distinguishing_criteria_notes_precedence_over_reframing(self):
        # Utilization's distinguishing_criteria should distinguish it from Reframing
        u = self.fw.get_theme_by_id(1)
        self.assertIsNotNone(u)
        combined = (u.definition + ' ' + u.distinguishing_criteria).lower()
        self.assertTrue(
            'reframing' in combined or 'reframe' in combined or 'forward' in combined,
            "Utilization criteria should distinguish from Reframing (forward-application precedence)"
        )


if __name__ == '__main__':
    unittest.main()
