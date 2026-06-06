"""
tests/unit/test_methodology_lift.py
-------------------------------------
Conceptual conformance: cross-framework lift statistics (§4.5).

Methodology references:
  §4.5 Stage 4 — Cross-Framework Lift Statistics:
    Lift(stage, code) = P(code | stage) / P(code)
    Values above 1.0 indicate code appears more in that stage than overall.
    Values above 1.5 treated as substantively elevated.
    Minimum count filter (default 3) suppresses sparse codes.

  §3.3 (disclosure): purely exploratory; no automated expected-vs-observed
    comparison yet — lift table is generated and inspected by humans.

Source: process/cross_validation.py
  compute_theme_codebook_cooccurrence() — raw co-occurrence computation
  summarize_theme_code_associations() — filtered lift summary (min_lift=1.5, min_count=3)

Tests:
  1. Code independent of stage → lift ≈ 1.0 (within tolerance).
  2. Code perfectly concentrated in one stage → lift > 1.5 for that stage.
  3. Code appearing < 3 times across the corpus is dropped by the min_count filter.
  4. Lift formula: P(code|stage) / P(code) — verified numerically.
  5. No code appears in associations for a stage where it has 0 co-occurrences.
"""

import os
import sys
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

import pandas as pd

from process.cross_validation import (
    compute_theme_codebook_cooccurrence,
    summarize_theme_code_associations,
)
from tests.testhelpers import tiny_vaamr_framework


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------

def _corpus(stage_code_pairs, n_stages=5):
    """
    Build a DataFrame where each row is a segment with a given (stage, [codes]).

    stage_code_pairs: list of (stage_id, [code_str, ...])
    """
    rows = []
    for i, (stage, codes) in enumerate(stage_code_pairs):
        rows.append({
            'primary_stage': stage,
            'codebook_labels_ensemble': codes,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLiftIndependentCode(unittest.TestCase):
    """
    §4.5: a code appearing with equal probability across all stages has lift ≈ 1.0.
    We distribute CODE_A equally (one occurrence per stage) across five stages.
    """

    def setUp(self):
        self.fw = tiny_vaamr_framework()
        # One segment per stage, all carrying 'code_a' → uniform distribution
        pairs = [(i, ['code_a']) for i in range(5)]
        # Add enough segments so min_count(3) is satisfied: 5 segs with code_a
        # We need more than 3, so we repeat
        pairs = pairs * 3   # 15 segments, 3 per stage, all with code_a
        self.df = _corpus(pairs)

    def test_independent_code_lift_approximately_one(self):
        """code_a distributed equally → lift ≈ 1.0 for each stage."""
        cooc = compute_theme_codebook_cooccurrence(
            self.df, self.fw,
            codebook_label_column='codebook_labels_ensemble',
            theme_label_column='primary_stage',
        )
        for stage_key, stats in cooc.items():
            if 'code_a' in stats:
                lift = stats['code_a']['lift']
                self.assertAlmostEqual(
                    lift, 1.0, delta=0.1,
                    msg=f"Independent code should have lift ≈ 1.0, got {lift} for {stage_key}"
                )


class TestLiftPerfectlyConcentratedCode(unittest.TestCase):
    """
    §4.5: a code appearing only in one stage should have lift > 1.5 for that stage.
    """

    def setUp(self):
        self.fw = tiny_vaamr_framework()
        # 'rare_code' appears only in stage 4 (Reappraisal), 4 times
        # Other stages (0-3) appear 4 times each without 'rare_code'
        pairs = (
            [(4, ['rare_code', 'common_code'])] * 4  # stage 4: has rare_code
            + [(0, ['common_code'])] * 4              # stage 0: no rare_code
            + [(1, ['common_code'])] * 4
            + [(2, ['common_code'])] * 4
            + [(3, ['common_code'])] * 4
        )
        self.df = _corpus(pairs)

    def test_concentrated_code_has_lift_above_threshold(self):
        """rare_code perfectly concentrated in stage 4 → lift >> 1.5 for stage 4."""
        cooc = compute_theme_codebook_cooccurrence(
            self.df, self.fw,
            codebook_label_column='codebook_labels_ensemble',
            theme_label_column='primary_stage',
        )
        # stage 4's key in tiny framework is 'reappraisal'
        reappraisal_stats = cooc.get('reappraisal', {})
        self.assertIn('rare_code', reappraisal_stats,
                      "rare_code must appear in stage 4 (reappraisal) co-occurrence")
        lift = reappraisal_stats['rare_code']['lift']
        self.assertGreater(lift, 1.5,
                           f"Perfectly concentrated code should have lift >> 1.5, got {lift}")

    def test_concentrated_code_has_lift_near_zero_in_other_stages(self):
        """rare_code not present in other stages → not in their co-occurrence stats."""
        cooc = compute_theme_codebook_cooccurrence(
            self.df, self.fw,
            codebook_label_column='codebook_labels_ensemble',
            theme_label_column='primary_stage',
        )
        for stage_key in ('vigilance', 'avoidance', 'attention_regulation', 'metacognition'):
            stage_stats = cooc.get(stage_key, {})
            # rare_code should not appear in these stages at all (count=0)
            self.assertNotIn(
                'rare_code', stage_stats,
                f"rare_code should not appear in {stage_key} (count=0)"
            )


class TestLiftMinCountFilter(unittest.TestCase):
    """
    §4.5 / §3.3: minimum count filter (default 3) drops codes appearing < 3 times.
    """

    def setUp(self):
        self.fw = tiny_vaamr_framework()
        # 'sparse_code' appears only twice (< 3), 'common_code' appears 6 times
        pairs = (
            [(2, ['sparse_code', 'common_code'])] * 2  # sparse_code: 2 occurrences
            + [(2, ['common_code'])] * 4               # common_code: 6 total
        )
        self.df = _corpus(pairs)

    def test_sparse_code_excluded_by_min_count_filter(self):
        """sparse_code (count=2) is excluded when min_count=3."""
        cooc = compute_theme_codebook_cooccurrence(
            self.df, self.fw,
            codebook_label_column='codebook_labels_ensemble',
            theme_label_column='primary_stage',
        )
        summary = summarize_theme_code_associations(cooc, min_lift=1.0, min_count=3)
        # Retrieve associations for stage 2 (attention_regulation)
        stage_summary = summary.get('attention_regulation', {})
        assoc_codes = [a['code'] for a in stage_summary.get('top_associations', [])]
        self.assertNotIn(
            'sparse_code', assoc_codes,
            "sparse_code (count=2) must be excluded by the min_count=3 filter (§4.5)"
        )

    def test_common_code_retained_above_min_count(self):
        """common_code (count=6) survives the min_count=3 filter when lift ≥ min_lift."""
        cooc = compute_theme_codebook_cooccurrence(
            self.df, self.fw,
            codebook_label_column='codebook_labels_ensemble',
            theme_label_column='primary_stage',
        )
        summary = summarize_theme_code_associations(cooc, min_lift=1.0, min_count=3)
        stage_summary = summary.get('attention_regulation', {})
        assoc_codes = [a['code'] for a in stage_summary.get('top_associations', [])]
        self.assertIn(
            'common_code', assoc_codes,
            "common_code (count=6) should survive the min_count=3 filter"
        )

    def test_code_with_exactly_two_occurrences_dropped(self):
        """Exactly 2 occurrences (strictly < 3) must be dropped."""
        pairs = [(0, ['exact_two'])] * 2 + [(0, ['other'])] * 4
        df = _corpus(pairs)
        cooc = compute_theme_codebook_cooccurrence(
            df, self.fw,
            codebook_label_column='codebook_labels_ensemble',
            theme_label_column='primary_stage',
        )
        summary = summarize_theme_code_associations(cooc, min_lift=1.0, min_count=3)
        vigilance = summary.get('vigilance', {})
        codes = [a['code'] for a in vigilance.get('top_associations', [])]
        self.assertNotIn('exact_two', codes,
                         "code with exactly 2 occurrences must be filtered out")

    def test_code_with_exactly_three_occurrences_retained(self):
        """Exactly 3 occurrences meets the min_count=3 threshold and is retained."""
        pairs = [(0, ['exact_three'])] * 3 + [(0, [])] * 2
        df = _corpus(pairs)
        cooc = compute_theme_codebook_cooccurrence(
            df, self.fw,
            codebook_label_column='codebook_labels_ensemble',
            theme_label_column='primary_stage',
        )
        summary = summarize_theme_code_associations(cooc, min_lift=1.0, min_count=3)
        vigilance = summary.get('vigilance', {})
        codes = [a['code'] for a in vigilance.get('top_associations', [])]
        self.assertIn('exact_three', codes,
                      "code with exactly 3 occurrences should be retained (min_count=3)")


class TestLiftFormula(unittest.TestCase):
    """
    §4.5: Lift(stage, code) = P(code | stage) / P(code)
    Verify the formula numerically with a controlled corpus.
    """

    def setUp(self):
        self.fw = tiny_vaamr_framework()

    def test_lift_formula_numerically(self):
        """
        Corpus: 10 segments total.
          Stage 0: 4 segments, 4 carry 'target_code'  → P(code|stage0) = 4/4 = 1.0
          Stage 1: 6 segments, 0 carry 'target_code'  → P(code|stage1) = 0/6 = 0.0
          P(target_code) = 4/10 = 0.4
          Lift(stage0, target_code) = 1.0 / 0.4 = 2.5
        """
        pairs = (
            [(0, ['target_code'])] * 4
            + [(1, [])] * 6
        )
        df = _corpus(pairs)
        cooc = compute_theme_codebook_cooccurrence(
            df, self.fw,
            codebook_label_column='codebook_labels_ensemble',
            theme_label_column='primary_stage',
        )
        vigilance = cooc.get('vigilance', {})
        self.assertIn('target_code', vigilance)
        expected_lift = 1.0 / 0.4  # = 2.5
        self.assertAlmostEqual(
            vigilance['target_code']['lift'], expected_lift, delta=0.01,
            msg=f"Lift formula P(code|stage)/P(code): expected {expected_lift}"
        )

    def test_lift_base_rate_computation(self):
        """
        base_rate is computed over ALL labeled segments, not just the stage.
        Verify that rate and base_rate are stored correctly.
        """
        # Stage 0: 3 segments, all with 'the_code'
        # Stage 1: 3 segments, none with 'the_code'
        # P(the_code) = 3/6 = 0.5
        # P(the_code | stage 0) = 3/3 = 1.0
        # lift = 1.0/0.5 = 2.0
        pairs = [(0, ['the_code'])] * 3 + [(1, [])] * 3
        df = _corpus(pairs)
        cooc = compute_theme_codebook_cooccurrence(
            df, self.fw,
            codebook_label_column='codebook_labels_ensemble',
            theme_label_column='primary_stage',
        )
        stats = cooc['vigilance']['the_code']
        self.assertAlmostEqual(stats['rate'], 1.0, delta=0.01)
        self.assertAlmostEqual(stats['base_rate'], 0.5, delta=0.01)
        self.assertAlmostEqual(stats['lift'], 2.0, delta=0.05)


class TestSummarizeAssociations(unittest.TestCase):
    """
    summarize_theme_code_associations applies min_lift and min_count jointly.
    Only codes with both lift ≥ min_lift AND count ≥ min_count appear.
    """

    def setUp(self):
        self.fw = tiny_vaamr_framework()

    def test_only_high_lift_codes_in_top_associations(self):
        """Codes with lift < 1.5 are excluded even if count >= 3."""
        # 'independent_code' uniform across stages → lift ≈ 1.0 < 1.5
        pairs = [(i, ['independent_code']) for i in range(5)] * 3
        df = _corpus(pairs)
        cooc = compute_theme_codebook_cooccurrence(
            df, self.fw,
            codebook_label_column='codebook_labels_ensemble',
            theme_label_column='primary_stage',
        )
        summary = summarize_theme_code_associations(cooc, min_lift=1.5, min_count=3)
        for stage_key, stage_data in summary.items():
            assoc_codes = [a['code'] for a in stage_data.get('top_associations', [])]
            self.assertNotIn(
                'independent_code', assoc_codes,
                f"uniform code (lift≈1.0) should be excluded from {stage_key} associations"
            )


if __name__ == '__main__':
    unittest.main()
