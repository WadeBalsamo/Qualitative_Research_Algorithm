"""
tests/unit/test_methodology_confidence_tiers.py
------------------------------------------------
Conceptual conformance: confidence tier assignment matches the documented
boundaries.

Methodology reference §4.3:
  "A confidence-tiering system integrates run consistency with per-run
   confidence into a single reliability indicator:
     *high*   — unanimous agreement across runs AND per-run confidence > 0.80
     *medium* — majority agreement AND confidence > 0.60
     *low*    — everything else (including all split-vote segments regardless
                of confidence)"

The real thresholds are stored in process/config.py:ConfidenceTierConfig
and consumed by process/assembly/master_dataset.py:assemble_master_dataset.

Tests drive assemble_master_dataset with segments designed to hit each tier
boundary exactly, asserting label_confidence_tier.

Tier logic (from master_dataset.py):
  HIGH:   llm_run_consistency == high_consistency (3)
          AND llm_confidence_primary > high_confidence (0.80)
  MEDIUM: llm_run_consistency >= medium_min_consistency (2)
          AND llm_confidence_primary > medium_min_confidence (0.60)
  LOW:    everything else

NB: The condition is strict >, not >=, for both confidence thresholds.
    Consistency is == for HIGH (unanimous = all 3 runs agree), >= for MEDIUM.
"""

import os
import sys
import tempfile
import shutil
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from classification_tools.data_structures import Segment
from process.assembly.master_dataset import assemble_master_dataset
from process.config import ConfidenceTierConfig


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _seg(sid, consistency, confidence, primary_stage=2):
    """Build a participant segment with the given consistency/confidence."""
    seg = Segment(
        segment_id=sid, session_id='c1s1',
        speaker='participant',
        text='I noticed the sensation.',
    )
    seg.primary_stage = primary_stage
    seg.llm_run_consistency = consistency
    seg.llm_confidence_primary = confidence
    return seg


def _run(segments):
    tmpdir = tempfile.mkdtemp()
    try:
        outpath = os.path.join(tmpdir, 'master.jsonl')
        df = assemble_master_dataset(
            segments=segments,
            output_path=outpath,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return df


def _tier(df, sid):
    return df[df['segment_id'] == sid].iloc[0]['label_confidence_tier']


# ---------------------------------------------------------------------------
# Verify ConfidenceTierConfig defaults match §4.3 documented boundaries
# ---------------------------------------------------------------------------

class TestConfidenceTierConfigDefaults(unittest.TestCase):
    """
    §4.3: the documented thresholds (0.80 high, 0.60 medium, 3 runs unanimous)
    must be reflected in ConfidenceTierConfig defaults.
    """

    def setUp(self):
        self.ct = ConfidenceTierConfig()

    def test_high_consistency_is_three(self):
        # "unanimous agreement across runs" — for n_runs=3, that means all 3
        self.assertEqual(self.ct.high_consistency, 3)

    def test_high_confidence_threshold_is_080(self):
        # §4.3: "per-run confidence > 0.80"
        self.assertAlmostEqual(self.ct.high_confidence, 0.80)

    def test_medium_min_consistency_is_two(self):
        # majority for 3 runs means at least 2 agree
        self.assertEqual(self.ct.medium_min_consistency, 2)

    def test_medium_min_confidence_threshold_is_060(self):
        # §4.3: "confidence above 0.60"
        self.assertAlmostEqual(self.ct.medium_min_confidence, 0.60)


# ---------------------------------------------------------------------------
# HIGH tier
# ---------------------------------------------------------------------------

class TestHighConfidenceTier(unittest.TestCase):
    """
    §4.3: HIGH requires unanimous agreement (consistency==3) AND confidence > 0.80.
    """

    def test_high_tier_unanimous_and_above_threshold(self):
        # Exactly unanimous (3/3) and confidence well above 0.80
        seg = _seg('h1', consistency=3, confidence=0.90)
        df = _run([seg])
        self.assertEqual(_tier(df, 'h1'), 'high')

    def test_high_tier_at_exactly_081_confidence(self):
        # Just above the 0.80 boundary (strict >)
        seg = _seg('h2', consistency=3, confidence=0.81)
        df = _run([seg])
        self.assertEqual(_tier(df, 'h2'), 'high')

    def test_high_tier_requires_unanimous_consistency(self):
        # consistency=2 (majority, not unanimous) → cannot be high, even with high confidence
        seg = _seg('h3', consistency=2, confidence=0.95)
        df = _run([seg])
        self.assertNotEqual(_tier(df, 'h3'), 'high',
                            "consistency < 3 must not qualify for high tier (non-unanimous)")

    def test_high_tier_fails_at_exactly_080_confidence(self):
        # Threshold is strictly > 0.80; exactly 0.80 should NOT qualify
        seg = _seg('h4', consistency=3, confidence=0.80)
        df = _run([seg])
        self.assertNotEqual(_tier(df, 'h4'), 'high',
                            "confidence == 0.80 does not satisfy > 0.80 (high tier)")


# ---------------------------------------------------------------------------
# MEDIUM tier
# ---------------------------------------------------------------------------

class TestMediumConfidenceTier(unittest.TestCase):
    """
    §4.3: MEDIUM requires majority agreement (consistency >= 2) AND confidence > 0.60.
    """

    def test_medium_tier_majority_and_above_threshold(self):
        # 2 of 3 agree, confidence above 0.60
        seg = _seg('m1', consistency=2, confidence=0.75)
        df = _run([seg])
        self.assertEqual(_tier(df, 'm1'), 'medium')

    def test_medium_tier_at_exactly_061_confidence(self):
        # Just above 0.60 (strict >)
        seg = _seg('m2', consistency=2, confidence=0.61)
        df = _run([seg])
        self.assertEqual(_tier(df, 'm2'), 'medium')

    def test_medium_tier_with_unanimous_but_low_confidence(self):
        # consistency=3 (unanimous) but confidence just above 0.60 → medium
        # (unanimous + low confidence is not high, but is medium)
        seg = _seg('m3', consistency=3, confidence=0.70)
        df = _run([seg])
        # Not high (confidence <= 0.80), should be medium (consistency >= 2, conf > 0.60)
        self.assertEqual(_tier(df, 'm3'), 'medium')

    def test_medium_tier_fails_at_exactly_060_confidence(self):
        # Exactly 0.60 does not satisfy > 0.60
        seg = _seg('m4', consistency=2, confidence=0.60)
        df = _run([seg])
        self.assertNotEqual(_tier(df, 'm4'), 'medium',
                            "confidence == 0.60 does not satisfy > 0.60 (medium tier)")

    def test_medium_tier_fails_with_consistency_one(self):
        # Split/minority vote (consistency=1) → low even with high confidence
        seg = _seg('m5', consistency=1, confidence=0.90)
        df = _run([seg])
        self.assertNotEqual(_tier(df, 'm5'), 'medium')


# ---------------------------------------------------------------------------
# LOW tier
# ---------------------------------------------------------------------------

class TestLowConfidenceTier(unittest.TestCase):
    """
    §4.3: LOW covers everything else — including all split-vote segments
    regardless of confidence.
    """

    def test_low_tier_split_vote_regardless_of_confidence(self):
        # §4.3: "all split-vote segments regardless of confidence"
        # consistency=1 = split/minority
        seg = _seg('l1', consistency=1, confidence=0.99)
        df = _run([seg])
        self.assertEqual(_tier(df, 'l1'), 'low',
                         "Split-vote segment must be low confidence regardless of confidence score")

    def test_low_tier_none_consistency(self):
        # No consistency info → low
        seg = _seg('l2', consistency=None, confidence=0.90)
        df = _run([seg])
        self.assertEqual(_tier(df, 'l2'), 'low')

    def test_low_tier_none_confidence(self):
        # No confidence info → low
        seg = _seg('l3', consistency=3, confidence=None)
        df = _run([seg])
        self.assertEqual(_tier(df, 'l3'), 'low')

    def test_low_tier_low_confidence_with_majority(self):
        # Majority consistency but confidence below threshold → low
        seg = _seg('l4', consistency=2, confidence=0.40)
        df = _run([seg])
        self.assertEqual(_tier(df, 'l4'), 'low')

    def test_low_tier_zero_consistency(self):
        # consistency=0 → low
        seg = _seg('l5', consistency=0, confidence=0.95)
        df = _run([seg])
        self.assertEqual(_tier(df, 'l5'), 'low')


# ---------------------------------------------------------------------------
# All three tiers coexist correctly in a single DataFrame
# ---------------------------------------------------------------------------

class TestAllThreeTiersInOneDataset(unittest.TestCase):
    """
    §4.3: the tiering system covers all three tiers distinctly.
    A dataset with one segment per tier should produce all three in the output.
    """

    def test_three_tiers_in_one_assembly(self):
        segs = [
            _seg('high', consistency=3, confidence=0.95),   # high
            _seg('med', consistency=2, confidence=0.70),    # medium
            _seg('low', consistency=1, confidence=0.99),    # low (split-vote)
        ]
        df = _run(segs)
        self.assertEqual(_tier(df, 'high'), 'high')
        self.assertEqual(_tier(df, 'med'), 'medium')
        self.assertEqual(_tier(df, 'low'), 'low')


if __name__ == '__main__':
    unittest.main()
