"""
tests/unit/test_response_parser.py
------------------------------------
Unit tests for classification_tools/response_parser.py

Covers parse_all_results and parse_purer_results:

- Valid consensus (primary_stage set) → sets primary_stage, secondary_stage,
  llm_confidence_primary, llm_justification, secondary_agreement_level,
  secondary_agreement_fraction, agreement_level, agreement_fraction,
  needs_review, consensus_vote, tie_broken_by_confidence, llm_run_consistency
- ABSTAIN consensus (consensus_vote='ABSTAIN', primary_stage=None) →
  primary_stage None, llm_confidence_primary and llm_justification still set,
  counted in 'abstained', NOT in 'classified'
- agreement_level='none' (all raters failed) → counted in 'all_raters_failed',
  needs_review=True
- agreement_level='split' → counted in 'split'
- Non-dict result or missing consensus key → counted in 'malformed'
- Unknown segment_id is silently skipped (no KeyError)
- rater_ids / rater_votes propagated from top-level and consensus fallback
- tie_broken_by_confidence propagated
- stats dict shape: keys total/parsed/abstained/split/all_raters_failed/malformed/error_breakdown
- parse_purer_results writes purer_* fields, not VAAMR fields
- Agreement fraction computed from n_agree / n_raters
"""

import os
import sys
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from classification_tools.data_structures import Segment
from classification_tools.response_parser import parse_all_results, parse_purer_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_seg(seg_id='s1', speaker='participant'):
    """Return a bare Segment with just an id and speaker."""
    return Segment(segment_id=seg_id, speaker=speaker)


def make_result(
    primary_stage=2,
    primary_confidence=0.8,
    secondary_stage=None,
    secondary_confidence=None,
    justification="test reason",
    consensus_vote=None,
    agreement_level='unanimous',
    n_agree=3,
    n_raters=3,
    n_ballots=3,
    needs_review=False,
    tie_broken_by_confidence=False,
    rater_votes=None,
    secondary_agreement_level=None,
    secondary_agreement_fraction=None,
    rater_ids=None,
):
    """Build a minimal merge-result dict matching the documented input shape."""
    if consensus_vote is None:
        consensus_vote = primary_stage  # default: coded stage
    consensus = {
        'primary_stage': primary_stage,
        'primary_confidence': primary_confidence,
        'secondary_stage': secondary_stage,
        'secondary_confidence': secondary_confidence,
        'justification': justification,
        'consensus_vote': consensus_vote,
        'agreement_level': agreement_level,
        'n_agree': n_agree,
        'n_raters': n_raters,
        'n_ballots': n_ballots,
        'needs_review': needs_review,
        'tie_broken_by_confidence': tie_broken_by_confidence,
        'rater_votes': rater_votes or [],
        'secondary_agreement_level': secondary_agreement_level,
        'secondary_agreement_fraction': secondary_agreement_fraction,
    }
    return {
        'rater_ids': rater_ids or ['run_1', 'run_2', 'run_3'],
        'rater_votes': rater_votes or [],
        'consensus': consensus,
    }


def make_abstain_result(n_raters=3, justification="not relevant"):
    """Build a result dict representing a unanimous ABSTAIN."""
    return make_result(
        primary_stage=None,
        consensus_vote='ABSTAIN',
        agreement_level='unanimous',
        n_agree=n_raters,
        n_raters=n_raters,
        n_ballots=n_raters,
        needs_review=False,
        justification=justification,
        primary_confidence=0.0,
    )


# ---------------------------------------------------------------------------
# parse_all_results: valid classified segment
# ---------------------------------------------------------------------------

class TestParseAllResultsClassified(unittest.TestCase):

    def setUp(self):
        self.seg = make_seg('s1')
        result = make_result(
            primary_stage=3,
            primary_confidence=0.87,
            secondary_stage=2,
            secondary_confidence=0.45,
            justification="Clear metacognition pattern",
            agreement_level='unanimous',
            n_agree=3,
            n_raters=3,
            n_ballots=3,
            needs_review=False,
            tie_broken_by_confidence=False,
            secondary_agreement_level='unanimous',
            secondary_agreement_fraction=1.0,
            rater_votes=[{'rater': 'run_1', 'vote': 'CODED', 'stage': 3}],
        )
        _, self.stats = parse_all_results({'s1': result}, [self.seg], {})

    def test_primary_stage_set(self):
        self.assertEqual(self.seg.primary_stage, 3)

    def test_secondary_stage_set(self):
        self.assertEqual(self.seg.secondary_stage, 2)

    def test_llm_confidence_primary_set(self):
        self.assertAlmostEqual(self.seg.llm_confidence_primary, 0.87)

    def test_llm_confidence_secondary_set(self):
        self.assertAlmostEqual(self.seg.llm_confidence_secondary, 0.45)

    def test_llm_justification_set(self):
        self.assertEqual(self.seg.llm_justification, "Clear metacognition pattern")

    def test_agreement_level_set(self):
        self.assertEqual(self.seg.agreement_level, 'unanimous')

    def test_agreement_fraction_set(self):
        self.assertAlmostEqual(self.seg.agreement_fraction, 1.0)

    def test_needs_review_set(self):
        self.assertFalse(self.seg.needs_review)

    def test_consensus_vote_set(self):
        self.assertEqual(self.seg.consensus_vote, 3)

    def test_llm_run_consistency_is_n_agree(self):
        self.assertEqual(self.seg.llm_run_consistency, 3)

    def test_secondary_agreement_level_propagated(self):
        self.assertEqual(self.seg.secondary_agreement_level, 'unanimous')

    def test_secondary_agreement_fraction_propagated(self):
        self.assertAlmostEqual(self.seg.secondary_agreement_fraction, 1.0)

    def test_stats_classified_count(self):
        self.assertEqual(self.stats['parsed'], 1)
        self.assertEqual(self.stats['abstained'], 0)
        self.assertEqual(self.stats['split'], 0)
        self.assertEqual(self.stats['all_raters_failed'], 0)
        self.assertEqual(self.stats['malformed'], 0)
        self.assertEqual(self.stats['total'], 1)


class TestParseAllResultsTieBrokenByConfidence(unittest.TestCase):

    def test_tie_broken_by_confidence_propagated(self):
        seg = make_seg('s2')
        result = make_result(
            primary_stage=1,
            tie_broken_by_confidence=True,
            agreement_level='majority',
            n_agree=2,
            n_raters=3,
        )
        parse_all_results({'s2': result}, [seg], {})
        self.assertTrue(seg.tie_broken_by_confidence)


# ---------------------------------------------------------------------------
# parse_all_results: rater IDs / votes
# ---------------------------------------------------------------------------

class TestParseAllResultsRaterFields(unittest.TestCase):

    def test_rater_ids_from_top_level(self):
        seg = make_seg('s1')
        result = make_result(rater_ids=['modelA', 'modelB'])
        parse_all_results({'s1': result}, [seg], {})
        self.assertEqual(seg.rater_ids, ['modelA', 'modelB'])

    def test_rater_votes_from_top_level(self):
        votes = [{'rater': 'run_1', 'vote': 'CODED', 'stage': 2}]
        seg = make_seg('s1')
        result = make_result(rater_votes=votes)
        parse_all_results({'s1': result}, [seg], {})
        self.assertEqual(seg.rater_votes, votes)

    def test_rater_votes_fallback_from_consensus(self):
        """When top-level rater_votes is absent, fall back to consensus['rater_votes']."""
        seg = make_seg('s1')
        result = make_result(rater_votes=[])
        # Inject votes only into consensus
        consensus_votes = [{'rater': 'run_1', 'vote': 'CODED', 'stage': 1}]
        result['rater_votes'] = []  # empty top-level
        result['consensus']['rater_votes'] = consensus_votes
        parse_all_results({'s1': result}, [seg], {})
        self.assertEqual(seg.rater_votes, consensus_votes)


# ---------------------------------------------------------------------------
# parse_all_results: ABSTAIN
# ---------------------------------------------------------------------------

class TestParseAllResultsAbstain(unittest.TestCase):

    def setUp(self):
        self.seg = make_seg('abs1')
        result = make_abstain_result(justification="therapist only content")
        _, self.stats = parse_all_results({'abs1': result}, [self.seg], {})

    def test_primary_stage_is_none(self):
        self.assertIsNone(self.seg.primary_stage)

    def test_secondary_stage_is_none(self):
        self.assertIsNone(self.seg.secondary_stage)

    def test_consensus_vote_is_abstain(self):
        self.assertEqual(self.seg.consensus_vote, 'ABSTAIN')

    def test_justification_still_set(self):
        self.assertEqual(self.seg.llm_justification, "therapist only content")

    def test_llm_confidence_primary_still_set(self):
        # primary_confidence=0.0 from make_abstain_result
        self.assertIsNotNone(self.seg.llm_confidence_primary)

    def test_counted_as_abstained(self):
        self.assertEqual(self.stats['abstained'], 1)
        self.assertEqual(self.stats['parsed'], 0)


# ---------------------------------------------------------------------------
# parse_all_results: split vote
# ---------------------------------------------------------------------------

class TestParseAllResultsSplit(unittest.TestCase):

    def setUp(self):
        self.seg = make_seg('sp1')
        result = make_result(
            primary_stage=None,
            consensus_vote=None,
            agreement_level='split',
            needs_review=True,
            n_agree=1,
            n_raters=3,
        )
        _, self.stats = parse_all_results({'sp1': result}, [self.seg], {})

    def test_primary_stage_is_none(self):
        self.assertIsNone(self.seg.primary_stage)

    def test_needs_review_is_true(self):
        self.assertTrue(self.seg.needs_review)

    def test_counted_as_split(self):
        self.assertEqual(self.stats['split'], 1)
        self.assertEqual(self.stats['parsed'], 0)


# ---------------------------------------------------------------------------
# parse_all_results: all raters failed (agreement_level='none')
# ---------------------------------------------------------------------------

class TestParseAllResultsAllRatersFailed(unittest.TestCase):

    def setUp(self):
        self.seg = make_seg('err1')
        result = make_result(
            primary_stage=None,
            consensus_vote=None,
            agreement_level='none',
            needs_review=True,
            n_agree=0,
            n_raters=3,
            n_ballots=0,
        )
        _, self.stats = parse_all_results({'err1': result}, [self.seg], {})

    def test_primary_stage_is_none(self):
        self.assertIsNone(self.seg.primary_stage)

    def test_needs_review_true(self):
        self.assertTrue(self.seg.needs_review)

    def test_counted_as_all_raters_failed(self):
        self.assertEqual(self.stats['all_raters_failed'], 1)
        self.assertEqual(self.stats['parsed'], 0)


# ---------------------------------------------------------------------------
# parse_all_results: malformed / non-dict result
# ---------------------------------------------------------------------------

class TestParseAllResultsMalformed(unittest.TestCase):

    def test_non_dict_result_counted_as_malformed(self):
        seg = make_seg('m1')
        _, stats = parse_all_results({'m1': "not a dict"}, [seg], {})
        self.assertEqual(stats['malformed'], 1)
        self.assertEqual(stats['parsed'], 0)
        # Segment should be untouched (primary_stage still None)
        self.assertIsNone(seg.primary_stage)

    def test_missing_consensus_key_counted_as_malformed(self):
        seg = make_seg('m2')
        result = {'rater_ids': ['r1'], 'rater_votes': []}  # no 'consensus' key
        _, stats = parse_all_results({'m2': result}, [seg], {})
        self.assertEqual(stats['malformed'], 1)

    def test_consensus_not_dict_counted_as_malformed(self):
        seg = make_seg('m3')
        result = {'rater_ids': [], 'rater_votes': [], 'consensus': "oops"}
        _, stats = parse_all_results({'m3': result}, [seg], {})
        self.assertEqual(stats['malformed'], 1)

    def test_none_result_counted_as_malformed(self):
        seg = make_seg('m4')
        _, stats = parse_all_results({'m4': None}, [seg], {})
        self.assertEqual(stats['malformed'], 1)


# ---------------------------------------------------------------------------
# parse_all_results: unknown segment_id
# ---------------------------------------------------------------------------

class TestParseAllResultsUnknownSegmentId(unittest.TestCase):

    def test_unknown_id_silently_skipped(self):
        seg = make_seg('known')
        result = make_result(primary_stage=1)
        # Pass a result for 'unknown' — should not raise
        _, stats = parse_all_results({'unknown': result}, [seg], {})
        self.assertIsNone(seg.primary_stage)  # 'known' was not updated

    def test_total_counts_all_keys_including_unknown(self):
        seg = make_seg('s1')
        results = {
            's1': make_result(primary_stage=0),
            'ghost': make_result(primary_stage=1),
        }
        _, stats = parse_all_results(results, [seg], {})
        self.assertEqual(stats['total'], 2)
        # Only s1 was updated (ghost has no matching segment)
        self.assertEqual(stats['parsed'], 1)


# ---------------------------------------------------------------------------
# parse_all_results: stats dict shape
# ---------------------------------------------------------------------------

class TestParseAllResultsStatsShape(unittest.TestCase):

    def test_stats_keys_present(self):
        _, stats = parse_all_results({}, [], {})
        for key in ('total', 'parsed', 'abstained', 'split',
                    'all_raters_failed', 'malformed', 'error_breakdown'):
            with self.subTest(key=key):
                self.assertIn(key, stats)

    def test_error_breakdown_is_dict(self):
        _, stats = parse_all_results({}, [], {})
        self.assertIsInstance(stats['error_breakdown'], dict)

    def test_error_breakdown_populated_on_abstain(self):
        seg = make_seg('abs')
        result = make_abstain_result()
        _, stats = parse_all_results({'abs': result}, [seg], {})
        # 'abstain' key should appear in error_breakdown
        self.assertIn('abstain', stats['error_breakdown'])

    def test_agreement_fraction_computed_correctly(self):
        seg = make_seg('s1')
        result = make_result(
            primary_stage=2,
            n_agree=2,
            n_raters=3,
            agreement_level='majority',
        )
        parse_all_results({'s1': result}, [seg], {})
        self.assertAlmostEqual(seg.agreement_fraction, 2 / 3)


# ---------------------------------------------------------------------------
# parse_all_results: multiple segments mixed outcomes
# ---------------------------------------------------------------------------

class TestParseAllResultsMixedBatch(unittest.TestCase):

    def test_mixed_batch_counts(self):
        segs = [make_seg('c'), make_seg('a'), make_seg('s'), make_seg('e')]
        results = {
            'c': make_result(primary_stage=1, agreement_level='unanimous'),
            'a': make_abstain_result(),
            's': make_result(
                primary_stage=None,
                consensus_vote=None,
                agreement_level='split',
                needs_review=True,
                n_agree=1,
                n_raters=3,
            ),
            'e': make_result(
                primary_stage=None,
                consensus_vote=None,
                agreement_level='none',
                needs_review=True,
                n_agree=0,
                n_raters=3,
            ),
        }
        _, stats = parse_all_results(results, segs, {})
        self.assertEqual(stats['parsed'], 1)
        self.assertEqual(stats['abstained'], 1)
        self.assertEqual(stats['split'], 1)
        self.assertEqual(stats['all_raters_failed'], 1)
        self.assertEqual(stats['total'], 4)

    def test_returns_same_segment_list(self):
        segs = [make_seg('s1')]
        result = make_result(primary_stage=2)
        returned_segs, _ = parse_all_results({'s1': result}, segs, {})
        self.assertIs(returned_segs, segs)


# ---------------------------------------------------------------------------
# parse_purer_results: writes purer_* fields
# ---------------------------------------------------------------------------

class TestParsePurerResults(unittest.TestCase):

    def setUp(self):
        self.seg = make_seg('t1', speaker='therapist')
        result = make_result(
            primary_stage=0,  # Phenomenological
            primary_confidence=0.88,
            secondary_stage=4,
            secondary_confidence=0.32,
            justification="Eliciting somatic experience",
            agreement_level='majority',
            n_agree=2,
            n_raters=3,
            n_ballots=3,
            needs_review=False,
        )
        _, self.stats = parse_purer_results({'t1': result}, [self.seg])

    def test_purer_primary_set(self):
        self.assertEqual(self.seg.purer_primary, 0)

    def test_purer_secondary_set(self):
        self.assertEqual(self.seg.purer_secondary, 4)

    def test_purer_confidence_primary_set(self):
        self.assertAlmostEqual(self.seg.purer_confidence_primary, 0.88)

    def test_purer_confidence_secondary_set(self):
        self.assertAlmostEqual(self.seg.purer_confidence_secondary, 0.32)

    def test_purer_justification_set(self):
        self.assertEqual(self.seg.purer_justification, "Eliciting somatic experience")

    def test_purer_agreement_level_set(self):
        self.assertEqual(self.seg.purer_agreement_level, 'majority')

    def test_purer_agreement_fraction_set(self):
        self.assertAlmostEqual(self.seg.purer_agreement_fraction, 2 / 3)

    def test_purer_needs_review_set(self):
        self.assertFalse(self.seg.purer_needs_review)

    def test_purer_run_consistency_is_n_agree(self):
        self.assertEqual(self.seg.purer_run_consistency, 2)

    def test_vaamr_fields_untouched(self):
        """parse_purer_results must not write VAAMR fields."""
        self.assertIsNone(self.seg.primary_stage)
        self.assertIsNone(self.seg.llm_confidence_primary)
        self.assertIsNone(self.seg.llm_justification)

    def test_stats_parsed_count(self):
        self.assertEqual(self.stats['parsed'], 1)


class TestParsePurerResultsAbstain(unittest.TestCase):

    def test_purer_abstain_primary_is_none(self):
        seg = make_seg('t2', speaker='therapist')
        result = make_abstain_result()
        _, stats = parse_purer_results({'t2': result}, [seg])
        self.assertIsNone(seg.purer_primary)
        self.assertEqual(seg.purer_justification, 'not relevant')
        self.assertEqual(stats['abstained'], 1)
        self.assertEqual(stats['parsed'], 0)


class TestParsePurerResultsMalformed(unittest.TestCase):

    def test_non_dict_counted_as_malformed(self):
        seg = make_seg('t3', speaker='therapist')
        _, stats = parse_purer_results({'t3': 42}, [seg])
        self.assertEqual(stats['malformed'], 1)

    def test_missing_consensus_counted_as_malformed(self):
        seg = make_seg('t4', speaker='therapist')
        _, stats = parse_purer_results({'t4': {'rater_ids': []}}, [seg])
        self.assertEqual(stats['malformed'], 1)


class TestParsePurerResultsSplit(unittest.TestCase):

    def test_purer_split_needs_review(self):
        seg = make_seg('t5', speaker='therapist')
        result = make_result(
            primary_stage=None,
            consensus_vote=None,
            agreement_level='split',
            needs_review=True,
            n_agree=1,
            n_raters=3,
        )
        _, stats = parse_purer_results({'t5': result}, [seg])
        self.assertIsNone(seg.purer_primary)
        self.assertTrue(seg.purer_needs_review)
        self.assertEqual(stats['split'], 1)


class TestParsePurerResultsRaterFields(unittest.TestCase):

    def test_purer_rater_ids_set(self):
        seg = make_seg('t6', speaker='therapist')
        result = make_result(rater_ids=['modelX', 'modelY'])
        parse_purer_results({'t6': result}, [seg])
        self.assertEqual(seg.purer_rater_ids, ['modelX', 'modelY'])

    def test_purer_rater_votes_set(self):
        votes = [{'rater': 'modelX', 'vote': 'CODED', 'stage': 2}]
        seg = make_seg('t7', speaker='therapist')
        result = make_result(rater_votes=votes)
        parse_purer_results({'t7': result}, [seg])
        self.assertEqual(seg.purer_rater_votes, votes)


class TestParsePurerResultsStatsShape(unittest.TestCase):

    def test_stats_keys_present(self):
        _, stats = parse_purer_results({}, [])
        for key in ('total', 'parsed', 'abstained', 'split',
                    'all_raters_failed', 'malformed', 'error_breakdown'):
            with self.subTest(key=key):
                self.assertIn(key, stats)

    def test_returns_same_segment_list(self):
        segs = [make_seg('t1', speaker='therapist')]
        result = make_result(primary_stage=1)
        returned_segs, _ = parse_purer_results({'t1': result}, segs)
        self.assertIs(returned_segs, segs)


if __name__ == '__main__':
    unittest.main()
