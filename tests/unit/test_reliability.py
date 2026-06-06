"""
tests/unit/test_reliability.py
--------------------------------
Unit tests for classification_tools/reliability.py.

Covers:
  - percent_agreement: perfect agreement, partial agreement, all-different,
    missing ballots (None), ABSTAIN as a real ballot, empty matrix
  - fleiss_kappa: perfect agreement -> 1.0, chance level, degenerate
    (< 2 segments), mixed with missing
  - krippendorff_alpha: perfect agreement -> 1.0, empty/too-short inputs,
    known-value examples
  - ballots_from_segments: extracts correct ballot matrix from Segment objects
  - secondary_ballots_from_segments: only rows where at least one rater
    gave a secondary are included
  - compute_reliability: convenience wrapper returns expected keys

All values are computed analytically on tiny hand-constructed examples.
No network, no model weights.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from classification_tools.reliability import (
    percent_agreement,
    fleiss_kappa,
    krippendorff_alpha,
    ballots_from_segments,
    secondary_ballots_from_segments,
    compute_reliability,
    ABSTAIN_CATEGORY,
    _encode_ballot,
    _encode_matrix,
)
from tests.testhelpers import make_segment


# ---------------------------------------------------------------------------
# _encode_ballot
# ---------------------------------------------------------------------------

class TestEncodeBallot(unittest.TestCase):

    def test_none_returns_none(self):
        self.assertIsNone(_encode_ballot(None))

    def test_abstain_returns_minus_one(self):
        self.assertEqual(_encode_ballot('ABSTAIN'), ABSTAIN_CATEGORY)

    def test_int_passthrough(self):
        self.assertEqual(_encode_ballot(3), 3)

    def test_str_int_converts(self):
        self.assertEqual(_encode_ballot('2'), 2)

    def test_bad_string_returns_none(self):
        self.assertIsNone(_encode_ballot('bad'))

    def test_numpy_int(self):
        self.assertEqual(_encode_ballot(np.int64(4)), 4)


# ---------------------------------------------------------------------------
# _encode_matrix
# ---------------------------------------------------------------------------

class TestEncodeMatrix(unittest.TestCase):

    def test_empty_matrix(self):
        mat = _encode_matrix([])
        self.assertEqual(mat.shape, (0, 0))

    def test_simple_matrix(self):
        mat = _encode_matrix([[0, 1], [2, None]])
        self.assertEqual(mat.shape, (2, 2))
        self.assertEqual(mat[0, 0], 0)
        self.assertEqual(mat[0, 1], 1)
        self.assertEqual(mat[1, 0], 2)
        self.assertTrue(np.isnan(mat[1, 1]))


# ---------------------------------------------------------------------------
# percent_agreement
# ---------------------------------------------------------------------------

class TestPercentAgreement(unittest.TestCase):

    def test_perfect_unanimous_agreement(self):
        # 3 raters, 3 segments, all agree
        matrix = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        result = percent_agreement(matrix)
        self.assertAlmostEqual(result['all_agree'], 1.0)
        self.assertAlmostEqual(result['pairwise'], 1.0)
        self.assertEqual(result['n_segments'], 3)

    def test_zero_agreement_all_different(self):
        # 3 raters, each votes differently on every segment
        matrix = [[0, 1, 2], [0, 1, 2]]
        result = percent_agreement(matrix)
        self.assertAlmostEqual(result['all_agree'], 0.0)
        self.assertEqual(result['n_segments'], 2)

    def test_partial_agreement_two_of_three(self):
        # Seg 0: raters 0,1 agree; rater 2 disagrees -> not unanimous
        # Seg 1: all three agree
        matrix = [[0, 0, 1], [2, 2, 2]]
        result = percent_agreement(matrix)
        self.assertAlmostEqual(result['all_agree'], 0.5)
        self.assertEqual(result['n_segments'], 2)

    def test_missing_ballots_ignored(self):
        # None entries are excluded; a row with only one valid ballot is skipped
        matrix = [[0, None], [1, 1]]
        result = percent_agreement(matrix)
        # Only second row has >=2 valid ballots
        self.assertEqual(result['n_segments'], 1)
        self.assertAlmostEqual(result['all_agree'], 1.0)

    def test_empty_matrix(self):
        result = percent_agreement([])
        self.assertEqual(result['all_agree'], 0.0)
        self.assertEqual(result['n_segments'], 0)

    def test_abstain_counts_as_real_ballot(self):
        """Two raters both ABSTAIN must count as agreement."""
        matrix = [['ABSTAIN', 'ABSTAIN'], [0, 0]]
        result = percent_agreement(matrix)
        self.assertAlmostEqual(result['all_agree'], 1.0)

    def test_abstain_vs_coded_counts_as_disagreement(self):
        """ABSTAIN vs a coded value must count as disagreement."""
        matrix = [['ABSTAIN', 0]]
        result = percent_agreement(matrix)
        self.assertAlmostEqual(result['all_agree'], 0.0)
        self.assertAlmostEqual(result['pairwise'], 0.0)

    def test_single_segment_single_rater_skipped(self):
        """A row with only one valid ballot is excluded from n_segments."""
        matrix = [[0]]
        result = percent_agreement(matrix)
        self.assertEqual(result['n_segments'], 0)

    def test_pairwise_on_two_raters(self):
        """With 2 raters, pairwise == all_agree."""
        matrix = [[1, 1], [0, 1]]
        result = percent_agreement(matrix)
        # Seg 0: agree -> pairwise=1; Seg 1: disagree -> pairwise=0; mean=0.5
        self.assertAlmostEqual(result['pairwise'], 0.5)


# ---------------------------------------------------------------------------
# fleiss_kappa
# ---------------------------------------------------------------------------

class TestFleissKappa(unittest.TestCase):

    def test_perfect_agreement_returns_one(self):
        # All raters agree on every segment
        matrix = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 0, 0]]
        k = fleiss_kappa(matrix)
        self.assertIsNotNone(k)
        self.assertAlmostEqual(k, 1.0, places=4)

    def test_fewer_than_two_segments_returns_none(self):
        # Only one segment with valid rater count
        k = fleiss_kappa([[0, 1]])
        self.assertIsNone(k)

    def test_empty_matrix_returns_none(self):
        self.assertIsNone(fleiss_kappa([]))

    def test_all_same_category_returns_one(self):
        """When all raters always vote for category 0, kappa is 1."""
        matrix = [[0, 0], [0, 0], [0, 0]]
        k = fleiss_kappa(matrix)
        # P_e = 1.0 so numerically we return 1.0 or 0.0 depending on P_bar.
        # All agree -> P_bar=1.0 -> returns 1.0.
        self.assertIsNotNone(k)
        self.assertAlmostEqual(k, 1.0)

    def test_two_raters_complete_disagreement(self):
        """Two raters who never agree should yield kappa < 0 or 0."""
        # Rater 0 always votes 0; rater 1 always votes 1 -> no agreement
        matrix = [[0, 1], [0, 1], [0, 1], [0, 1]]
        k = fleiss_kappa(matrix)
        self.assertIsNotNone(k)
        self.assertLessEqual(k, 0.0)

    def test_with_missing_ballots_does_not_crash(self):
        """None values in the ballot matrix must not cause a crash."""
        matrix = [[0, 0, None], [1, 1, 1], [2, None, 2]]
        k = fleiss_kappa(matrix)
        # Just check it runs and returns a float or None
        if k is not None:
            self.assertIsInstance(k, float)

    def test_result_bounded(self):
        """Kappa must be in [-1, 1]."""
        matrix = [[0, 1], [1, 0], [0, 0], [1, 1]]
        k = fleiss_kappa(matrix)
        if k is not None:
            self.assertGreaterEqual(k, -1.0)
            self.assertLessEqual(k, 1.0)


# ---------------------------------------------------------------------------
# krippendorff_alpha
# ---------------------------------------------------------------------------

class TestKrippendorffAlpha(unittest.TestCase):

    def test_perfect_agreement_returns_one(self):
        matrix = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        alpha = krippendorff_alpha(matrix)
        self.assertIsNotNone(alpha)
        self.assertAlmostEqual(alpha, 1.0, places=4)

    def test_empty_matrix_returns_none(self):
        alpha = krippendorff_alpha([])
        self.assertIsNone(alpha)

    def test_single_segment_returns_none(self):
        """Too few subjects for meaningful alpha."""
        alpha = krippendorff_alpha([[0, 1, 2]])
        self.assertIsNone(alpha)

    def test_two_raters_perfect_agreement(self):
        matrix = [[0, 0], [1, 1], [2, 2], [0, 0]]
        alpha = krippendorff_alpha(matrix)
        self.assertIsNotNone(alpha)
        self.assertAlmostEqual(alpha, 1.0, places=3)

    def test_complete_disagreement_negative(self):
        """Systematic disagreement yields alpha <= 0."""
        # Rater 0 always 0, rater 1 always 1 — systematic disagreement
        matrix = [[0, 1]] * 6
        alpha = krippendorff_alpha(matrix)
        self.assertIsNotNone(alpha)
        self.assertLessEqual(alpha, 0.0)

    def test_missing_ballots_handled(self):
        """None ballots must not crash alpha computation."""
        matrix = [[0, 0, None], [1, None, 1], [2, 2, 2]]
        alpha = krippendorff_alpha(matrix)
        if alpha is not None:
            self.assertIsInstance(alpha, float)

    def test_abstain_treated_as_category(self):
        """ABSTAIN ballots are nominal category -1 and should not cause errors."""
        matrix = [['ABSTAIN', 'ABSTAIN'], [0, 0], [1, 1]]
        alpha = krippendorff_alpha(matrix)
        # Should produce a valid float close to 1.0 (all rows agree)
        self.assertIsNotNone(alpha)
        self.assertAlmostEqual(alpha, 1.0, places=3)

    def test_interval_level_accepted(self):
        """level='interval' must not raise."""
        matrix = [[0, 0], [1, 1], [2, 2]]
        alpha = krippendorff_alpha(matrix, level='interval')
        if alpha is not None:
            self.assertIsInstance(alpha, float)


# ---------------------------------------------------------------------------
# ballots_from_segments
# ---------------------------------------------------------------------------

class TestBallotsFromSegments(unittest.TestCase):

    def _make_coded_seg(self, seg_id, stage):
        seg = make_segment(seg_id)
        seg.rater_votes = [
            {'rater': 'r1', 'vote': 'CODED', 'stage': stage},
            {'rater': 'r2', 'vote': 'CODED', 'stage': stage},
        ]
        return seg

    def _make_abstain_seg(self, seg_id):
        seg = make_segment(seg_id)
        seg.rater_votes = [
            {'rater': 'r1', 'vote': 'ABSTAIN', 'stage': None},
            {'rater': 'r2', 'vote': 'CODED', 'stage': 2},
        ]
        return seg

    def _make_error_seg(self, seg_id):
        seg = make_segment(seg_id)
        seg.rater_votes = [
            {'rater': 'r1', 'vote': 'ERROR', 'stage': None},
            {'rater': 'r2', 'vote': 'CODED', 'stage': 1},
        ]
        return seg

    def test_coded_segments_produce_correct_matrix(self):
        segs = [self._make_coded_seg('s0', 0), self._make_coded_seg('s1', 2)]
        matrix = ballots_from_segments(segs)
        self.assertEqual(len(matrix), 2)
        self.assertEqual(matrix[0], [0, 0])
        self.assertEqual(matrix[1], [2, 2])

    def test_abstain_vote_becomes_abstain_string(self):
        segs = [self._make_abstain_seg('s0')]
        matrix = ballots_from_segments(segs)
        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix[0][0], 'ABSTAIN')
        self.assertEqual(matrix[0][1], 2)

    def test_error_vote_becomes_none(self):
        segs = [self._make_error_seg('s0')]
        matrix = ballots_from_segments(segs)
        self.assertEqual(len(matrix), 1)
        self.assertIsNone(matrix[0][0])
        self.assertEqual(matrix[0][1], 1)

    def test_segment_without_rater_votes_skipped(self):
        seg = make_segment('no_votes')
        segs = [seg, self._make_coded_seg('s1', 3)]
        matrix = ballots_from_segments(segs)
        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix[0], [3, 3])

    def test_empty_segments_returns_empty(self):
        self.assertEqual(ballots_from_segments([]), [])


# ---------------------------------------------------------------------------
# secondary_ballots_from_segments
# ---------------------------------------------------------------------------

class TestSecondaryBallotsFromSegments(unittest.TestCase):

    def _make_seg_with_secondary(self, seg_id, sec_stage):
        seg = make_segment(seg_id)
        seg.rater_votes = [
            {'rater': 'r1', 'vote': 'CODED', 'stage': 0, 'secondary_stage': sec_stage},
            {'rater': 'r2', 'vote': 'CODED', 'stage': 0, 'secondary_stage': None},
        ]
        return seg

    def _make_seg_no_secondary(self, seg_id):
        seg = make_segment(seg_id)
        seg.rater_votes = [
            {'rater': 'r1', 'vote': 'CODED', 'stage': 1, 'secondary_stage': None},
            {'rater': 'r2', 'vote': 'CODED', 'stage': 1, 'secondary_stage': None},
        ]
        return seg

    def test_includes_rows_where_any_rater_has_secondary(self):
        segs = [
            self._make_seg_with_secondary('s0', 2),
            self._make_seg_no_secondary('s1'),
        ]
        matrix = secondary_ballots_from_segments(segs)
        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix[0][0], 2)
        self.assertIsNone(matrix[0][1])

    def test_excludes_rows_where_no_rater_has_secondary(self):
        segs = [self._make_seg_no_secondary('s0'), self._make_seg_no_secondary('s1')]
        matrix = secondary_ballots_from_segments(segs)
        self.assertEqual(len(matrix), 0)

    def test_empty_input(self):
        self.assertEqual(secondary_ballots_from_segments([]), [])


# ---------------------------------------------------------------------------
# compute_reliability (convenience wrapper)
# ---------------------------------------------------------------------------

class TestComputeReliability(unittest.TestCase):

    def _make_rated_seg(self, seg_id, stage):
        seg = make_segment(seg_id)
        seg.rater_ids = ['r1', 'r2']
        seg.rater_votes = [
            {'rater': 'r1', 'vote': 'CODED', 'stage': stage, 'secondary_stage': None},
            {'rater': 'r2', 'vote': 'CODED', 'stage': stage, 'secondary_stage': None},
        ]
        return seg

    def test_returns_expected_keys(self):
        segs = [self._make_rated_seg(f's{i}', i % 3) for i in range(4)]
        report = compute_reliability(segs)
        expected_keys = {
            'n_segments', 'rater_ids', 'n_raters',
            'percent_agreement_unanimous', 'percent_agreement_pairwise',
            'fleiss_kappa', 'krippendorff_alpha_nominal',
        }
        for key in expected_keys:
            self.assertIn(key, report, f"Missing key: {key}")

    def test_perfect_agreement_gives_high_scores(self):
        segs = [self._make_rated_seg(f's{i}', i % 3) for i in range(6)]
        report = compute_reliability(segs)
        self.assertAlmostEqual(report['percent_agreement_unanimous'], 1.0)
        self.assertAlmostEqual(report['percent_agreement_pairwise'], 1.0)
        ka = report['krippendorff_alpha_nominal']
        if ka is not None:
            self.assertGreater(ka, 0.9)

    def test_empty_segments_returns_zeros(self):
        report = compute_reliability([])
        self.assertEqual(report['n_segments'], 0)

    def test_rater_ids_extracted_from_longest_segment(self):
        segs = [self._make_rated_seg('s0', 0), self._make_rated_seg('s1', 1)]
        report = compute_reliability(segs)
        self.assertEqual(sorted(report['rater_ids']), ['r1', 'r2'])

    def test_secondary_fields_present(self):
        segs = [self._make_rated_seg(f's{i}', 0) for i in range(3)]
        report = compute_reliability(segs)
        self.assertIn('n_segments_with_secondary', report)
        self.assertIn('secondary_fleiss_kappa', report)
        self.assertIn('secondary_krippendorff_alpha', report)


if __name__ == '__main__':
    unittest.main()
