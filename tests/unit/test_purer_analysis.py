"""
tests/test_purer_analysis.py
----------------------------
Comprehensive tests for analysis/purer_analysis.py.

Covers:
  - CueBlock construction (transition type, profile counting, dominant PURER,
    empty detection)
  - compute_cue_block_purer_profiles() — temporal window logic, speaker
    attribution, multi-session handling, edge cases
  - compute_purer_transition_influence() — empty-cue rate math, mediated
    conditional counts, fraction-of-mediated sums, lift matrix arithmetic
  - Speaker attribution invariants (VAAMR labels only from participant segs,
    PURER labels only from therapist segs drive cue profiles)
"""

import os
import sys
import unittest

import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from analysis.purer_analysis import (
    CueBlock,
    compute_cue_block_purer_profiles,
    compute_purer_transition_influence,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_row(
    session_id='s1',
    speaker='participant',
    start_ms=0,
    end_ms=5000,
    final_label=1,
    purer_primary=None,
    cohort_id=1,
    segment_id='seg_001',
):
    return {
        'session_id': session_id,
        'speaker': speaker,
        'start_time_ms': start_ms,
        'end_time_ms': end_ms,
        'final_label': final_label,
        'purer_primary': purer_primary,
        'cohort_id': cohort_id,
        'segment_id': segment_id,
    }


def _df(*rows):
    return pd.DataFrame(rows)


# ── CueBlock construction ──────────────────────────────────────────────────────

class TestCueBlockConstruction(unittest.TestCase):

    def test_empty_purer_labels_is_empty(self):
        cb = CueBlock('s1', 1, 'a', 'b', 1, 2, [])
        self.assertTrue(cb.is_empty)
        self.assertEqual(cb.n_therapist_segs, 0)
        self.assertIsNone(cb.dominant_purer)
        self.assertEqual(cb.purer_profile, {})

    def test_all_none_labels_not_empty_but_no_dominant(self):
        # Therapist segs present but all without PURER labels
        cb = CueBlock('s1', 1, 'a', 'b', 1, 2, [None, None, None])
        self.assertFalse(cb.is_empty)
        self.assertEqual(cb.n_therapist_segs, 3)
        self.assertIsNone(cb.dominant_purer)
        self.assertEqual(cb.purer_profile, {})

    def test_purer_profile_counts_correctly(self):
        cb = CueBlock('s1', 1, 'a', 'b', 1, 2, [3, 3, 0, 3])
        self.assertEqual(cb.purer_profile, {3: 3, 0: 1})
        self.assertEqual(cb.dominant_purer, 3)

    def test_dominant_purer_is_most_frequent(self):
        cb = CueBlock('s1', 1, 'a', 'b', 1, 2, [0, 1, 0, 2, 0])
        self.assertEqual(cb.dominant_purer, 0)

    def test_single_purer_label(self):
        cb = CueBlock('s1', 1, 'a', 'b', 1, 2, [4])
        self.assertEqual(cb.dominant_purer, 4)
        self.assertFalse(cb.is_empty)

    def test_transition_type_forward(self):
        cb = CueBlock('s1', 1, 'a', 'b', 0, 3, [])
        self.assertEqual(cb.transition_type, 'forward')

    def test_transition_type_backward(self):
        cb = CueBlock('s1', 1, 'a', 'b', 3, 1, [])
        self.assertEqual(cb.transition_type, 'backward')

    def test_transition_type_lateral(self):
        cb = CueBlock('s1', 1, 'a', 'b', 2, 2, [])
        self.assertEqual(cb.transition_type, 'lateral')

    def test_mixed_none_and_valid_labels(self):
        cb = CueBlock('s1', 1, 'a', 'b', 0, 1, [None, 2, None, 2, 1])
        # only non-None labels counted
        self.assertEqual(cb.purer_profile, {2: 2, 1: 1})
        self.assertEqual(cb.dominant_purer, 2)
        self.assertFalse(cb.is_empty)
        self.assertEqual(cb.n_therapist_segs, 5)

    def test_cohort_id_stored(self):
        cb = CueBlock('s1', 7, 'a', 'b', 0, 1, [])
        self.assertEqual(cb.cohort_id, 7)

    def test_segment_ids_stored(self):
        cb = CueBlock('s1', 1, 'seg_from', 'seg_to', 0, 1, [])
        self.assertEqual(cb.from_seg_id, 'seg_from')
        self.assertEqual(cb.to_seg_id, 'seg_to')


# ── compute_cue_block_purer_profiles ──────────────────────────────────────────

class TestComputeCueBlockPurerProfiles(unittest.TestCase):

    def test_empty_dataframe_returns_empty_list(self):
        df = pd.DataFrame(columns=[
            'session_id', 'speaker', 'final_label',
            'start_time_ms', 'end_time_ms', 'segment_id',
        ])
        result = compute_cue_block_purer_profiles(df)
        self.assertEqual(result, [])

    def test_missing_required_column_returns_empty_list(self):
        # Missing 'end_time_ms'
        df = _df(
            _make_row(end_ms=5000),
            _make_row(start_ms=6000, end_ms=10000),
        )
        df = df.drop(columns=['end_time_ms'])
        result = compute_cue_block_purer_profiles(df)
        self.assertEqual(result, [])

    def test_session_with_one_participant_segment_produces_no_blocks(self):
        df = _df(
            _make_row(session_id='s1', speaker='participant',
                      start_ms=0, end_ms=5000, final_label=1, segment_id='p1'),
            _make_row(session_id='s1', speaker='therapist',
                      start_ms=5000, end_ms=8000, final_label=None,
                      purer_primary=0, segment_id='t1'),
        )
        result = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(result), 0)

    def test_unlabeled_participant_segment_not_counted(self):
        # Two participant segments but only one has final_label
        df = _df(
            _make_row(session_id='s1', speaker='participant',
                      start_ms=0, end_ms=5000, final_label=None, segment_id='p1'),
            _make_row(session_id='s1', speaker='participant',
                      start_ms=8000, end_ms=12000, final_label=2, segment_id='p2'),
        )
        result = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(result), 0)

    def test_two_participant_segments_produces_one_block(self):
        df = _df(
            _make_row(session_id='s1', speaker='participant',
                      start_ms=0, end_ms=5000, final_label=1, segment_id='p1'),
            _make_row(session_id='s1', speaker='participant',
                      start_ms=8000, end_ms=12000, final_label=2, segment_id='p2'),
        )
        result = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].from_stage, 1)
        self.assertEqual(result[0].to_stage, 2)

    def test_therapist_in_window_included_in_block(self):
        # Participant: 0-5000ms, Therapist: 5000-7000ms, Participant: 7000-10000ms
        df = _df(
            _make_row(session_id='s1', speaker='participant',
                      start_ms=0, end_ms=5000, final_label=0, segment_id='p1'),
            _make_row(session_id='s1', speaker='therapist',
                      start_ms=5000, end_ms=7000, final_label=None,
                      purer_primary=3, segment_id='t1'),
            _make_row(session_id='s1', speaker='participant',
                      start_ms=7000, end_ms=10000, final_label=2, segment_id='p2'),
        )
        result = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(result), 1)
        block = result[0]
        self.assertFalse(block.is_empty)
        self.assertEqual(block.n_therapist_segs, 1)
        self.assertEqual(block.purer_profile, {3: 1})
        self.assertEqual(block.dominant_purer, 3)

    def test_therapist_ending_before_window_excluded(self):
        # Therapist ends at 4000ms, window starts at 5000ms → NOT in window
        df = _df(
            _make_row(session_id='s1', speaker='participant',
                      start_ms=0, end_ms=5000, final_label=0, segment_id='p1'),
            _make_row(session_id='s1', speaker='therapist',
                      start_ms=2000, end_ms=4000, final_label=None,
                      purer_primary=1, segment_id='t1'),
            _make_row(session_id='s1', speaker='participant',
                      start_ms=8000, end_ms=12000, final_label=1, segment_id='p2'),
        )
        result = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(result), 1)
        block = result[0]
        # Therapist end_time_ms=4000 is NOT > from_end_ms=5000 → excluded
        self.assertTrue(block.is_empty)
        self.assertEqual(block.n_therapist_segs, 0)

    def test_therapist_starting_after_window_excluded(self):
        # Therapist starts at 9000ms, window ends at 8000ms → NOT in window
        df = _df(
            _make_row(session_id='s1', speaker='participant',
                      start_ms=0, end_ms=5000, final_label=0, segment_id='p1'),
            _make_row(session_id='s1', speaker='therapist',
                      start_ms=9000, end_ms=11000, final_label=None,
                      purer_primary=2, segment_id='t1'),
            _make_row(session_id='s1', speaker='participant',
                      start_ms=8000, end_ms=12000, final_label=1, segment_id='p2'),
        )
        result = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(result), 1)
        block = result[0]
        # Therapist start_time_ms=9000 is NOT < to_start_ms=8000 → excluded
        self.assertTrue(block.is_empty)

    def test_therapist_partially_overlapping_window_left_included(self):
        # Therapist starts before window but ends within it
        # Window: from_end_ms=5000 to to_start_ms=8000
        # Therapist: 3000-6000ms → overlaps (6000 > 5000 AND 3000 < 8000)
        df = _df(
            _make_row(session_id='s1', speaker='participant',
                      start_ms=0, end_ms=5000, final_label=0, segment_id='p1'),
            _make_row(session_id='s1', speaker='therapist',
                      start_ms=3000, end_ms=6000, final_label=None,
                      purer_primary=4, segment_id='t1'),
            _make_row(session_id='s1', speaker='participant',
                      start_ms=8000, end_ms=12000, final_label=2, segment_id='p2'),
        )
        result = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0].is_empty)
        self.assertEqual(result[0].purer_profile, {4: 1})

    def test_adjacent_participant_segs_no_gap_creates_empty_block(self):
        # from_end_ms == to_start_ms → to_start_ms NOT > from_end_ms → empty block
        df = _df(
            _make_row(session_id='s1', speaker='participant',
                      start_ms=0, end_ms=5000, final_label=0, segment_id='p1'),
            _make_row(session_id='s1', speaker='therapist',
                      start_ms=5000, end_ms=7000, final_label=None,
                      purer_primary=0, segment_id='t1'),
            _make_row(session_id='s1', speaker='participant',
                      start_ms=5000, end_ms=9000, final_label=1, segment_id='p2'),
        )
        result = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(result), 1)
        # to_start_ms=5000 == from_end_ms=5000 → condition `to_start_ms > from_end_ms` is False
        self.assertTrue(result[0].is_empty)

    def test_multiple_therapist_segs_all_in_window(self):
        df = _df(
            _make_row(session_id='s1', speaker='participant',
                      start_ms=0, end_ms=3000, final_label=0, segment_id='p1'),
            _make_row(session_id='s1', speaker='therapist',
                      start_ms=3000, end_ms=5000, final_label=None,
                      purer_primary=0, segment_id='t1'),
            _make_row(session_id='s1', speaker='therapist',
                      start_ms=5000, end_ms=7000, final_label=None,
                      purer_primary=0, segment_id='t2'),
            _make_row(session_id='s1', speaker='therapist',
                      start_ms=7000, end_ms=9000, final_label=None,
                      purer_primary=3, segment_id='t3'),
            _make_row(session_id='s1', speaker='participant',
                      start_ms=10000, end_ms=15000, final_label=1, segment_id='p2'),
        )
        result = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(result), 1)
        block = result[0]
        self.assertEqual(block.n_therapist_segs, 3)
        self.assertEqual(block.purer_profile, {0: 2, 3: 1})
        self.assertEqual(block.dominant_purer, 0)

    def test_multiple_sessions_correct_block_count(self):
        rows = []
        # Session s1: 2 participant segs → 1 block
        rows.append(_make_row(session_id='s1', speaker='participant',
                               start_ms=0, end_ms=5000, final_label=0, segment_id='s1p1'))
        rows.append(_make_row(session_id='s1', speaker='participant',
                               start_ms=6000, end_ms=10000, final_label=1, segment_id='s1p2'))
        # Session s2: 3 participant segs → 2 blocks
        rows.append(_make_row(session_id='s2', speaker='participant',
                               start_ms=0, end_ms=4000, final_label=2, segment_id='s2p1'))
        rows.append(_make_row(session_id='s2', speaker='participant',
                               start_ms=5000, end_ms=8000, final_label=3, segment_id='s2p2'))
        rows.append(_make_row(session_id='s2', speaker='participant',
                               start_ms=9000, end_ms=12000, final_label=4, segment_id='s2p3'))
        df = pd.DataFrame(rows)
        result = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(result), 3)
        s1_blocks = [cb for cb in result if cb.session_id == 's1']
        s2_blocks = [cb for cb in result if cb.session_id == 's2']
        self.assertEqual(len(s1_blocks), 1)
        self.assertEqual(len(s2_blocks), 2)

    def test_blocks_not_crossed_between_sessions(self):
        # Therapist from s2 must not appear in s1's cue block
        rows = [
            _make_row(session_id='s1', speaker='participant',
                      start_ms=0, end_ms=5000, final_label=0, segment_id='s1p1'),
            _make_row(session_id='s1', speaker='participant',
                      start_ms=7000, end_ms=10000, final_label=1, segment_id='s1p2'),
            # Therapist claimed to be in s2 but has same timestamps as s1 window
            _make_row(session_id='s2', speaker='therapist',
                      start_ms=5000, end_ms=7000, final_label=None,
                      purer_primary=2, segment_id='s2t1'),
        ]
        df = pd.DataFrame(rows)
        result = compute_cue_block_purer_profiles(df)
        # s1 block must be empty (no s1 therapist segs)
        s1_blocks = [cb for cb in result if cb.session_id == 's1']
        self.assertEqual(len(s1_blocks), 1)
        self.assertTrue(s1_blocks[0].is_empty)

    def test_no_purer_column_builds_blocks_with_none_labels(self):
        df = _df(
            _make_row(session_id='s1', speaker='participant',
                      start_ms=0, end_ms=5000, final_label=0, segment_id='p1'),
            _make_row(session_id='s1', speaker='therapist',
                      start_ms=5000, end_ms=7000, final_label=None,
                      purer_primary=None, segment_id='t1'),
            _make_row(session_id='s1', speaker='participant',
                      start_ms=8000, end_ms=12000, final_label=2, segment_id='p2'),
        )
        df = df.drop(columns=['purer_primary'])
        result = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(result), 1)
        block = result[0]
        # Therapist present but no PURER column → n_therapist_segs=1, dominant=None
        self.assertFalse(block.is_empty)
        self.assertEqual(block.n_therapist_segs, 1)
        self.assertIsNone(block.dominant_purer)

    def test_from_to_stage_attribution_from_participant_rows_only(self):
        # Therapist has a final_label value — it must be ignored
        df = _df(
            _make_row(session_id='s1', speaker='participant',
                      start_ms=0, end_ms=5000, final_label=0, segment_id='p1'),
            _make_row(session_id='s1', speaker='therapist',
                      start_ms=5000, end_ms=7000, final_label=99,  # must be ignored
                      purer_primary=1, segment_id='t1'),
            _make_row(session_id='s1', speaker='participant',
                      start_ms=7000, end_ms=10000, final_label=4, segment_id='p2'),
        )
        result = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(result), 1)
        block = result[0]
        self.assertEqual(block.from_stage, 0)
        self.assertEqual(block.to_stage, 4)

    def test_cohort_id_extracted_from_session(self):
        df = _df(
            _make_row(session_id='s1', speaker='participant',
                      start_ms=0, end_ms=5000, final_label=0,
                      cohort_id=3, segment_id='p1'),
            _make_row(session_id='s1', speaker='participant',
                      start_ms=6000, end_ms=9000, final_label=1,
                      cohort_id=3, segment_id='p2'),
        )
        result = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].cohort_id, 3)

    def test_five_participant_segs_four_blocks(self):
        rows = []
        stages = [0, 1, 2, 3, 4]
        for i, stage in enumerate(stages):
            rows.append(_make_row(
                session_id='s1', speaker='participant',
                start_ms=i * 10000, end_ms=i * 10000 + 5000,
                final_label=stage, segment_id=f'p{i}',
            ))
        df = pd.DataFrame(rows)
        result = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(result), 4)
        for i, block in enumerate(result):
            self.assertEqual(block.from_stage, stages[i])
            self.assertEqual(block.to_stage, stages[i + 1])
            self.assertEqual(block.transition_type, 'forward')


# ── compute_purer_transition_influence ────────────────────────────────────────

class TestComputePurerTransitionInfluence(unittest.TestCase):

    def test_empty_cue_blocks_returns_empty_dfs(self):
        result = compute_purer_transition_influence([])
        self.assertTrue(result['transition_profiles'].empty)
        self.assertTrue(result['lift_matrix'].empty)
        self.assertTrue(result['empty_cue_rates'].empty)
        self.assertTrue(result['conditional_table'].empty)
        self.assertEqual(result['raw_cue_blocks'], [])

    def _make_block(self, from_stage, to_stage, purer_labels, session='s1'):
        return CueBlock(session, 1, 'a', 'b', from_stage, to_stage, purer_labels)

    def test_all_empty_blocks_produces_empty_cue_rates_but_no_mediated(self):
        blocks = [
            self._make_block(0, 1, []),
            self._make_block(0, 1, []),
            self._make_block(1, 2, []),
        ]
        result = compute_purer_transition_influence(blocks)
        ecr = result['empty_cue_rates']
        self.assertFalse(ecr.empty)
        # All blocks are empty
        self.assertEqual(ecr.loc[ecr['from_stage'] == 0, 'empty_cue_blocks'].iloc[0], 2)
        self.assertEqual(ecr.loc[ecr['from_stage'] == 0, 'total_cue_blocks'].iloc[0], 2)
        self.assertEqual(
            ecr.loc[ecr['from_stage'] == 0, 'empty_fraction'].iloc[0], 1.0
        )
        # No mediated blocks → transition profiles and lift matrix empty
        self.assertTrue(result['transition_profiles'].empty)
        self.assertTrue(result['lift_matrix'].empty)

    def test_empty_cue_rate_totals_match_block_counts(self):
        blocks = [
            self._make_block(0, 1, [0]),    # mediated
            self._make_block(0, 1, []),      # empty
            self._make_block(0, 1, [1]),    # mediated
            self._make_block(1, 2, []),      # empty
        ]
        result = compute_purer_transition_influence(blocks)
        ecr = result['empty_cue_rates']
        row_01 = ecr[(ecr['from_stage'] == 0) & (ecr['to_stage'] == 1)].iloc[0]
        self.assertEqual(row_01['total_cue_blocks'], 3)
        self.assertEqual(row_01['empty_cue_blocks'], 1)
        self.assertAlmostEqual(row_01['empty_fraction'], round(1 / 3, 3))

        row_12 = ecr[(ecr['from_stage'] == 1) & (ecr['to_stage'] == 2)].iloc[0]
        self.assertEqual(row_12['total_cue_blocks'], 1)
        self.assertEqual(row_12['empty_cue_blocks'], 1)
        self.assertAlmostEqual(row_12['empty_fraction'], 1.0)

    def test_fraction_of_mediated_sums_to_one_per_transition(self):
        blocks = [
            self._make_block(0, 1, [0]),
            self._make_block(0, 1, [1]),
            self._make_block(0, 1, [0]),
            self._make_block(0, 1, [2]),
            self._make_block(1, 2, [3]),
            self._make_block(1, 2, [3]),
        ]
        result = compute_purer_transition_influence(blocks)
        ct = result['conditional_table']
        for (fs, ts), grp in ct.groupby(['from_stage', 'to_stage']):
            total_frac = grp['fraction_of_mediated'].sum()
            self.assertAlmostEqual(total_frac, 1.0, places=2,
                msg=f'Fractions for ({fs}→{ts}) sum to {total_frac}, expected 1.0')

    def test_fraction_of_mediated_correct_values(self):
        # 2x PURER=0, 1x PURER=1 for transition 0→1
        blocks = [
            self._make_block(0, 1, [0]),
            self._make_block(0, 1, [0]),
            self._make_block(0, 1, [1]),
        ]
        result = compute_purer_transition_influence(blocks)
        ct = result['conditional_table']
        row_p0 = ct[(ct['from_stage'] == 0) & (ct['to_stage'] == 1) & (ct['dominant_purer'] == 0)]
        self.assertEqual(len(row_p0), 1)
        self.assertAlmostEqual(row_p0.iloc[0]['fraction_of_mediated'], round(2 / 3, 3))
        row_p1 = ct[(ct['from_stage'] == 0) & (ct['to_stage'] == 1) & (ct['dominant_purer'] == 1)]
        self.assertAlmostEqual(row_p1.iloc[0]['fraction_of_mediated'], round(1 / 3, 3))

    def test_mediated_total_column_is_correct(self):
        blocks = [
            self._make_block(0, 1, [2]),
            self._make_block(0, 1, [2]),
            self._make_block(0, 1, [3]),
            self._make_block(0, 1, []),    # empty — not counted in mediated
        ]
        result = compute_purer_transition_influence(blocks)
        ct = result['conditional_table']
        # mediated blocks for 0→1 = 3
        self.assertTrue((ct['mediated_total'] == 3).all())

    def test_lift_matrix_arithmetic(self):
        """
        Manual calculation:
          3 mediated blocks:
            PURER=P(0) → to_stage=0
            PURER=P(0) → to_stage=1
            PURER=E(3) → to_stage=0

          Base rates:  P(to_stage=0) = 2/3, P(to_stage=1) = 1/3
          P(to=0 | P) = 1/2 → lift(P→0) = (1/2)/(2/3) = 0.75
          P(to=1 | P) = 1/2 → lift(P→1) = (1/2)/(1/3) = 1.50
          P(to=0 | E) = 1/1 → lift(E→0) = (1/1)/(2/3) = 1.50
          P(to=1 | E) = 0/1 → lift(E→1) = 0.0
        """
        blocks = [
            self._make_block(0, 0, [0]),   # PURER=0(P), to_stage=0
            self._make_block(0, 1, [0]),   # PURER=0(P), to_stage=1
            self._make_block(1, 0, [3]),   # PURER=3(E), to_stage=0
        ]
        result = compute_purer_transition_influence(blocks)
        lm = result['lift_matrix']
        self.assertFalse(lm.empty)

        row_p = lm[lm['purer_construct'] == 'Phenomenology'].iloc[0]
        row_e = lm[lm['purer_construct'] == 'Education'].iloc[0]

        self.assertAlmostEqual(float(row_p['lift_to_0']), 0.75, places=1)
        self.assertAlmostEqual(float(row_p['lift_to_1']), 1.5, places=1)
        self.assertAlmostEqual(float(row_e['lift_to_0']), 1.5, places=1)
        self.assertAlmostEqual(float(row_e['lift_to_1']), 0.0, places=1)

    def test_lift_equals_one_when_uniform_distribution(self):
        """
        When PURER=X always leads to the same distribution as the overall base rate,
        lift should be 1.0.
        Set up 4 blocks with PURER=0 → 2x to_stage=0, 2x to_stage=1.
        P(to=0) base = 2/4 = 0.5, P(to=1) base = 2/4 = 0.5
        P(to=0 | P) = 2/4 = 0.5 → lift = 1.0
        P(to=1 | P) = 2/4 = 0.5 → lift = 1.0
        """
        blocks = [
            self._make_block(0, 0, [0]),
            self._make_block(0, 0, [0]),
            self._make_block(0, 1, [0]),
            self._make_block(0, 1, [0]),
        ]
        result = compute_purer_transition_influence(blocks)
        lm = result['lift_matrix']
        row = lm[lm['purer_construct'] == 'Phenomenology'].iloc[0]
        self.assertAlmostEqual(float(row['lift_to_0']), 1.0, places=2)
        self.assertAlmostEqual(float(row['lift_to_1']), 1.0, places=2)

    def test_lift_zero_when_purer_never_leads_to_stage(self):
        """PURER=0 only leads to to_stage=1; lift for to_stage=0 should be 0.0."""
        blocks = [
            self._make_block(0, 1, [0]),   # P→stage1
            self._make_block(0, 1, [0]),   # P→stage1
            self._make_block(0, 0, [3]),   # E→stage0
        ]
        result = compute_purer_transition_influence(blocks)
        lm = result['lift_matrix']
        row_p = lm[lm['purer_construct'] == 'Phenomenology'].iloc[0]
        self.assertAlmostEqual(float(row_p['lift_to_0']), 0.0, places=2)

    def test_n_blocks_column_in_lift_matrix(self):
        blocks = [
            self._make_block(0, 1, [0]),
            self._make_block(0, 2, [0]),
            self._make_block(1, 2, [1]),
        ]
        result = compute_purer_transition_influence(blocks)
        lm = result['lift_matrix']
        row_p = lm[lm['purer_construct'] == 'Phenomenology'].iloc[0]
        row_u = lm[lm['purer_construct'] == 'Utilization'].iloc[0]
        self.assertEqual(int(row_p['n_blocks']), 2)
        self.assertEqual(int(row_u['n_blocks']), 1)

    def test_blocks_with_none_dominant_excluded_from_mediated(self):
        # Blocks with therapist segs but all-None PURER labels have dominant_purer=None
        # They are NOT mediated in the conditional/lift sense
        none_block = CueBlock('s1', 1, 'a', 'b', 0, 1, [None, None])
        good_block = self._make_block(0, 1, [2])
        result = compute_purer_transition_influence([none_block, good_block])
        # Only good_block is mediated (has dominant_purer)
        ct = result['conditional_table']
        self.assertEqual(len(ct), 1)
        self.assertEqual(ct.iloc[0]['dominant_purer'], 2)
        # But empty_cue_rates counts both
        ecr = result['empty_cue_rates']
        row = ecr[(ecr['from_stage'] == 0) & (ecr['to_stage'] == 1)].iloc[0]
        self.assertEqual(row['total_cue_blocks'], 2)
        # none_block is NOT is_empty (therapist segs present) → empty count=0
        self.assertEqual(row['empty_cue_blocks'], 0)

    def test_empty_cue_fraction_zero_when_no_empty_blocks(self):
        blocks = [
            self._make_block(0, 1, [0]),
            self._make_block(0, 1, [1]),
        ]
        result = compute_purer_transition_influence(blocks)
        ecr = result['empty_cue_rates']
        row = ecr[(ecr['from_stage'] == 0) & (ecr['to_stage'] == 1)].iloc[0]
        self.assertEqual(row['empty_cue_blocks'], 0)
        self.assertAlmostEqual(row['empty_fraction'], 0.0)

    def test_count_in_conditional_table_correct(self):
        blocks = [
            self._make_block(0, 1, [4]),
            self._make_block(0, 1, [4]),
            self._make_block(0, 1, [4]),
            self._make_block(0, 1, [0]),
        ]
        result = compute_purer_transition_influence(blocks)
        ct = result['conditional_table']
        row_r2 = ct[(ct['dominant_purer'] == 4)].iloc[0]
        self.assertEqual(row_r2['count'], 3)
        row_p = ct[(ct['dominant_purer'] == 0)].iloc[0]
        self.assertEqual(row_p['count'], 1)

    def test_purer_names_populated_in_conditional_table(self):
        blocks = [self._make_block(0, 1, [0])]
        result = compute_purer_transition_influence(blocks)
        ct = result['conditional_table']
        self.assertEqual(ct.iloc[0]['purer_name'], 'Phenomenology')
        self.assertEqual(ct.iloc[0]['purer_short'], 'P')

    def test_raw_cue_blocks_returned_unchanged(self):
        blocks = [
            self._make_block(0, 1, [0]),
            self._make_block(1, 2, []),
        ]
        result = compute_purer_transition_influence(blocks)
        self.assertIs(result['raw_cue_blocks'], blocks)


# ── Speaker attribution invariants ────────────────────────────────────────────

class TestSpeakerAttributionInvariants(unittest.TestCase):
    """Ensure VAAMR stages are sourced from participant segments only,
    and PURER profiles are sourced from therapist segments only."""

    def _make_block(self, from_stage, to_stage, purer_labels):
        return CueBlock('s1', 1, 'a', 'b', from_stage, to_stage, purer_labels)

    def test_participant_rows_define_from_to_stage(self):
        df = pd.DataFrame([
            _make_row(session_id='s1', speaker='participant',
                      start_ms=0, end_ms=5000, final_label=0, segment_id='p1'),
            _make_row(session_id='s1', speaker='participant',
                      start_ms=8000, end_ms=12000, final_label=3, segment_id='p2'),
            # therapist with final_label — must be ignored for from/to stage
            _make_row(session_id='s1', speaker='therapist',
                      start_ms=5000, end_ms=8000, final_label=99,
                      purer_primary=2, segment_id='t1'),
        ])
        blocks = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].from_stage, 0)
        self.assertEqual(blocks[0].to_stage, 3)

    def test_purer_profile_sourced_exclusively_from_therapist_rows(self):
        df = pd.DataFrame([
            _make_row(session_id='s1', speaker='participant',
                      start_ms=0, end_ms=5000, final_label=0,
                      purer_primary=1,   # participant has purer_primary — must be ignored
                      segment_id='p1'),
            _make_row(session_id='s1', speaker='therapist',
                      start_ms=5000, end_ms=8000, final_label=None,
                      purer_primary=3, segment_id='t1'),
            _make_row(session_id='s1', speaker='participant',
                      start_ms=8000, end_ms=12000, final_label=2,
                      purer_primary=4,  # participant has purer_primary — must be ignored
                      segment_id='p2'),
        ])
        blocks = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(blocks), 1)
        # Only therapist PURER label (3) should appear in profile
        self.assertEqual(blocks[0].purer_profile, {3: 1})

    def test_empty_session_with_only_therapist_rows_no_blocks(self):
        df = pd.DataFrame([
            _make_row(session_id='s1', speaker='therapist',
                      start_ms=0, end_ms=5000, final_label=None,
                      purer_primary=0, segment_id='t1'),
            _make_row(session_id='s1', speaker='therapist',
                      start_ms=5000, end_ms=9000, final_label=None,
                      purer_primary=1, segment_id='t2'),
        ])
        blocks = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(blocks), 0)

    def test_no_therapist_speech_all_blocks_empty(self):
        """All transitions spontaneous — no therapist speech at all."""
        rows = []
        for i in range(4):
            rows.append(_make_row(
                session_id='s1', speaker='participant',
                start_ms=i * 10000, end_ms=i * 10000 + 5000,
                final_label=i, segment_id=f'p{i}',
            ))
        df = pd.DataFrame(rows)
        blocks = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(blocks), 3)
        for block in blocks:
            self.assertTrue(block.is_empty)
            self.assertEqual(block.n_therapist_segs, 0)


if __name__ == '__main__':
    unittest.main()
