"""
tests/test_cue_collection.py
----------------------------
Tests for analysis/reports/_formatting.py helpers:
  - _collect_therapist_cue() — temporal window logic, speaker filtering,
    PURER annotation, session scoping
  - _collect_cue_block_purer_profile() — primary/secondary counting,
    temporal overlap, missing-column guard
  - _format_purer_profile() — display string formatting

These are the low-level functions that both the transition report and the
purer_analysis module rely on. "Empty cues" are most often caused by edge
cases in the temporal overlap logic tested here.
"""

import os
import sys
import unittest

import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from analysis.reports._formatting import (
    _collect_therapist_cue,
    _collect_cue_block_purer_profile,
    _format_purer_profile,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _row(
    session_id='s1',
    speaker='therapist',
    start_ms=5000,
    end_ms=8000,
    text='Therapist said something.',
    purer_primary=None,
    purer_secondary=None,
):
    return {
        'session_id': session_id,
        'speaker': speaker,
        'start_time_ms': start_ms,
        'end_time_ms': end_ms,
        'text': text,
        'purer_primary': purer_primary,
        'purer_secondary': purer_secondary,
    }


def _df(*rows):
    return pd.DataFrame(rows)


# ── _collect_therapist_cue ────────────────────────────────────────────────────

class TestCollectTherapistCue(unittest.TestCase):

    def _df_with_therapist(self, start_ms=5000, end_ms=8000,
                            text='Therapist text.', session='s1',
                            purer_primary=None):
        return _df(_row(session_id=session, speaker='therapist',
                        start_ms=start_ms, end_ms=end_ms,
                        text=text, purer_primary=purer_primary))

    # ── Early-exit guards ────────────────────────────────────────────────────

    def test_returns_empty_when_to_start_equals_from_end(self):
        df = self._df_with_therapist(start_ms=5000, end_ms=8000)
        result = _collect_therapist_cue(df, 's1', from_end_ms=8000, to_start_ms=8000)
        self.assertEqual(result, '')

    def test_returns_empty_when_to_start_less_than_from_end(self):
        df = self._df_with_therapist(start_ms=5000, end_ms=8000)
        result = _collect_therapist_cue(df, 's1', from_end_ms=9000, to_start_ms=7000)
        self.assertEqual(result, '')

    def test_returns_empty_when_from_end_ms_is_none(self):
        df = self._df_with_therapist()
        result = _collect_therapist_cue(df, 's1', from_end_ms=None, to_start_ms=8000)
        self.assertEqual(result, '')

    def test_returns_empty_when_to_start_ms_is_none(self):
        df = self._df_with_therapist()
        result = _collect_therapist_cue(df, 's1', from_end_ms=5000, to_start_ms=None)
        self.assertEqual(result, '')

    def test_returns_empty_when_speaker_column_missing(self):
        df = _df(_row()).drop(columns=['speaker'])
        result = _collect_therapist_cue(df, 's1', from_end_ms=0, to_start_ms=10000)
        self.assertEqual(result, '')

    def test_returns_empty_when_start_time_ms_column_missing(self):
        df = _df(_row()).drop(columns=['start_time_ms'])
        result = _collect_therapist_cue(df, 's1', from_end_ms=0, to_start_ms=10000)
        self.assertEqual(result, '')

    # ── Session scoping ──────────────────────────────────────────────────────

    def test_returns_empty_for_wrong_session(self):
        df = _df(_row(session_id='s2', speaker='therapist',
                      start_ms=5000, end_ms=8000, text='Other session.'))
        result = _collect_therapist_cue(df, 's1', from_end_ms=4000, to_start_ms=9000)
        self.assertEqual(result, '')

    def test_only_collects_from_specified_session(self):
        df = _df(
            _row(session_id='s1', speaker='therapist', start_ms=5000, end_ms=8000,
                 text='Session 1 therapist.'),
            _row(session_id='s2', speaker='therapist', start_ms=5000, end_ms=8000,
                 text='Session 2 therapist.'),
        )
        result = _collect_therapist_cue(df, 's1', from_end_ms=4000, to_start_ms=9000)
        self.assertIn('Session 1 therapist.', result)
        self.assertNotIn('Session 2 therapist.', result)

    # ── Temporal overlap logic ───────────────────────────────────────────────
    # Overlap condition: start_time_ms < to_start_ms AND end_time_ms > from_end_ms

    def test_therapist_fully_within_window_included(self):
        df = self._df_with_therapist(start_ms=6000, end_ms=7000)
        result = _collect_therapist_cue(df, 's1', from_end_ms=5000, to_start_ms=8000)
        self.assertNotEqual(result, '')
        self.assertIn('Therapist text.', result)

    def test_therapist_spanning_entire_window_included(self):
        # start < from_end... wait, overlap: start < to_start AND end > from_end
        # Therapist: 3000-10000, window: 5000-8000
        df = self._df_with_therapist(start_ms=3000, end_ms=10000)
        result = _collect_therapist_cue(df, 's1', from_end_ms=5000, to_start_ms=8000)
        self.assertNotEqual(result, '')

    def test_therapist_overlapping_left_edge_included(self):
        # Therapist: 3000-6000, window: 5000-8000
        # end(6000) > from_end(5000) ✓ AND start(3000) < to_start(8000) ✓ → included
        df = self._df_with_therapist(start_ms=3000, end_ms=6000)
        result = _collect_therapist_cue(df, 's1', from_end_ms=5000, to_start_ms=8000)
        self.assertNotEqual(result, '')

    def test_therapist_overlapping_right_edge_included(self):
        # Therapist: 7000-10000, window: 5000-8000
        # end(10000) > from_end(5000) ✓ AND start(7000) < to_start(8000) ✓ → included
        df = self._df_with_therapist(start_ms=7000, end_ms=10000)
        result = _collect_therapist_cue(df, 's1', from_end_ms=5000, to_start_ms=8000)
        self.assertNotEqual(result, '')

    def test_therapist_ending_exactly_at_from_end_excluded(self):
        # Therapist: 2000-5000, window: 5000-8000
        # end(5000) NOT > from_end(5000) → excluded
        df = self._df_with_therapist(start_ms=2000, end_ms=5000)
        result = _collect_therapist_cue(df, 's1', from_end_ms=5000, to_start_ms=8000)
        self.assertEqual(result, '')

    def test_therapist_starting_exactly_at_to_start_excluded(self):
        # Therapist: 8000-10000, window: 5000-8000
        # start(8000) NOT < to_start(8000) → excluded
        df = self._df_with_therapist(start_ms=8000, end_ms=10000)
        result = _collect_therapist_cue(df, 's1', from_end_ms=5000, to_start_ms=8000)
        self.assertEqual(result, '')

    def test_therapist_entirely_before_window_excluded(self):
        # Therapist: 0-4000, window: 5000-8000
        # end(4000) NOT > from_end(5000) → excluded
        df = self._df_with_therapist(start_ms=0, end_ms=4000)
        result = _collect_therapist_cue(df, 's1', from_end_ms=5000, to_start_ms=8000)
        self.assertEqual(result, '')

    def test_therapist_entirely_after_window_excluded(self):
        # Therapist: 9000-12000, window: 5000-8000
        # start(9000) NOT < to_start(8000) → excluded
        df = self._df_with_therapist(start_ms=9000, end_ms=12000)
        result = _collect_therapist_cue(df, 's1', from_end_ms=5000, to_start_ms=8000)
        self.assertEqual(result, '')

    # ── Speaker filtering ────────────────────────────────────────────────────

    def test_participant_segments_in_window_not_collected(self):
        df = _df(
            _row(session_id='s1', speaker='participant',
                 start_ms=5000, end_ms=8000, text='Participant text.'),
            _row(session_id='s1', speaker='therapist',
                 start_ms=5500, end_ms=7500, text='Therapist text.'),
        )
        result = _collect_therapist_cue(df, 's1', from_end_ms=4000, to_start_ms=9000)
        self.assertIn('Therapist text.', result)
        self.assertNotIn('Participant text.', result)

    def test_no_therapist_rows_returns_empty(self):
        df = _df(
            _row(session_id='s1', speaker='participant',
                 start_ms=5000, end_ms=8000, text='Only participant.'),
        )
        result = _collect_therapist_cue(df, 's1', from_end_ms=4000, to_start_ms=9000)
        self.assertEqual(result, '')

    # ── Multiple segments and ordering ───────────────────────────────────────

    def test_multiple_therapist_segs_joined_with_newline(self):
        df = _df(
            _row(session_id='s1', speaker='therapist',
                 start_ms=5000, end_ms=6000, text='First.'),
            _row(session_id='s1', speaker='therapist',
                 start_ms=6500, end_ms=7500, text='Second.'),
        )
        result = _collect_therapist_cue(df, 's1', from_end_ms=4000, to_start_ms=8000)
        self.assertEqual(result, 'First.\nSecond.')

    def test_segments_sorted_by_start_time(self):
        # Insert in reverse order; output should be chronological
        df = _df(
            _row(session_id='s1', speaker='therapist',
                 start_ms=6500, end_ms=7500, text='Second.'),
            _row(session_id='s1', speaker='therapist',
                 start_ms=5000, end_ms=6000, text='First.'),
        )
        result = _collect_therapist_cue(df, 's1', from_end_ms=4000, to_start_ms=8000)
        self.assertEqual(result, 'First.\nSecond.')

    def test_empty_text_therapist_seg_skipped(self):
        df = _df(
            _row(session_id='s1', speaker='therapist',
                 start_ms=5000, end_ms=6000, text=''),
            _row(session_id='s1', speaker='therapist',
                 start_ms=6500, end_ms=7500, text='Real text.'),
        )
        result = _collect_therapist_cue(df, 's1', from_end_ms=4000, to_start_ms=8000)
        self.assertEqual(result, 'Real text.')

    def test_whitespace_only_text_skipped(self):
        df = _df(
            _row(session_id='s1', speaker='therapist',
                 start_ms=5000, end_ms=6000, text='   '),
            _row(session_id='s1', speaker='therapist',
                 start_ms=6500, end_ms=7500, text='Real.'),
        )
        result = _collect_therapist_cue(df, 's1', from_end_ms=4000, to_start_ms=8000)
        self.assertEqual(result, 'Real.')

    # ── PURER annotation ─────────────────────────────────────────────────────

    def test_annotate_purer_adds_tag_prefix(self):
        df = _df(
            _row(session_id='s1', speaker='therapist',
                 start_ms=5000, end_ms=8000, text='Education text.',
                 purer_primary=3),
        )
        result = _collect_therapist_cue(
            df, 's1', from_end_ms=4000, to_start_ms=9000, annotate_purer=True
        )
        self.assertTrue(result.startswith('[E]'))
        self.assertIn('Education text.', result)

    def test_annotate_purer_all_five_codes(self):
        expected_tags = {0: 'P', 1: 'U', 2: 'R', 3: 'E', 4: 'R2'}
        for code, tag in expected_tags.items():
            df = _df(_row(session_id='s1', speaker='therapist',
                          start_ms=5000, end_ms=8000, text='Text.',
                          purer_primary=code))
            result = _collect_therapist_cue(
                df, 's1', from_end_ms=4000, to_start_ms=9000, annotate_purer=True
            )
            self.assertTrue(result.startswith(f'[{tag}]'),
                            f'Expected [{tag}] prefix for PURER code {code}, got: {result!r}')

    def test_annotate_purer_nan_label_no_prefix(self):
        df = _df(
            _row(session_id='s1', speaker='therapist',
                 start_ms=5000, end_ms=8000, text='No label.',
                 purer_primary=None),
        )
        result = _collect_therapist_cue(
            df, 's1', from_end_ms=4000, to_start_ms=9000, annotate_purer=True
        )
        self.assertFalse(result.startswith('['))
        self.assertIn('No label.', result)

    def test_annotate_purer_false_no_prefix_even_when_labeled(self):
        df = _df(
            _row(session_id='s1', speaker='therapist',
                 start_ms=5000, end_ms=8000, text='Labeled.',
                 purer_primary=0),
        )
        result = _collect_therapist_cue(
            df, 's1', from_end_ms=4000, to_start_ms=9000, annotate_purer=False
        )
        self.assertFalse(result.startswith('['))
        self.assertEqual(result, 'Labeled.')

    def test_annotate_purer_no_purer_column_no_crash(self):
        df = _df(_row(session_id='s1', speaker='therapist',
                      start_ms=5000, end_ms=8000, text='No column.'))
        df = df.drop(columns=['purer_primary'])
        result = _collect_therapist_cue(
            df, 's1', from_end_ms=4000, to_start_ms=9000, annotate_purer=True
        )
        self.assertEqual(result, 'No column.')

    # ── Known empty-cue scenario: therapist absorbed into participant segment ─

    def test_therapist_absorbed_into_participant_timestamp_span_is_empty_cue(self):
        """
        Reproduces the scenario where segmentation merges participant utterances
        P1(0-8s) and P2(12-20s) into one segment with timestamps 0-20s.
        Therapist spoke at 8-12s (within the merged participant's time span).
        The NEXT cue block has from_end_ms=20000.
        The therapist segment ends at 12000ms, which is NOT > from_end_ms=20000.
        → This is expected to be an empty cue (by design, not a bug).
        """
        df = _df(
            # Therapist spoke at 8000-12000ms
            _row(session_id='s1', speaker='therapist',
                 start_ms=8000, end_ms=12000, text='Therapist response.'),
        )
        # from_end_ms = end of merged participant segment (0-20s)
        result = _collect_therapist_cue(df, 's1', from_end_ms=20000, to_start_ms=30000)
        # therapist end(12000) NOT > from_end(20000) → excluded → empty cue
        self.assertEqual(result, '')


# ── _collect_cue_block_purer_profile ─────────────────────────────────────────

class TestCollectCueBlockPurerProfile(unittest.TestCase):

    def _make_df(self, rows):
        return pd.DataFrame(rows)

    def test_returns_empty_dict_when_no_purer_column(self):
        df = _df(_row(speaker='therapist', start_ms=5000, end_ms=8000))
        df = df.drop(columns=['purer_primary'])
        result = _collect_cue_block_purer_profile(df, 's1', 4000, 9000)
        self.assertEqual(result, {})

    def test_returns_empty_dict_when_no_speaker_column(self):
        df = _df(_row(speaker='therapist', start_ms=5000, end_ms=8000, purer_primary=0))
        df = df.drop(columns=['speaker'])
        result = _collect_cue_block_purer_profile(df, 's1', 4000, 9000)
        self.assertEqual(result, {})

    def test_returns_empty_when_to_start_le_from_end(self):
        df = _df(_row(speaker='therapist', start_ms=5000, end_ms=8000, purer_primary=0))
        result = _collect_cue_block_purer_profile(df, 's1', 9000, 8000)
        self.assertEqual(result, {})

    def test_returns_empty_when_adjacent_segs_no_gap(self):
        df = _df(_row(speaker='therapist', start_ms=5000, end_ms=8000, purer_primary=0))
        result = _collect_cue_block_purer_profile(df, 's1', 8000, 8000)
        self.assertEqual(result, {})

    def test_returns_empty_when_from_end_is_none(self):
        df = _df(_row(speaker='therapist', start_ms=5000, end_ms=8000, purer_primary=0))
        result = _collect_cue_block_purer_profile(df, 's1', None, 9000)
        self.assertEqual(result, {})

    def test_returns_empty_when_no_therapist_in_window(self):
        # Therapist ends before window
        df = _df(_row(speaker='therapist', start_ms=1000, end_ms=3000, purer_primary=0))
        result = _collect_cue_block_purer_profile(df, 's1', 5000, 9000)
        self.assertEqual(result, {})

    def test_primary_labels_counted_per_segment(self):
        rows = [
            _row(session_id='s1', speaker='therapist',
                 start_ms=5000, end_ms=6000, purer_primary=3),
            _row(session_id='s1', speaker='therapist',
                 start_ms=6500, end_ms=7500, purer_primary=3),
            _row(session_id='s1', speaker='therapist',
                 start_ms=7500, end_ms=8500, purer_primary=0),
        ]
        df = pd.DataFrame(rows)
        result = _collect_cue_block_purer_profile(df, 's1', 4000, 9000)
        self.assertEqual(result, {3: 2, 0: 1})

    def test_none_primary_labels_skipped(self):
        rows = [
            _row(session_id='s1', speaker='therapist',
                 start_ms=5000, end_ms=6000, purer_primary=None),
            _row(session_id='s1', speaker='therapist',
                 start_ms=6500, end_ms=7500, purer_primary=1),
        ]
        df = pd.DataFrame(rows)
        result = _collect_cue_block_purer_profile(df, 's1', 4000, 9000)
        self.assertEqual(result, {1: 1})

    def test_secondary_labels_counted_per_segment(self):
        # Turn-level PURER: each therapist turn carries its own secondary, so
        # every turn's secondary is counted (not just the first).
        rows = [
            _row(session_id='s1', speaker='therapist',
                 start_ms=5000, end_ms=6000,
                 purer_primary=3, purer_secondary=2),
            _row(session_id='s1', speaker='therapist',
                 start_ms=6500, end_ms=7500,
                 purer_primary=3, purer_secondary=2),
            _row(session_id='s1', speaker='therapist',
                 start_ms=7500, end_ms=8500,
                 purer_primary=3, purer_secondary=2),
        ]
        df = pd.DataFrame(rows)
        result = _collect_cue_block_purer_profile(df, 's1', 4000, 9000,
                                                   include_secondary=True)
        # Primary 3 counted 3 times; secondary 2 also counted once per turn.
        self.assertEqual(result.get(3), 3)
        self.assertEqual(result.get(2), 3)

    def test_distinct_secondary_labels_each_counted(self):
        # Per-turn secondaries can differ across turns in a cue — count each.
        rows = [
            _row(session_id='s1', speaker='therapist',
                 start_ms=5000, end_ms=6000,
                 purer_primary=3, purer_secondary=2),
            _row(session_id='s1', speaker='therapist',
                 start_ms=6500, end_ms=7500,
                 purer_primary=0, purer_secondary=4),
        ]
        df = pd.DataFrame(rows)
        result = _collect_cue_block_purer_profile(df, 's1', 4000, 9000,
                                                   include_secondary=True)
        # primaries: 3×1, 0×1 ; secondaries: 2×1, 4×1
        self.assertEqual(result, {3: 1, 0: 1, 2: 1, 4: 1})

    def test_secondary_not_added_when_flag_false(self):
        rows = [
            _row(session_id='s1', speaker='therapist',
                 start_ms=5000, end_ms=6000,
                 purer_primary=3, purer_secondary=2),
        ]
        df = pd.DataFrame(rows)
        result = _collect_cue_block_purer_profile(df, 's1', 4000, 9000,
                                                   include_secondary=False)
        self.assertEqual(result, {3: 1})
        self.assertNotIn(2, result)

    def test_secondary_none_not_added(self):
        rows = [
            _row(session_id='s1', speaker='therapist',
                 start_ms=5000, end_ms=6000,
                 purer_primary=3, purer_secondary=None),
        ]
        df = pd.DataFrame(rows)
        result = _collect_cue_block_purer_profile(df, 's1', 4000, 9000,
                                                   include_secondary=True)
        self.assertEqual(result, {3: 1})

    def test_participant_segments_excluded_from_profile(self):
        rows = [
            _row(session_id='s1', speaker='participant',
                 start_ms=5000, end_ms=6000, purer_primary=0),
            _row(session_id='s1', speaker='therapist',
                 start_ms=6000, end_ms=7000, purer_primary=3),
        ]
        df = pd.DataFrame(rows)
        result = _collect_cue_block_purer_profile(df, 's1', 4000, 9000)
        # Only therapist PURER (3) counted, not participant's (0)
        self.assertEqual(result, {3: 1})

    def test_uses_same_temporal_overlap_as_collect_cue(self):
        # Therapist ending exactly at from_end: NOT included
        rows = [
            _row(session_id='s1', speaker='therapist',
                 start_ms=2000, end_ms=5000, purer_primary=1),
        ]
        df = pd.DataFrame(rows)
        # from_end=5000, end=5000 → end(5000) NOT > from_end(5000) → excluded
        result = _collect_cue_block_purer_profile(df, 's1', 5000, 9000)
        self.assertEqual(result, {})

    def test_session_scoping_correct(self):
        rows = [
            _row(session_id='s1', speaker='therapist',
                 start_ms=5000, end_ms=7000, purer_primary=0),
            _row(session_id='s2', speaker='therapist',
                 start_ms=5000, end_ms=7000, purer_primary=4),
        ]
        df = pd.DataFrame(rows)
        result = _collect_cue_block_purer_profile(df, 's1', 4000, 9000)
        self.assertEqual(result, {0: 1})


# ── _format_purer_profile ─────────────────────────────────────────────────────

class TestFormatPurerProfile(unittest.TestCase):

    def test_empty_profile_returns_empty_string(self):
        self.assertEqual(_format_purer_profile({}), '')

    def test_single_label(self):
        result = _format_purer_profile({3: 1})
        self.assertEqual(result, '[E×1]')

    def test_multiple_labels_sorted_by_count_descending(self):
        result = _format_purer_profile({0: 1, 3: 3, 1: 2})
        # Should be E×3, U×2, P×1
        self.assertEqual(result, '[E×3, U×2, P×1]')

    def test_unknown_purer_id_shown_as_str(self):
        result = _format_purer_profile({99: 2})
        self.assertIn('99', result)

    def test_uses_short_codes(self):
        result = _format_purer_profile({4: 1})
        self.assertIn('R2', result)


if __name__ == '__main__':
    unittest.main()
