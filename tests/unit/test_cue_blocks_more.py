"""
tests/unit/test_cue_blocks_more.py
-----------------------------------
Gap-filling tests for process/cue_blocks.py not covered by test_cue_blocks.py.

Covers:
  - split_by_word_budget: greedy packing, singleton over-budget turn, content/order
    preservation, empty input
  - format_purer_coverage: report structure, percentage calculation, zero-denominator
    sentinel, per-session breakdown
  - Index-based fallback when end_time_ms == 0 (additional edge cases beyond existing)
  - Touching/adjacent timestamps boundary (therapist STARTING exactly at from_end is
    excluded; therapist ENDING exactly at from_end is excluded — boundary is strict)
  - Methodology Section 4.8: therapist segments strictly between two consecutive
    participant turns are grouped into one cue block; therapist segments outside the
    window are excluded
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from process.cue_blocks import (
    CueBlockSpec,
    build_cue_blocks,
    cue_blocks_from_segments,
    cue_blocks_from_records,
    split_by_word_budget,
    format_purer_coverage,
)
from classification_tools.data_structures import Segment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(
    segment_id,
    session_id='s1',
    speaker='participant',
    start_ms=0,
    end_ms=5000,
    primary_stage=1,
    text='',
):
    return Segment(
        segment_id=segment_id,
        session_id=session_id,
        speaker=speaker,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        primary_stage=primary_stage,
        text=text,
    )


def _rec(
    segment_id,
    session_id='s1',
    speaker='participant',
    start_ms=0,
    end_ms=5000,
    final_label=1,
):
    return {
        'segment_id': segment_id,
        'session_id': session_id,
        'speaker': speaker,
        'start_time_ms': start_ms,
        'end_time_ms': end_ms,
        'final_label': final_label,
    }


def _item(text):
    """Minimal item object with a .text attribute for split_by_word_budget."""
    class _Item:
        pass
    obj = _Item()
    obj.text = text
    return obj


# ---------------------------------------------------------------------------
# split_by_word_budget
# ---------------------------------------------------------------------------

class TestSplitByWordBudget(unittest.TestCase):
    """Greedy packing of items into sub-lists by combined word count."""

    def _text_of(self, item):
        return item.text if hasattr(item, 'text') else str(item)

    def test_empty_items_returns_empty_list(self):
        result = split_by_word_budget([], max_words=10, text_of=lambda x: x)
        self.assertEqual(result, [])

    def test_single_item_under_budget(self):
        items = [_item('hello world')]
        groups = split_by_word_budget(items, max_words=10, text_of=self._text_of)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 1)
        self.assertIs(groups[0][0], items[0])

    def test_single_item_over_budget_becomes_singleton(self):
        """A single item exceeding max_words must still form its own sub-list."""
        big_text = ' '.join(['word'] * 200)
        items = [_item(big_text)]
        groups = split_by_word_budget(items, max_words=50, text_of=self._text_of)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 1)

    def test_multiple_items_all_fit_in_one_group(self):
        items = [_item('one two'), _item('three four'), _item('five')]
        # 2 + 2 + 1 = 5 words, budget = 10
        groups = split_by_word_budget(items, max_words=10, text_of=self._text_of)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 3)

    def test_split_into_two_groups(self):
        # items: 5 words, 5 words, 5 words — budget 7 → [0+1 nope; flush 0 → [0], then 1+2]
        # Actually: start with item0 (5w), item1 would push to 10 > 7 → flush → group2: item1 (5w),
        # item2 would push to 10 > 7 → flush → group3: item2.  So 3 groups of 1.
        # Let's use budget=6 so item0+item1 = 10 > 6 → separate; verify 2 groups for 2 items.
        items = [_item('a b c'), _item('d e f')]  # 3 + 3 = 6 words, budget 5
        groups = split_by_word_budget(items, max_words=5, text_of=self._text_of)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 1)
        self.assertEqual(len(groups[1]), 1)

    def test_greedy_packs_as_many_as_fit(self):
        # items each 3 words, budget 9 → groups of 3
        items = [_item('a b c')] * 6
        groups = split_by_word_budget(items, max_words=9, text_of=self._text_of)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 3)
        self.assertEqual(len(groups[1]), 3)

    def test_content_preserved_all_items_in_output(self):
        """Flattened output must equal original items in order."""
        items = [_item(f'word{i}') for i in range(10)]
        groups = split_by_word_budget(items, max_words=3, text_of=self._text_of)
        flat = [x for g in groups for x in g]
        self.assertEqual(flat, items)

    def test_order_preserved(self):
        items = [_item('alpha'), _item('beta'), _item('gamma'), _item('delta')]
        groups = split_by_word_budget(items, max_words=1, text_of=self._text_of)
        flat = [x for g in groups for x in g]
        self.assertEqual([x.text for x in flat], ['alpha', 'beta', 'gamma', 'delta'])

    def test_over_budget_item_followed_by_small_items(self):
        """Over-budget singleton must NOT absorb subsequent items."""
        big = _item(' '.join(['x'] * 100))  # 100 words
        small1 = _item('a')
        small2 = _item('b')
        groups = split_by_word_budget([big, small1, small2], max_words=5, text_of=self._text_of)
        # First group: just [big]; second group: [small1, small2] (1+1=2 ≤ 5)
        self.assertEqual(len(groups), 2)
        self.assertIn(big, groups[0])
        self.assertIn(small1, groups[1])
        self.assertIn(small2, groups[1])

    def test_empty_text_items_contribute_zero_words(self):
        """Whitespace/empty items should not cause extra splits."""
        items = [_item(''), _item('   '), _item('word'), _item('')]
        groups = split_by_word_budget(items, max_words=5, text_of=self._text_of)
        flat = [x for g in groups for x in g]
        self.assertEqual(len(flat), 4)

    def test_budget_exactly_met_no_extra_group(self):
        """When cumulative words exactly equal max_words, no split yet."""
        # item0: 3 words, item1: 3 words, budget 6 → both fit in one group
        items = [_item('a b c'), _item('d e f')]
        groups = split_by_word_budget(items, max_words=6, text_of=self._text_of)
        self.assertEqual(len(groups), 1)

    def test_string_items_with_custom_text_of(self):
        """text_of callable is honoured."""
        items = ['alpha beta', 'gamma delta epsilon', 'zeta']
        groups = split_by_word_budget(items, max_words=3, text_of=lambda s: s)
        flat = [x for g in groups for x in g]
        self.assertEqual(flat, items)


# ---------------------------------------------------------------------------
# format_purer_coverage
# ---------------------------------------------------------------------------

class TestFormatPurerCoverage(unittest.TestCase):
    def test_returns_string(self):
        result = format_purer_coverage({})
        self.assertIsInstance(result, str)

    def test_header_present(self):
        result = format_purer_coverage({})
        self.assertIn('PURER CLASSIFICATION COVERAGE REPORT', result)

    def test_overall_totals_section_present(self):
        result = format_purer_coverage({'n_blocks': 10})
        self.assertIn('Overall totals', result)

    def test_coverage_pct_computed_correctly(self):
        stats = {
            'n_labeled_segments': 3,
            'labeled_words': 30,
            'total_therapist_words': 100,
        }
        result = format_purer_coverage(stats)
        self.assertIn('30.0%', result)

    def test_zero_denominator_yields_na(self):
        """When total_therapist_words == 0 the coverage should show 'n/a'."""
        stats = {'labeled_words': 0, 'total_therapist_words': 0}
        result = format_purer_coverage(stats)
        self.assertIn('n/a', result)

    def test_per_session_breakdown_rendered(self):
        stats = {
            'per_session': {
                'sess_01': {
                    'n_blocks': 5,
                    'labeled_words': 50,
                    'total_therapist_words': 80,
                }
            }
        }
        result = format_purer_coverage(stats)
        self.assertIn('sess_01', result)
        self.assertIn('Per-session breakdown', result)

    def test_per_session_coverage_pct(self):
        stats = {
            'per_session': {
                'sx': {
                    'labeled_words': 10,
                    'total_therapist_words': 40,
                }
            }
        }
        result = format_purer_coverage(stats)
        self.assertIn('25.0%', result)

    def test_missing_keys_default_to_zero_not_crash(self):
        """Missing stat keys must not raise; defaults to 0."""
        result = format_purer_coverage({'n_blocks': 7})
        self.assertIn('7', result)

    def test_skipped_lesson_shown(self):
        stats = {
            'n_skipped_lesson': 3,
            'skipped_lesson_words': 150,
        }
        result = format_purer_coverage(stats)
        self.assertIn('3', result)
        self.assertIn('150', result)

    def test_multiple_sessions_sorted_alphabetically(self):
        stats = {
            'per_session': {
                'sess_03': {'n_blocks': 1},
                'sess_01': {'n_blocks': 2},
                'sess_02': {'n_blocks': 3},
            }
        }
        result = format_purer_coverage(stats)
        idx01 = result.index('sess_01')
        idx02 = result.index('sess_02')
        idx03 = result.index('sess_03')
        self.assertLess(idx01, idx02)
        self.assertLess(idx02, idx03)


# ---------------------------------------------------------------------------
# Index-based fallback — additional edge cases
# ---------------------------------------------------------------------------

class TestIndexFallbackEdgeCases(unittest.TestCase):
    """
    Supplement the existing TestIndexFallback suite with boundary conditions
    not yet covered.
    """

    def test_fallback_all_end_zero_multiple_therapists_only_middle_included(self):
        """Only the therapist physically between the two participants is included."""
        segs = [
            _seg('p1', start_ms=1000, end_ms=0, primary_stage=0),
            _seg('t_before', speaker='therapist', start_ms=500, end_ms=900, primary_stage=None),
            _seg('t_middle', speaker='therapist', start_ms=2000, end_ms=3000, primary_stage=None),
            _seg('p2', start_ms=4000, end_ms=0, primary_stage=2),
            _seg('t_after', speaker='therapist', start_ms=5000, end_ms=6000, primary_stage=None),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(len(specs), 1)
        ids = [s.segment_id for s in specs[0].therapist_items]
        self.assertIn('t_middle', ids)
        self.assertNotIn('t_before', ids)
        self.assertNotIn('t_after', ids)

    def test_fallback_no_therapist_between_participants(self):
        segs = [
            _seg('p1', start_ms=0, end_ms=0, primary_stage=0),
            _seg('p2', start_ms=1000, end_ms=0, primary_stage=1),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].therapist_items, [])

    def test_fallback_records_wrapper(self):
        """cue_blocks_from_records also uses index fallback when end_ms==0."""
        records = [
            _rec('p1', start_ms=0, end_ms=0, final_label=0),
            _rec('t1', speaker='therapist', start_ms=500, end_ms=0, final_label=None),
            _rec('p2', start_ms=1000, end_ms=0, final_label=2),
        ]
        specs = cue_blocks_from_records(records)
        self.assertEqual(len(specs), 1)
        self.assertEqual(len(specs[0].therapist_items), 1)
        self.assertEqual(specs[0].therapist_items[0]['segment_id'], 't1')


# ---------------------------------------------------------------------------
# Touching/adjacent timestamp boundary — strict window semantics
# ---------------------------------------------------------------------------

class TestTimestampBoundaryStrictness(unittest.TestCase):
    """
    Verify the exact boundary conditions:
      window: t.start_time_ms < to_start  AND  t.end_time_ms > from_end
    Both conditions are STRICT inequalities.
    """

    def test_therapist_starting_exactly_at_to_start_is_excluded(self):
        """t.start == to_start → NOT start < to_start → excluded."""
        segs = [
            _seg('p1', start_ms=0, end_ms=3000, primary_stage=0),
            _seg('t1', speaker='therapist', start_ms=7000, end_ms=9000, primary_stage=None),
            _seg('p2', start_ms=7000, end_ms=10000, primary_stage=1),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].therapist_items, [])

    def test_therapist_ending_exactly_at_from_end_is_excluded(self):
        """t.end == from_end → NOT end > from_end → excluded."""
        segs = [
            _seg('p1', start_ms=0, end_ms=5000, primary_stage=0),
            _seg('t1', speaker='therapist', start_ms=3000, end_ms=5000, primary_stage=None),
            _seg('p2', start_ms=8000, end_ms=11000, primary_stage=2),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].therapist_items, [])

    def test_therapist_entirely_before_window_is_excluded(self):
        segs = [
            _seg('p1', start_ms=5000, end_ms=8000, primary_stage=0),
            _seg('t_early', speaker='therapist', start_ms=1000, end_ms=4000, primary_stage=None),
            _seg('p2', start_ms=10000, end_ms=13000, primary_stage=2),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].therapist_items, [])

    def test_therapist_entirely_after_window_is_excluded(self):
        segs = [
            _seg('p1', start_ms=0, end_ms=3000, primary_stage=0),
            _seg('p2', start_ms=5000, end_ms=7000, primary_stage=2),
            _seg('t_late', speaker='therapist', start_ms=8000, end_ms=9000, primary_stage=None),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].therapist_items, [])


# ---------------------------------------------------------------------------
# Methodology Section 4.8: cue-block definition
# FROM participant → therapist turns between → TO participant
# ---------------------------------------------------------------------------

class TestCueBlockMethodologyDefinition(unittest.TestCase):
    """
    Methodology §4.8: A cue block is the set of therapist segments that appear
    between two consecutive participant turns.  These tests validate that the
    grouping follows this definition exactly.
    """

    def test_all_therapist_segments_between_two_participants_grouped_together(self):
        """All therapist segments in the window form one block (not split)."""
        segs = [
            _seg('p1', start_ms=0, end_ms=2000, primary_stage=0),
            _seg('t1', speaker='therapist', start_ms=2000, end_ms=3000, primary_stage=None),
            _seg('t2', speaker='therapist', start_ms=3000, end_ms=4000, primary_stage=None),
            _seg('t3', speaker='therapist', start_ms=4000, end_ms=5000, primary_stage=None),
            _seg('p2', start_ms=6000, end_ms=8000, primary_stage=2),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(len(specs), 1)
        self.assertEqual(len(specs[0].therapist_items), 3)

    def test_therapist_outside_any_window_not_attributed_to_any_block(self):
        """Therapist segments before the first participant or after the last
        participant are not attributed to any cue block."""
        segs = [
            _seg('t_before', speaker='therapist', start_ms=0, end_ms=1000, primary_stage=None),
            _seg('p1', start_ms=2000, end_ms=4000, primary_stage=1),
            _seg('p2', start_ms=6000, end_ms=8000, primary_stage=2),
            _seg('t_after', speaker='therapist', start_ms=9000, end_ms=10000, primary_stage=None),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(len(specs), 1)
        # the p1->p2 block has no therapist between them
        self.assertEqual(specs[0].therapist_items, [])

    def test_from_and_to_items_are_participant_segments(self):
        """from_item and to_item must always be participant segments."""
        segs = [
            _seg('p1', start_ms=0, end_ms=3000, primary_stage=0),
            _seg('t1', speaker='therapist', start_ms=3000, end_ms=5000, primary_stage=None),
            _seg('p2', start_ms=6000, end_ms=9000, primary_stage=3),
        ]
        _, specs = cue_blocks_from_segments(segs)
        spec = specs[0]
        self.assertEqual(spec.from_item.speaker, 'participant')
        self.assertEqual(spec.to_item.speaker, 'participant')

    def test_therapist_in_one_window_not_duplicated_in_adjacent_window(self):
        """A therapist segment in one window must not appear in the adjacent block."""
        segs = [
            _seg('p1', start_ms=0, end_ms=2000, primary_stage=0),
            _seg('t1', speaker='therapist', start_ms=2000, end_ms=3000, primary_stage=None),
            _seg('p2', start_ms=4000, end_ms=6000, primary_stage=1),
            _seg('t2', speaker='therapist', start_ms=6000, end_ms=7000, primary_stage=None),
            _seg('p3', start_ms=8000, end_ms=10000, primary_stage=2),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(len(specs), 2)
        ids_block1 = {s.segment_id for s in specs[0].therapist_items}
        ids_block2 = {s.segment_id for s in specs[1].therapist_items}
        # No overlap
        self.assertFalse(ids_block1 & ids_block2)
        self.assertIn('t1', ids_block1)
        self.assertIn('t2', ids_block2)

    def test_session_id_carried_to_spec(self):
        segs = [
            _seg('p1', session_id='session_42', start_ms=0, end_ms=3000, primary_stage=0),
            _seg('p2', session_id='session_42', start_ms=5000, end_ms=8000, primary_stage=1),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(specs[0].session_id, 'session_42')

    def test_three_participants_yield_exactly_two_cue_blocks(self):
        """N participants yield N-1 cue blocks (the pairwise window count)."""
        segs = [
            _seg('p1', start_ms=0, end_ms=2000, primary_stage=0),
            _seg('p2', start_ms=3000, end_ms=5000, primary_stage=1),
            _seg('p3', start_ms=6000, end_ms=8000, primary_stage=2),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(len(specs), 2)

    def test_single_participant_yields_no_cue_blocks(self):
        """With only one participant there are no consecutive pairs."""
        segs = [
            _seg('p1', start_ms=0, end_ms=3000, primary_stage=1),
            _seg('t1', speaker='therapist', start_ms=3000, end_ms=5000, primary_stage=None),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(specs, [])

    def test_no_participants_yields_no_cue_blocks(self):
        segs = [
            _seg('t1', speaker='therapist', start_ms=0, end_ms=3000, primary_stage=None),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(specs, [])

    def test_from_stage_and_to_stage_match_participant_labels(self):
        segs = [
            _seg('p1', start_ms=0, end_ms=2000, primary_stage=2),
            _seg('p2', start_ms=4000, end_ms=6000, primary_stage=4),
        ]
        _, specs = cue_blocks_from_segments(segs)
        spec = specs[0]
        self.assertEqual(spec.from_stage, 2)
        self.assertEqual(spec.to_stage, 4)
        self.assertEqual(spec.transition_type, 'forward')


if __name__ == '__main__':
    unittest.main()
