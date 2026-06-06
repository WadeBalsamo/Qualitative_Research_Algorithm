"""
tests/test_cue_blocks.py
------------------------
Unit tests for process/cue_blocks.py.

Covers:
  - Generic build_cue_blocks core
  - cue_blocks_from_segments wrapper (Segment dataclass objects)
  - cue_blocks_from_records wrapper (list-of-dicts / DataFrame rows)
  - Touching-timestamps regression (fe == ts with fe > 0 → therapist included)
  - Index-fallback (fe == 0 → position window)
  - Multi-turn blocks / single-turn blocks
  - Empty blocks yielded even when no therapist items between pair
  - require_stage=True skips participants with null stage
  - Cross-check: same fixture as Segment objects and as dicts produces identical results
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
    final_label=None,
    text='',
):
    """Create a minimal Segment for testing."""
    return Segment(
        segment_id=segment_id,
        session_id=session_id,
        speaker=speaker,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        primary_stage=primary_stage,
        final_label=final_label,
        text=text,
    )


def _rec(
    segment_id,
    session_id='s1',
    speaker='participant',
    start_ms=0,
    end_ms=5000,
    final_label=1,
    purer_primary=None,
):
    """Create a minimal record dict for testing."""
    return {
        'segment_id': segment_id,
        'session_id': session_id,
        'speaker': speaker,
        'start_time_ms': start_ms,
        'end_time_ms': end_ms,
        'final_label': final_label,
        'purer_primary': purer_primary,
    }


# ---------------------------------------------------------------------------
# CueBlockSpec dataclass
# ---------------------------------------------------------------------------

class TestCueBlockSpec(unittest.TestCase):
    def test_fields_accessible(self):
        spec = CueBlockSpec(
            session_id='s1',
            from_item='a',
            to_item='b',
            from_index=0,
            to_index=2,
            from_stage=1,
            to_stage=3,
            transition_type='forward',
            therapist_items=['t'],
        )
        self.assertEqual(spec.session_id, 's1')
        self.assertEqual(spec.from_stage, 1)
        self.assertEqual(spec.to_stage, 3)
        self.assertEqual(spec.transition_type, 'forward')
        self.assertEqual(spec.therapist_items, ['t'])

    def test_default_therapist_items_empty(self):
        spec = CueBlockSpec('s', None, None, 0, 1, 0, 0, 'lateral')
        self.assertEqual(spec.therapist_items, [])


# ---------------------------------------------------------------------------
# Touching timestamps regression
# ---------------------------------------------------------------------------

class TestTouchingTimestamps(unittest.TestCase):
    """
    Regression: old analysis code used ``if to_start_ms > from_end_ms: ... else: empty_slice``.
    When from_end == to_start the old code always emitted an empty slice, skipping the
    timestamp window entirely.  The fix uses the canonical timestamp window unconditionally
    (when fe > 0), so a therapist whose timestamps actually overlap the boundary IS included.

    Concretely: with fe=5000, ts=5000:
      - Old code: empty slice (bug — no window lookup attempted at all)
      - New code: apply window ``t.start < ts AND t.end > fe``
        → therapist with start=4000, end=6000 satisfies 4000<5000 AND 6000>5000 → included
        → therapist with start=5000, end=7000 does NOT satisfy 5000<5000 → NOT included
    """

    def test_touching_timestamps_therapist_overlapping_boundary_is_included_segments(self):
        # from_end == to_start == 5000; therapist 4000-6000 overlaps: start(4000)<ts(5000)
        # and end(6000)>fe(5000) → included.  Old code would have returned empty slice.
        segs = [
            _seg('p1', start_ms=0, end_ms=5000, primary_stage=0),
            _seg('t1', speaker='therapist', start_ms=4000, end_ms=6000, primary_stage=None),
            _seg('p2', start_ms=5000, end_ms=9000, primary_stage=2),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(len(specs), 1)
        spec = specs[0]
        self.assertEqual(len(spec.therapist_items), 1)
        self.assertEqual(spec.therapist_items[0].segment_id, 't1')

    def test_touching_timestamps_therapist_overlapping_boundary_is_included_records(self):
        records = [
            _rec('p1', start_ms=0, end_ms=5000, final_label=0),
            _rec('t1', speaker='therapist', start_ms=4000, end_ms=6000, final_label=None),
            _rec('p2', start_ms=5000, end_ms=9000, final_label=2),
        ]
        specs = cue_blocks_from_records(records)
        self.assertEqual(len(specs), 1)
        self.assertEqual(len(specs[0].therapist_items), 1)
        self.assertEqual(specs[0].therapist_items[0]['segment_id'], 't1')

    def test_touching_timestamps_old_code_would_have_been_empty(self):
        # With fe==ts, the old 'to_start > from_end' guard would have emitted empty.
        # New code applies the window: therapist at 4500-5500 with fe=ts=5000
        # satisfies 4500<5000 AND 5500>5000 → included.
        segs = [
            _seg('p1', start_ms=0, end_ms=5000, primary_stage=0),
            _seg('t1', speaker='therapist', start_ms=4500, end_ms=5500, primary_stage=None),
            _seg('p2', start_ms=5000, end_ms=9000, primary_stage=1),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(len(specs), 1)
        # New code includes this therapist; old code would have given 0
        self.assertEqual(len(specs[0].therapist_items), 1)


# ---------------------------------------------------------------------------
# Index fallback (fe == 0)
# ---------------------------------------------------------------------------

class TestIndexFallback(unittest.TestCase):
    """
    When from_seg.end_time_ms == 0 (field unset / missing), the builder must
    fall back to a sorted-position window instead of a timestamp window.
    """

    def test_fallback_includes_therapist_between_positions_segments(self):
        # from_end = 0 triggers fallback; therapist is physically between the
        # two participants in sorted order.
        segs = [
            _seg('p1', start_ms=1000, end_ms=0, primary_stage=0),    # end_ms=0 → fallback
            _seg('t1', speaker='therapist', start_ms=2000, end_ms=3000, primary_stage=None),
            _seg('p2', start_ms=4000, end_ms=0, primary_stage=2),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(len(specs), 1)
        self.assertEqual(len(specs[0].therapist_items), 1)
        self.assertEqual(specs[0].therapist_items[0].segment_id, 't1')

    def test_fallback_excludes_therapist_after_to_position_segments(self):
        segs = [
            _seg('p1', start_ms=1000, end_ms=0, primary_stage=0),
            _seg('p2', start_ms=2000, end_ms=0, primary_stage=2),
            # therapist is AFTER p2 in sorted order → must NOT be in p1→p2 block
            _seg('t1', speaker='therapist', start_ms=3000, end_ms=4000, primary_stage=None),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(len(specs), 1)
        self.assertEqual(len(specs[0].therapist_items), 0)

    def test_fallback_includes_therapist_between_positions_records(self):
        records = [
            _rec('p1', start_ms=1000, end_ms=0, final_label=0),
            _rec('t1', speaker='therapist', start_ms=2000, end_ms=3000, final_label=None),
            _rec('p2', start_ms=4000, end_ms=0, final_label=2),
        ]
        specs = cue_blocks_from_records(records)
        self.assertEqual(len(specs), 1)
        self.assertEqual(len(specs[0].therapist_items), 1)


# ---------------------------------------------------------------------------
# Multi-turn and single-turn blocks
# ---------------------------------------------------------------------------

class TestBlockItemOrder(unittest.TestCase):
    def test_multi_turn_returns_all_therapist_items_in_order(self):
        segs = [
            _seg('p1', start_ms=0, end_ms=3000, primary_stage=0),
            _seg('t1', speaker='therapist', start_ms=3000, end_ms=5000, primary_stage=None),
            _seg('t2', speaker='therapist', start_ms=5000, end_ms=7000, primary_stage=None),
            _seg('t3', speaker='therapist', start_ms=7000, end_ms=9000, primary_stage=None),
            _seg('p2', start_ms=10000, end_ms=13000, primary_stage=2),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(len(specs), 1)
        ids = [s.segment_id for s in specs[0].therapist_items]
        self.assertEqual(ids, ['t1', 't2', 't3'])

    def test_single_turn_block_has_one_therapist_item(self):
        segs = [
            _seg('p1', start_ms=0, end_ms=3000, primary_stage=1),
            _seg('t1', speaker='therapist', start_ms=3000, end_ms=5000, primary_stage=None),
            _seg('p2', start_ms=6000, end_ms=9000, primary_stage=3),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(len(specs), 1)
        self.assertEqual(len(specs[0].therapist_items), 1)


# ---------------------------------------------------------------------------
# Empty blocks are yielded
# ---------------------------------------------------------------------------

class TestEmptyBlocksYielded(unittest.TestCase):
    def test_empty_block_yielded_when_no_therapist_between_segments(self):
        segs = [
            _seg('p1', start_ms=0, end_ms=3000, primary_stage=0),
            # No therapist segment in window
            _seg('p2', start_ms=5000, end_ms=8000, primary_stage=1),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].therapist_items, [])

    def test_empty_block_yielded_records(self):
        records = [
            _rec('p1', start_ms=0, end_ms=3000, final_label=0),
            _rec('p2', start_ms=5000, end_ms=8000, final_label=1),
        ]
        specs = cue_blocks_from_records(records)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].therapist_items, [])

    def test_three_participants_yield_two_blocks_even_if_both_empty(self):
        segs = [
            _seg('p1', start_ms=0, end_ms=2000, primary_stage=0),
            _seg('p2', start_ms=3000, end_ms=5000, primary_stage=1),
            _seg('p3', start_ms=6000, end_ms=8000, primary_stage=2),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].therapist_items, [])
        self.assertEqual(specs[1].therapist_items, [])


# ---------------------------------------------------------------------------
# require_stage skips null-stage participants
# ---------------------------------------------------------------------------

class TestRequireStage(unittest.TestCase):
    def test_require_stage_true_skips_null_stage_participant(self):
        segs = [
            _seg('p1', start_ms=0, end_ms=3000, primary_stage=None),   # no stage → excluded
            _seg('t1', speaker='therapist', start_ms=3000, end_ms=5000, primary_stage=None),
            _seg('p2', start_ms=6000, end_ms=9000, primary_stage=1),
        ]
        _, specs = cue_blocks_from_segments(segs, require_stage=True)
        # p1 excluded → only 1 participant left → no pairs → no specs
        self.assertEqual(len(specs), 0)

    def test_require_stage_false_includes_null_stage_participant(self):
        segs = [
            _seg('p1', start_ms=0, end_ms=3000, primary_stage=None),
            _seg('p2', start_ms=6000, end_ms=9000, primary_stage=1),
        ]
        _, specs = cue_blocks_from_segments(segs, require_stage=False)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].from_stage, 0)  # None coerced to 0
        self.assertEqual(specs[0].to_stage, 1)

    def test_require_stage_records_skips_nan_final_label(self):
        import math
        records = [
            _rec('p1', start_ms=0, end_ms=3000, final_label=float('nan')),
            _rec('p2', start_ms=6000, end_ms=9000, final_label=2),
        ]
        specs = cue_blocks_from_records(records, require_stage=True)
        self.assertEqual(len(specs), 0)

    def test_require_stage_false_records_includes_nan_final_label(self):
        records = [
            _rec('p1', start_ms=0, end_ms=3000, final_label=float('nan')),
            _rec('p2', start_ms=6000, end_ms=9000, final_label=2),
        ]
        specs = cue_blocks_from_records(records, require_stage=False)
        self.assertEqual(len(specs), 1)


# ---------------------------------------------------------------------------
# Transition type
# ---------------------------------------------------------------------------

class TestTransitionType(unittest.TestCase):
    def test_forward(self):
        segs = [
            _seg('p1', start_ms=0, end_ms=3000, primary_stage=0),
            _seg('p2', start_ms=4000, end_ms=7000, primary_stage=3),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(specs[0].transition_type, 'forward')

    def test_backward(self):
        segs = [
            _seg('p1', start_ms=0, end_ms=3000, primary_stage=3),
            _seg('p2', start_ms=4000, end_ms=7000, primary_stage=1),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(specs[0].transition_type, 'backward')

    def test_lateral(self):
        segs = [
            _seg('p1', start_ms=0, end_ms=3000, primary_stage=2),
            _seg('p2', start_ms=4000, end_ms=7000, primary_stage=2),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(specs[0].transition_type, 'lateral')


# ---------------------------------------------------------------------------
# from_index / to_index are valid positions in sorted_items
# ---------------------------------------------------------------------------

class TestIndexFields(unittest.TestCase):
    def test_from_index_is_position_in_sorted_items(self):
        segs = [
            _seg('p1', start_ms=0, end_ms=3000, primary_stage=0),
            _seg('t1', speaker='therapist', start_ms=3500, end_ms=5000, primary_stage=None),
            _seg('p2', start_ms=6000, end_ms=8000, primary_stage=1),
        ]
        sorted_items, specs = cue_blocks_from_segments(segs)
        spec = specs[0]
        self.assertIs(sorted_items[spec.from_index], spec.from_item)
        self.assertIs(sorted_items[spec.to_index], spec.to_item)


# ---------------------------------------------------------------------------
# Multi-session: blocks don't cross sessions
# ---------------------------------------------------------------------------

class TestMultiSession(unittest.TestCase):
    def test_blocks_not_crossed_between_sessions(self):
        segs = [
            _seg('s1p1', session_id='s1', start_ms=0, end_ms=3000, primary_stage=0),
            _seg('s1p2', session_id='s1', start_ms=5000, end_ms=7000, primary_stage=1),
            # Therapist in s2 with same timestamps — must NOT appear in s1 block
            _seg('s2t1', session_id='s2', speaker='therapist',
                 start_ms=3000, end_ms=5000, primary_stage=None),
        ]
        _, specs = cue_blocks_from_segments(segs)
        s1_specs = [sp for sp in specs if sp.session_id == 's1']
        self.assertEqual(len(s1_specs), 1)
        self.assertEqual(s1_specs[0].therapist_items, [])

    def test_multiple_sessions_correct_spec_count(self):
        segs = []
        # s1: 2 participants → 1 block
        segs += [
            _seg('s1p1', session_id='s1', start_ms=0, end_ms=3000, primary_stage=0),
            _seg('s1p2', session_id='s1', start_ms=5000, end_ms=7000, primary_stage=1),
        ]
        # s2: 3 participants → 2 blocks
        segs += [
            _seg('s2p1', session_id='s2', start_ms=0, end_ms=3000, primary_stage=2),
            _seg('s2p2', session_id='s2', start_ms=5000, end_ms=7000, primary_stage=3),
            _seg('s2p3', session_id='s2', start_ms=9000, end_ms=11000, primary_stage=4),
        ]
        _, specs = cue_blocks_from_segments(segs)
        self.assertEqual(len([sp for sp in specs if sp.session_id == 's1']), 1)
        self.assertEqual(len([sp for sp in specs if sp.session_id == 's2']), 2)


# ---------------------------------------------------------------------------
# Cross-check: Segments and records wrappers agree
# ---------------------------------------------------------------------------

class TestCrossCheck(unittest.TestCase):
    """
    Build the same fixture as both Segment objects and as record dicts and assert
    the two wrappers agree on block count, from/to ids, and therapist counts.
    """

    def _make_fixture(self):
        """Return (segments_list, records_list) for the same scenario."""
        segs = [
            _seg('p1', session_id='s1', start_ms=0,     end_ms=3000,  primary_stage=0),
            _seg('t1', session_id='s1', speaker='therapist', start_ms=3000, end_ms=4500, primary_stage=None),
            _seg('t2', session_id='s1', speaker='therapist', start_ms=4500, end_ms=6000, primary_stage=None),
            _seg('p2', session_id='s1', start_ms=7000,  end_ms=9000,  primary_stage=1),
            _seg('p3', session_id='s1', start_ms=10000, end_ms=12000, primary_stage=2),
            _seg('s2p1', session_id='s2', start_ms=0,   end_ms=2000,  primary_stage=3),
            _seg('s2t1', session_id='s2', speaker='therapist', start_ms=2000, end_ms=3500, primary_stage=None),
            _seg('s2p2', session_id='s2', start_ms=4000, end_ms=6000, primary_stage=4),
        ]

        recs = [
            _rec('p1',   session_id='s1', start_ms=0,     end_ms=3000,  final_label=0),
            _rec('t1',   session_id='s1', speaker='therapist', start_ms=3000, end_ms=4500, final_label=None),
            _rec('t2',   session_id='s1', speaker='therapist', start_ms=4500, end_ms=6000, final_label=None),
            _rec('p2',   session_id='s1', start_ms=7000,  end_ms=9000,  final_label=1),
            _rec('p3',   session_id='s1', start_ms=10000, end_ms=12000, final_label=2),
            _rec('s2p1', session_id='s2', start_ms=0,     end_ms=2000,  final_label=3),
            _rec('s2t1', session_id='s2', speaker='therapist', start_ms=2000, end_ms=3500, final_label=None),
            _rec('s2p2', session_id='s2', start_ms=4000,  end_ms=6000,  final_label=4),
        ]
        return segs, recs

    def test_block_count_matches(self):
        segs, recs = self._make_fixture()
        _, seg_specs = cue_blocks_from_segments(segs, stage_attr='primary_stage')
        rec_specs = cue_blocks_from_records(recs, stage_key='final_label')
        self.assertEqual(len(seg_specs), len(rec_specs))

    def test_from_to_ids_match(self):
        segs, recs = self._make_fixture()
        _, seg_specs = cue_blocks_from_segments(segs, stage_attr='primary_stage')
        rec_specs = cue_blocks_from_records(recs, stage_key='final_label')
        for ss, rs in zip(seg_specs, rec_specs):
            self.assertEqual(ss.from_item.segment_id, rs.from_item['segment_id'])
            self.assertEqual(ss.to_item.segment_id, rs.to_item['segment_id'])

    def test_therapist_counts_match(self):
        segs, recs = self._make_fixture()
        _, seg_specs = cue_blocks_from_segments(segs, stage_attr='primary_stage')
        rec_specs = cue_blocks_from_records(recs, stage_key='final_label')
        for ss, rs in zip(seg_specs, rec_specs):
            self.assertEqual(len(ss.therapist_items), len(rs.therapist_items))

    def test_from_to_stages_match(self):
        segs, recs = self._make_fixture()
        _, seg_specs = cue_blocks_from_segments(segs, stage_attr='primary_stage')
        rec_specs = cue_blocks_from_records(recs, stage_key='final_label')
        for ss, rs in zip(seg_specs, rec_specs):
            self.assertEqual(ss.from_stage, rs.from_stage)
            self.assertEqual(ss.to_stage, rs.to_stage)
            self.assertEqual(ss.transition_type, rs.transition_type)


# ---------------------------------------------------------------------------
# Generic build_cue_blocks (direct call)
# ---------------------------------------------------------------------------

class TestGenericBuilder(unittest.TestCase):
    def test_custom_accessors_work(self):
        """Use plain dict items with custom accessor callables."""
        items = [
            {'id': 'p1', 'sess': 'A', 'spk': 'P', 's': 0,    'e': 1000, 'stg': 0},
            {'id': 't1', 'sess': 'A', 'spk': 'T', 's': 1000,  'e': 2000, 'stg': None},
            {'id': 'p2', 'sess': 'A', 'spk': 'P', 's': 3000,  'e': 4000, 'stg': 2},
        ]
        # Map 'P' → 'participant', 'T' → 'therapist'
        spk_map = {'P': 'participant', 'T': 'therapist'}
        _, specs = build_cue_blocks(
            items,
            get_session=lambda r: r['sess'],
            get_speaker=lambda r: spk_map.get(r['spk'], r['spk']),
            get_start=lambda r: r['s'],
            get_end=lambda r: r['e'],
            get_stage=lambda r: r['stg'],
            get_id=lambda r: r['id'],
        )
        self.assertEqual(len(specs), 1)
        self.assertEqual(len(specs[0].therapist_items), 1)
        self.assertEqual(specs[0].therapist_items[0]['id'], 't1')

    def test_sorted_items_returned_in_start_order(self):
        segs = [
            _seg('p2', start_ms=500, end_ms=1500, primary_stage=1),
            _seg('p1', start_ms=0,   end_ms=500,  primary_stage=0),
        ]
        sorted_items, _ = cue_blocks_from_segments(segs)
        self.assertEqual(sorted_items[0].segment_id, 'p1')
        self.assertEqual(sorted_items[1].segment_id, 'p2')


if __name__ == '__main__':
    unittest.main()
