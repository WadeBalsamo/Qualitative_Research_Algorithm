"""
tests/unit/test_purer_turn_level.py
-----------------------------------
Hermetic tests for the TURN-LEVEL PURER context construction added to
``process/orchestrator.py``:

  - ``_build_turn_exchange_context(sorted_segs, spec, target_seg, *,
    window_size, max_words)``

These tests build a small synthetic session of ``Segment`` objects, derive the
real cue-block specs via ``process.cue_blocks.cue_blocks_from_segments``, and
assert the layout contract of the per-turn "full exchange" context block:

  * the explanatory header is present,
  * the opening / next participant turns are labelled and carry their text,
  * the TARGET therapist turn is demarcated with ">>> THERAPIST TURN TO
    CLASSIFY", while sibling therapist turns are plain "THERAPIST:",
  * the essential exchange (opening participant + target turn + next
    participant) is NEVER squeezed out when ``max_words`` is tiny — only the
    preceding-context preamble is budgeted away.

Nothing here touches the network, downloads weights, or calls an LLM client:
``_build_turn_exchange_context`` is a pure string builder and the cue-block
construction is pure-Python.
"""

import os
import sys
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from classification_tools.data_structures import Segment
from process.cue_blocks import cue_blocks_from_segments
from process.orchestrator import _build_turn_exchange_context


# ── Helpers ────────────────────────────────────────────────────────────────────

def _seg(segment_id, speaker, start_ms, end_ms, text, primary_stage=None,
         session_id='c1s1'):
    seg = Segment(
        segment_id=segment_id,
        trial_id='trial_A',
        participant_id='participant_1' if speaker == 'participant' else 'therapist_1',
        session_id=session_id,
        session_number=1,
        cohort_id=1,
        session_variant='',
        segment_index=0,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        total_segments_in_session=0,
        speaker=speaker,
        text=text,
        word_count=len(text.split()),
        speakers_in_segment=[speaker],
        session_file='/data/input/c1s1.vtt',
    )
    if speaker == 'participant':
        seg.primary_stage = primary_stage
    return seg


def _build_session():
    """A session whose middle cue-block has TWO therapist sibling turns.

    Layout (sorted by start_time):
        P0 (stage 1) -> T1 -> T2 -> P1 (stage 2) -> T3 -> P2 (stage 0)

    The P0->P1 window is the spec we want: it holds T1 and T2 (>=2 therapist
    items), so we can verify the target-vs-sibling demarcation.
    """
    return [
        _seg('p0', 'participant', 0, 1000, 'OPENING participant utterance alpha', primary_stage=1),
        _seg('t1', 'therapist', 1000, 2000, 'THERAPIST FIRST sibling turn beta'),
        _seg('t2', 'therapist', 2000, 3000, 'THERAPIST TARGET turn gamma'),
        _seg('p1', 'participant', 3000, 4000, 'NEXT participant utterance delta', primary_stage=2),
        _seg('t3', 'therapist', 4000, 5000, 'trailing therapist turn epsilon'),
        _seg('p2', 'participant', 5000, 6000, 'final participant utterance zeta', primary_stage=0),
    ]


def _spec_with_two_therapists(sorted_segs, specs):
    for spec in specs:
        if len(spec.therapist_items) >= 2:
            return spec
    raise AssertionError("expected a cue-block spec with >=2 therapist items")


# ── Tests ───────────────────────────────────────────────────────────────────────

class TestBuildTurnExchangeContext(unittest.TestCase):
    def setUp(self):
        self.segs = _build_session()
        self.sorted_segs, self.specs = cue_blocks_from_segments(
            self.segs, stage_attr='primary_stage', require_stage=True,
        )
        self.spec = _spec_with_two_therapists(self.sorted_segs, self.specs)
        # Target is the SECOND therapist sibling in the block.
        self.target = self.spec.therapist_items[1]
        self.sibling = self.spec.therapist_items[0]

    def test_cue_block_has_two_therapist_items(self):
        # Guards the fixture: the P0->P1 window holds exactly T1 and T2.
        ids = [t.segment_id for t in self.spec.therapist_items]
        self.assertEqual(ids, ['t1', 't2'])

    def test_layout_contains_header_and_all_parts(self):
        ctx = _build_turn_exchange_context(
            self.sorted_segs, self.spec, self.target,
            window_size=2, max_words=1000,
        )
        # Header
        self.assertIn('FULL EXCHANGE FOR CONTEXT', ctx)
        self.assertIn('classify ONLY the demarcated THERAPIST', ctx)

        # Opening + next participant turns labelled with their text
        self.assertIn('PARTICIPANT (opening):', ctx)
        self.assertIn('OPENING participant utterance alpha', ctx)
        self.assertIn('PARTICIPANT (next):', ctx)
        self.assertIn('NEXT participant utterance delta', ctx)

        # The TARGET therapist turn is demarcated with the marker + carries text
        self.assertIn('>>> THERAPIST TURN TO CLASSIFY', ctx)
        self.assertIn('THERAPIST TARGET turn gamma', ctx)

        # The sibling therapist turn is plain "THERAPIST:" + carries its text
        self.assertIn('THERAPIST FIRST sibling turn beta', ctx)
        # A plain THERAPIST: line (not the demarcated one) exists for the sibling.
        sibling_line = next(
            ln for ln in ctx.splitlines()
            if 'THERAPIST FIRST sibling turn beta' in ln
        )
        self.assertTrue(sibling_line.startswith('THERAPIST:'))
        self.assertNotIn('>>>', sibling_line)

    def test_target_demarcation_applies_to_target_only(self):
        ctx = _build_turn_exchange_context(
            self.sorted_segs, self.spec, self.target,
            window_size=2, max_words=1000,
        )
        marker = '>>> THERAPIST TURN TO CLASSIFY'
        # The marker appears exactly once, on the line carrying the target text.
        self.assertEqual(ctx.count(marker), 1)
        marker_line = next(ln for ln in ctx.splitlines() if marker in ln)
        self.assertIn('THERAPIST TARGET turn gamma', marker_line)
        self.assertNotIn('THERAPIST FIRST sibling turn beta', marker_line)

    def test_demarcation_follows_target_choice(self):
        # When the FIRST therapist turn is the target, demarcation moves to it.
        ctx = _build_turn_exchange_context(
            self.sorted_segs, self.spec, self.sibling,
            window_size=2, max_words=1000,
        )
        marker = '>>> THERAPIST TURN TO CLASSIFY'
        marker_line = next(ln for ln in ctx.splitlines() if marker in ln)
        self.assertIn('THERAPIST FIRST sibling turn beta', marker_line)
        # The (now non-target) gamma turn is a plain THERAPIST: line.
        gamma_line = next(
            ln for ln in ctx.splitlines()
            if 'THERAPIST TARGET turn gamma' in ln
        )
        self.assertTrue(gamma_line.startswith('THERAPIST:'))

    def test_essential_exchange_survives_tiny_max_words(self):
        # With a tiny budget, the preceding-context preamble must be squeezed
        # away, but the essential exchange (opening participant + target turn +
        # next participant) must remain fully present and undamaged.
        ctx = _build_turn_exchange_context(
            self.sorted_segs, self.spec, self.target,
            window_size=2, max_words=1,
        )
        # Essential parts intact
        self.assertIn('PARTICIPANT (opening): OPENING participant utterance alpha', ctx)
        self.assertIn('PARTICIPANT (next): NEXT participant utterance delta', ctx)
        self.assertIn('>>> THERAPIST TURN TO CLASSIFY', ctx)
        self.assertIn('THERAPIST TARGET turn gamma', ctx)
        # Header is also protected (counted as part of the always-kept budget).
        self.assertIn('FULL EXCHANGE FOR CONTEXT', ctx)
        # The preamble label should NOT appear (no budget left for it).
        self.assertNotIn('PRECEDING CONTEXT:', ctx)

    def test_preceding_context_included_when_budget_allows(self):
        # The spec we use has from_index > 0? Build a session with a preceding
        # turn before the opening participant so PRECEDING CONTEXT can appear.
        segs = [
            _seg('pre', 'therapist', 0, 500, 'earlier preamble turn that is preceding context'),
            _seg('p0', 'participant', 1000, 2000, 'opening alpha', primary_stage=1),
            _seg('t1', 'therapist', 2000, 3000, 'sibling beta'),
            _seg('t2', 'therapist', 3000, 4000, 'target gamma'),
            _seg('p1', 'participant', 4000, 5000, 'next delta', primary_stage=2),
            _seg('p2', 'participant', 6000, 7000, 'final zeta', primary_stage=0),
        ]
        sorted_segs, specs = cue_blocks_from_segments(
            segs, stage_attr='primary_stage', require_stage=True,
        )
        spec = _spec_with_two_therapists(sorted_segs, specs)
        target = spec.therapist_items[1]
        ctx = _build_turn_exchange_context(
            sorted_segs, spec, target, window_size=4, max_words=1000,
        )
        # With a generous budget AND a from_index > 0, preceding context appears.
        self.assertIn('PRECEDING CONTEXT:', ctx)


class TestTurnUnitMapping(unittest.TestCase):
    """Indirect contract for ``_purer_llm_classify`` turn mode.

    Calling ``_purer_llm_classify`` requires an LLM client / network, so we do
    NOT invoke it.  Instead we assert the unit-construction contract it relies
    on: in 'turn' mode each cue unit corresponds 1:1 to a real therapist
    Segment within the spec (``spec.therapist_items``), preserving identity and
    order.  This is the invariant the orchestrator depends on when it propagates
    a per-turn label back to a single therapist segment.
    """

    def test_therapist_items_map_one_to_one_to_real_segments(self):
        segs = _build_session()
        by_id = {s.segment_id: s for s in segs if s.speaker == 'therapist'}
        sorted_segs, specs = cue_blocks_from_segments(
            segs, stage_attr='primary_stage', require_stage=True,
        )
        spec = _spec_with_two_therapists(sorted_segs, specs)

        # Each therapist_item IS one of the real therapist Segment objects
        # (identity, not a copy) — a per-turn label maps back to exactly one row.
        for t in spec.therapist_items:
            self.assertIs(t, by_id[t.segment_id])
            self.assertEqual(t.speaker, 'therapist')

        # No duplicate segment_ids across the cue unit (strict 1:1).
        ids = [t.segment_id for t in spec.therapist_items]
        self.assertEqual(len(ids), len(set(ids)))

        # Order matches start-time order within the block.
        starts = [t.start_time_ms for t in spec.therapist_items]
        self.assertEqual(starts, sorted(starts))


if __name__ == '__main__':
    unittest.main()
