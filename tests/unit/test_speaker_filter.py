"""
tests/unit/test_speaker_filter.py
----------------------------------
Unit tests for process/speaker_filter.py.

Covers:
  - 'none' mode: all segments returned unchanged
  - 'exclude' mode: therapist segments removed, participant segments kept
  - Empty speakers list: even with mode='exclude', all returned (speakers
    set is empty, so the function falls through to mode-based logic)
  - Unknown speaker: not filtered out by exclude mode (only 'therapist' is
    the hard-coded check)
  - Methodology Critical Design Rule:
      VAAMR/participant filter: participants pass, therapists are blocked
      PURER/therapist filter: the module filters BY speaker; the caller
      (orchestrator) controls which population is sent
  - Edge cases: empty segment list, mixed sessions
"""

import os
import sys
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from process.speaker_filter import apply_speaker_filter
from classification_tools.data_structures import Segment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(segment_id, speaker='participant', session_id='s1'):
    return Segment(
        segment_id=segment_id,
        session_id=session_id,
        speaker=speaker,
    )


class _FilerCfg:
    """Minimal mock of SpeakerFilterConfig."""
    def __init__(self, mode='none', speakers=None):
        self.mode = mode
        self.speakers = speakers or []


# ---------------------------------------------------------------------------
# Mode: 'none'
# ---------------------------------------------------------------------------

class TestModeNone(unittest.TestCase):
    def test_mode_none_returns_all_segments_unchanged(self):
        segs = [
            _seg('p1', speaker='participant'),
            _seg('t1', speaker='therapist'),
            _seg('p2', speaker='participant'),
        ]
        cfg = _FilerCfg(mode='none')
        result = apply_speaker_filter(segs, cfg)
        self.assertEqual(result, segs)

    def test_mode_none_with_speakers_list_still_returns_all(self):
        segs = [_seg('p1'), _seg('t1', speaker='therapist')]
        cfg = _FilerCfg(mode='none', speakers=['therapist'])
        result = apply_speaker_filter(segs, cfg)
        self.assertEqual(len(result), 2)

    def test_mode_none_empty_list_returns_empty(self):
        cfg = _FilerCfg(mode='none')
        self.assertEqual(apply_speaker_filter([], cfg), [])


# ---------------------------------------------------------------------------
# Mode: 'exclude'
# ---------------------------------------------------------------------------

class TestModeExclude(unittest.TestCase):
    def test_exclude_removes_therapist_keeps_participant(self):
        """Core VAAMR filter: therapist excluded, participant kept."""
        segs = [
            _seg('p1', speaker='participant'),
            _seg('t1', speaker='therapist'),
            _seg('p2', speaker='participant'),
        ]
        cfg = _FilerCfg(mode='exclude', speakers=['therapist'])
        result = apply_speaker_filter(segs, cfg)
        self.assertEqual(len(result), 2)
        speakers_out = {s.speaker for s in result}
        self.assertNotIn('therapist', speakers_out)
        self.assertIn('participant', speakers_out)

    def test_exclude_all_therapist_yields_only_participants(self):
        segs = [_seg('t1', speaker='therapist'), _seg('t2', speaker='therapist')]
        cfg = _FilerCfg(mode='exclude', speakers=['therapist'])
        result = apply_speaker_filter(segs, cfg)
        self.assertEqual(result, [])

    def test_exclude_no_therapist_segments_unchanged(self):
        segs = [_seg('p1'), _seg('p2'), _seg('p3')]
        cfg = _FilerCfg(mode='exclude', speakers=['therapist'])
        result = apply_speaker_filter(segs, cfg)
        self.assertEqual(result, segs)

    def test_exclude_empty_segment_list(self):
        cfg = _FilerCfg(mode='exclude', speakers=['therapist'])
        self.assertEqual(apply_speaker_filter([], cfg), [])

    def test_exclude_preserves_order(self):
        segs = [
            _seg('p1', speaker='participant'),
            _seg('t1', speaker='therapist'),
            _seg('p2', speaker='participant'),
            _seg('t2', speaker='therapist'),
            _seg('p3', speaker='participant'),
        ]
        cfg = _FilerCfg(mode='exclude', speakers=['therapist'])
        result = apply_speaker_filter(segs, cfg)
        self.assertEqual([s.segment_id for s in result], ['p1', 'p2', 'p3'])

    def test_exclude_empty_speakers_list_returns_all(self):
        """
        When mode='exclude' but speakers=[], the set is empty and the module
        returns all segments unchanged (early-exit on ``not speakers``).
        """
        segs = [_seg('p1'), _seg('t1', speaker='therapist')]
        cfg = _FilerCfg(mode='exclude', speakers=[])
        result = apply_speaker_filter(segs, cfg)
        # Implementation returns all when speakers set is empty
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# Unknown / other speakers
# ---------------------------------------------------------------------------

class TestUnknownSpeaker(unittest.TestCase):
    def test_unknown_speaker_not_excluded_by_exclude_mode(self):
        """
        The exclude filter only checks s.speaker != 'therapist'.  A segment
        with an unknown speaker label is treated as non-therapist and kept.
        """
        segs = [
            _seg('u1', speaker='unknown'),
            _seg('t1', speaker='therapist'),
            _seg('p1', speaker='participant'),
        ]
        cfg = _FilerCfg(mode='exclude', speakers=['therapist'])
        result = apply_speaker_filter(segs, cfg)
        ids = [s.segment_id for s in result]
        self.assertIn('u1', ids)
        self.assertIn('p1', ids)
        self.assertNotIn('t1', ids)

    def test_empty_speaker_string_kept_in_exclude_mode(self):
        segs = [_seg('x1', speaker=''), _seg('t1', speaker='therapist')]
        cfg = _FilerCfg(mode='exclude', speakers=['therapist'])
        result = apply_speaker_filter(segs, cfg)
        ids = [s.segment_id for s in result]
        self.assertIn('x1', ids)
        self.assertNotIn('t1', ids)


# ---------------------------------------------------------------------------
# Methodology Critical Design Rule
# ---------------------------------------------------------------------------

class TestMethodologyCriticalDesignRule(unittest.TestCase):
    """
    CLAUDE.md §Framework Boundaries: VAAMR applies exclusively to participant
    segments.  The speaker filter is the enforcement mechanism.  These tests
    confirm that a 'VAAMR-configured' filter (exclude therapists) admits only
    participant segments, enforcing the design rule at runtime.
    """

    def _build_mixed_session(self):
        return [
            _seg('p_01', speaker='participant', session_id='s1'),
            _seg('th_01', speaker='therapist', session_id='s1'),
            _seg('p_02', speaker='participant', session_id='s1'),
            _seg('th_02', speaker='therapist', session_id='s1'),
            _seg('p_03', speaker='participant', session_id='s1'),
        ]

    def test_vaamr_filter_admits_only_participants(self):
        """After VAAMR filter, only participant segments remain."""
        cfg = _FilerCfg(mode='exclude', speakers=['therapist'])
        segs = self._build_mixed_session()
        result = apply_speaker_filter(segs, cfg)
        self.assertTrue(all(s.speaker == 'participant' for s in result))

    def test_vaamr_filter_participant_count_correct(self):
        cfg = _FilerCfg(mode='exclude', speakers=['therapist'])
        segs = self._build_mixed_session()
        result = apply_speaker_filter(segs, cfg)
        self.assertEqual(len(result), 3)

    def test_no_filter_passes_all_for_context_window(self):
        """
        When no filter is applied (mode='none'), ALL segments (both participant
        and therapist) pass through — used when building context windows for
        classification prompts where therapist segments appear as read-only
        preceding context.
        """
        cfg = _FilerCfg(mode='none')
        segs = self._build_mixed_session()
        result = apply_speaker_filter(segs, cfg)
        self.assertEqual(len(result), 5)

    def test_filter_object_without_mode_attr_treated_as_none(self):
        """
        apply_speaker_filter uses getattr with default 'none', so an object
        missing .mode should behave like mode='none'.
        """
        class _NoMode:
            speakers = []
        segs = [_seg('p1'), _seg('t1', speaker='therapist')]
        result = apply_speaker_filter(segs, _NoMode())
        self.assertEqual(len(result), 2)

    def test_filter_object_without_speakers_attr_treated_as_empty(self):
        """An object missing .speakers defaults to empty → mode='exclude' passthrough."""
        class _NoSpeakers:
            mode = 'exclude'
        segs = [_seg('p1'), _seg('t1', speaker='therapist')]
        result = apply_speaker_filter(segs, _NoSpeakers())
        # speakers defaults to set() → empty → all returned
        self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()
