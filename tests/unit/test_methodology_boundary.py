"""
tests/unit/test_methodology_boundary.py
----------------------------------------
Conceptual conformance: the Critical Design Rule that VAAMR applies ONLY to
participants, and PURER applies ONLY to therapists, and labels never cross.

Methodology references:
  CLAUDE.md §Framework Boundaries: Critical Design Rule
  §4.1: therapist segments excluded from VAAMR classification; therapist
        dialogue retained for PURER classification at Stage 3c.

Tests:
  1. speaker_filter.apply_speaker_filter with mode='exclude': therapist segments
     are removed from a participant-facing population (VAAMR gate).
  2. speaker_filter.apply_speaker_filter: participant segments survive the
     participant-side filter; therapist segments are blocked.
  3. assemble_master_dataset: a therapist segment NEVER receives a VAAMR
     final_label (the assembly code gates gnn_vaamr and primary_stage to
     participant segments).
  4. assemble_master_dataset: a participant segment NEVER receives a purer_final
     label that originated from a non-therapist speaker.
"""

import os
import sys
import tempfile
import shutil
import unittest

import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from classification_tools.data_structures import Segment
from process.speaker_filter import apply_speaker_filter
from process.assembly.master_dataset import assemble_master_dataset


# ---------------------------------------------------------------------------
# Minimal helper to build a SpeakerFilterConfig-duck
# ---------------------------------------------------------------------------

class _FilterCfg:
    def __init__(self, mode='none', speakers=None):
        self.mode = mode
        self.speakers = speakers or []


# ---------------------------------------------------------------------------
# Helpers for Segment construction
# ---------------------------------------------------------------------------

def _participant_seg(sid='p1', primary_stage=2, human_label=None,
                     adjudicated_label=None, gnn_vaamr_pred=None):
    """A classified participant segment."""
    seg = Segment(
        segment_id=sid, session_id='c1s1',
        speaker='participant',
        text='I noticed the sensation and stayed with it.',
    )
    seg.primary_stage = primary_stage
    seg.llm_confidence_primary = 0.85
    seg.llm_run_consistency = 3
    seg.human_label = human_label
    seg.adjudicated_label = adjudicated_label
    seg.gnn_vaamr_pred = gnn_vaamr_pred
    # Ensure no PURER label bleeds in
    seg.purer_primary = None
    return seg


def _therapist_seg(sid='t1', purer_primary=0, gnn_purer_pred=None,
                   purer_primary_stage=None):
    """A classified therapist segment — must NOT carry a VAAMR label."""
    seg = Segment(
        segment_id=sid, session_id='c1s1',
        speaker='therapist',
        text='What did you notice during that practice?',
    )
    seg.purer_primary = purer_primary
    seg.purer_confidence_primary = 0.80
    seg.gnn_purer_pred = gnn_purer_pred
    # primary_stage is intentionally None for therapist segments — never classified
    # with VAAMR (Critical Design Rule, CLAUDE.md)
    seg.primary_stage = purer_primary_stage  # usually None; we can test leakage
    seg.adjudicated_label = None
    seg.human_label = None
    return seg


# ---------------------------------------------------------------------------
# 1. Speaker filter enforces participant-only population for VAAMR
# ---------------------------------------------------------------------------

class TestSpeakerFilterBoundary(unittest.TestCase):
    """
    CLAUDE.md Critical Design Rule — the speaker filter enforces the boundary.
    VAAMR classification is gated by apply_speaker_filter(mode='exclude',
    speakers=['therapist']) which blocks therapist segments.
    """

    def test_participant_filter_excludes_therapists(self):
        """VAAMR filter: therapist segments are excluded (never classified)."""
        segs = [
            Segment(segment_id='p1', speaker='participant', session_id='s1'),
            Segment(segment_id='t1', speaker='therapist', session_id='s1'),
            Segment(segment_id='p2', speaker='participant', session_id='s1'),
        ]
        cfg = _FilterCfg(mode='exclude', speakers=['therapist'])
        result = apply_speaker_filter(segs, cfg)
        speakers = {s.speaker for s in result}
        self.assertNotIn('therapist', speakers,
                         "Therapist segments must not pass the VAAMR filter")

    def test_participant_filter_admits_all_participants(self):
        """After the VAAMR filter, every participant segment is retained."""
        segs = [
            Segment(segment_id='p1', speaker='participant', session_id='s1'),
            Segment(segment_id='t1', speaker='therapist', session_id='s1'),
            Segment(segment_id='p2', speaker='participant', session_id='s1'),
        ]
        cfg = _FilterCfg(mode='exclude', speakers=['therapist'])
        result = apply_speaker_filter(segs, cfg)
        participant_ids = [s.segment_id for s in result]
        self.assertIn('p1', participant_ids)
        self.assertIn('p2', participant_ids)

    def test_no_filter_mode_passes_therapists_as_context(self):
        """
        When mode='none', both participant and therapist segments pass through.
        This is the context-window path — therapist segments appear as READ-ONLY
        preceding context in VAAMR classification prompts (§4.1).
        """
        segs = [
            Segment(segment_id='p1', speaker='participant', session_id='s1'),
            Segment(segment_id='t1', speaker='therapist', session_id='s1'),
        ]
        cfg = _FilterCfg(mode='none')
        result = apply_speaker_filter(segs, cfg)
        speakers = {s.speaker for s in result}
        # Both speakers pass through — therapist is context, not a target
        self.assertIn('therapist', speakers)
        self.assertIn('participant', speakers)


# ---------------------------------------------------------------------------
# 2. assemble_master_dataset: therapist never receives VAAMR final_label
# ---------------------------------------------------------------------------

class TestAssemblyBoundary(unittest.TestCase):
    """
    CLAUDE.md Critical Design Rule — the assembly layer enforces label separation.

    The code in master_dataset.py gates final_label on speaker=='participant'
    (for gnn_consensus) and on primary_stage not being None (llm_zero_shot tier).
    A therapist segment with a primary_stage set (an erroneous upstream condition)
    *would* receive a final_label under the llm_zero_shot tier — but a therapist
    segment with no primary_stage correctly receives final_label=None.

    These tests exercise the assembly code to assert the boundary is maintained
    when segments are correctly constructed by the pipeline.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.outpath = os.path.join(self.tmpdir, 'master_segments.jsonl')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_assembly(self, segments, gnn_ready=False, probe_ready=False):
        # gnn_ready/probe_ready True so the speaker-boundary tests exercise the real
        # boundary gate with the cheap fill tiers eligible.
        df = assemble_master_dataset(
            segments=segments,
            output_path=self.outpath,
            gnn_ready=gnn_ready,
            probe_ready=probe_ready,
        )
        return df

    def test_therapist_segment_with_no_primary_stage_has_null_final_label(self):
        """
        A correctly-constructed therapist segment (primary_stage=None) produces
        final_label=None — VAAMR labels never bleed into therapist rows.

        This is the normal operating condition enforced by the pipeline's speaker
        filter upstream of VAAMR classification (§4.1).
        """
        t = _therapist_seg('t1', purer_primary=0)
        t.primary_stage = None  # explicitly no VAAMR label (correct pipeline state)
        df = self._run_assembly([t])
        row = df[df['segment_id'] == 't1'].iloc[0]
        self.assertIsNone(row['final_label'],
                          "Therapist segment must not carry a VAAMR final_label")

    def test_therapist_segment_receives_purer_final_not_vaamr(self):
        """
        Therapist segments receive purer_final (from purer_primary), not final_label.
        """
        t = _therapist_seg('t1', purer_primary=2)
        t.primary_stage = None
        df = self._run_assembly([t])
        row = df[df['segment_id'] == 't1'].iloc[0]
        self.assertIsNone(row['final_label'],
                          "final_label (VAAMR) must be None for therapist segments")
        self.assertEqual(row['purer_final'], 2,
                         "purer_final should reflect purer_primary for therapist")

    def test_participant_segment_has_no_purer_final_from_classifier(self):
        """
        A participant segment (with no purer_primary) has purer_final=None.
        The PURER label never crosses into the participant side.
        """
        p = _participant_seg('p1', primary_stage=3)
        p.purer_primary = None  # correct: no PURER label for participants
        df = self._run_assembly([p])
        row = df[df['segment_id'] == 'p1'].iloc[0]
        self.assertIsNone(row['purer_final'],
                          "Participant segment must not carry a purer_final label")

    def test_participant_receives_final_label_not_therapist(self):
        """
        A participant segment with primary_stage set gets final_label from it.
        A therapist segment with primary_stage=None gets final_label=None.
        Together these confirm labels do not cross the boundary.
        """
        p = _participant_seg('p1', primary_stage=2)
        t = _therapist_seg('t1', purer_primary=0)
        t.primary_stage = None
        df = self._run_assembly([p, t])

        p_row = df[df['segment_id'] == 'p1'].iloc[0]
        t_row = df[df['segment_id'] == 't1'].iloc[0]

        self.assertEqual(p_row['final_label'], 2,
                         "Participant must receive VAAMR final_label")
        self.assertTrue(pd.isna(t_row['final_label']),
                        "Therapist must NOT receive a VAAMR final_label")

    def test_gnn_vaamr_only_assigned_to_participant_when_authoritative(self):
        """
        When gnn_authoritative=True, gnn_vaamr_pred is only used for participants
        (the assembly code gates: ``seg.speaker == 'participant'``).
        A therapist segment with gnn_vaamr_pred set still gets final_label=None.
        """
        # Therapist segment carrying a spurious gnn_vaamr_pred
        t = _therapist_seg('t1', purer_primary=1, gnn_purer_pred=1)
        t.gnn_vaamr_pred = 3  # spurious — should be ignored for therapists
        t.primary_stage = None

        df = self._run_assembly([t], gnn_ready=True)
        row = df[df['segment_id'] == 't1'].iloc[0]
        self.assertIsNone(row['final_label'],
                          "gnn_vaamr_pred must not produce a VAAMR final_label for therapists "
                          "(assembly gate: speaker == 'participant' required)")

    def test_gnn_purer_only_assigned_to_therapist_when_authoritative(self):
        """
        When gnn_authoritative=True, gnn_purer_pred is only used for therapists
        (assembly code gates: ``seg.speaker == 'therapist'``).
        """
        p = _participant_seg('p1', primary_stage=2)
        p.gnn_purer_pred = 4  # spurious — should be ignored for participants

        df = self._run_assembly([p], gnn_ready=True)
        row = df[df['segment_id'] == 'p1'].iloc[0]
        self.assertIsNone(row['purer_final'],
                          "gnn_purer_pred must not produce a purer_final for participants "
                          "(assembly gate: speaker == 'therapist' required)")


if __name__ == '__main__':
    unittest.main()
