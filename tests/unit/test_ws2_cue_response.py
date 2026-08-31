"""Hermetic unit tests for analysis/ws2_cue_response (WS2 re-derivation).

No network, no LLM, no live DB. Exercises the pure logic:
  - VTT parse + consecutive same-speaker cue collapse + role mapping
  - VAAMR stage inheritance by max temporal overlap (same participant)
  - within-participant cue unit building (lesson skip, empty skip)
  - mechanism computation (per-move delta, populated cells)
"""

import os
import sqlite3
import tempfile
import unittest

from analysis.ws2_cue_response import (
    parse_vtt_turns,
    inherit_stages,
    turns_to_segments,
    build_cue_units,
    FrozenParticipantSeg,
    open_ws2_db,
    persist_cue_units,
    compute_mechanism,
    CueUnit,
)


SPEAKER_KEY = {
    'Tina': {'role': 'therapist', 'anonymized_id': 'therapist_1'},
    'Pat': {'role': 'participant', 'anonymized_id': 'Participant_MM001'},
    'Quinn': {'role': 'participant', 'anonymized_id': 'Participant_MM002'},
    'Coord': {'role': 'staff', 'anonymized_id': 'program_coordinator'},
}

VTT = """WEBVTT

1
00:00:01,000 --> 00:00:03,000
Pat: I keep noticing my back hurts all the time.

2
00:00:03,000 --> 00:00:05,000
Tina: Tell me more about that sensation.

3
00:00:05,000 --> 00:00:07,000
Tina: What do you notice when you pause?

4
00:00:07,000 --> 00:00:09,000
Quinn: For me it is the shoulders.

5
00:00:09,000 --> 00:00:11,000
Pat: When I pause I can watch it instead of fighting it.

6
00:00:11,000 --> 00:00:13,000
Coord: Quick logistics note about the recording.

7
00:00:13,000 --> 00:00:15,000
Stranger: who am I?
"""


def _write_vtt(tmpdir):
    p = os.path.join(tmpdir, 'c9s9.vtt')
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write(VTT)
    return p


class TestParseVttTurns(unittest.TestCase):
    def test_collapse_and_roles(self):
        with tempfile.TemporaryDirectory() as td:
            p = _write_vtt(td)
            turns, n_cues, unknown = parse_vtt_turns(p, 'c9s9', SPEAKER_KEY)

        self.assertEqual(n_cues, 7)
        # Cues 2 and 3 are both Tina -> collapse to ONE therapist turn.
        speakers = [(t.speaker_name, t.speaker) for t in turns]
        self.assertEqual(speakers, [
            ('Pat', 'participant'),
            ('Tina', 'therapist'),     # cues 2+3 merged
            ('Quinn', 'participant'),
            ('Pat', 'participant'),
            ('Coord', 'other'),        # staff -> other
            ('Stranger', 'other'),     # unknown -> other
        ])
        # Merged therapist turn carries both sentences and the later end time.
        tina = turns[1]
        self.assertIn('Tell me more', tina.text)
        self.assertIn('What do you notice', tina.text)
        self.assertEqual(tina.end_ms, 7000)
        # Unknown speaker recorded.
        self.assertEqual(unknown, {'Stranger': 1})
        # Participant ids mapped; non-participants empty.
        self.assertEqual(turns[0].participant_id, 'Participant_MM001')
        self.assertEqual(tina.participant_id, '')
        # ms conversion.
        self.assertEqual(turns[0].start_ms, 1000)
        self.assertEqual(turns[0].end_ms, 3000)


class TestInheritStages(unittest.TestCase):
    def test_max_overlap_same_participant(self):
        with tempfile.TemporaryDirectory() as td:
            p = _write_vtt(td)
            turns, _, _ = parse_vtt_turns(p, 'c9s9', SPEAKER_KEY)

        # Two frozen participant segments for Pat covering the two Pat turns,
        # plus an overlapping Quinn segment that must NOT leak into Pat's turns.
        frozen = {'c9s9': [
            FrozenParticipantSeg('Participant_MM001', 'c9s9', 500, 3200, 0),    # Pat turn 1 -> Vigilance(0)
            FrozenParticipantSeg('Participant_MM001', 'c9s9', 10000, 12000, 3), # Pat turn 5 -> Metacog(3)
            FrozenParticipantSeg('Participant_MM002', 'c9s9', 800, 3500, 4),    # Quinn, overlaps Pat t1 in time
        ]}
        n_part, n_inh = inherit_stages(turns, frozen)
        self.assertEqual(n_part, 3)   # Pat, Quinn, Pat
        # Pat turn1 -> 0, Pat turn5 -> 3 (NOT Quinn's stage 4).
        pat_turns = [t for t in turns if t.participant_id == 'Participant_MM001']
        self.assertEqual(pat_turns[0].primary_stage, 0)
        self.assertEqual(pat_turns[1].primary_stage, 3)
        # Quinn has no overlapping/containing frozen seg of its own -> None.
        quinn = [t for t in turns if t.participant_id == 'Participant_MM002'][0]
        self.assertIsNone(quinn.primary_stage)
        self.assertEqual(n_inh, 2)


class TestBuildCueUnits(unittest.TestCase):
    def _staged_turns(self, td):
        p = _write_vtt(td)
        turns, _, _ = parse_vtt_turns(p, 'c9s9', SPEAKER_KEY)
        frozen = {'c9s9': [
            FrozenParticipantSeg('Participant_MM001', 'c9s9', 500, 3200, 0),
            FrozenParticipantSeg('Participant_MM001', 'c9s9', 10000, 12000, 3),
        ]}
        inherit_stages(turns, frozen)
        return turns

    def test_within_participant_cue_unit(self):
        with tempfile.TemporaryDirectory() as td:
            turns = self._staged_turns(td)
        units, stats = build_cue_units(turns_to_segments(turns), skip_lessons=True)
        # Pat has two staged turns -> one within-participant pair, with the
        # merged Tina therapist turn strictly between them as the cue.
        self.assertEqual(len(units), 1)
        u = units[0]
        self.assertEqual(u.participant_id, 'Participant_MM001')
        self.assertEqual(u.from_stage, 0)
        self.assertEqual(u.to_stage, 3)
        self.assertIn('Tell me more', u.cue_text)
        # Quinn's interleaved turn is excluded from the cue (therapist-only).
        self.assertNotIn('shoulders', u.cue_text)

    def test_lesson_skip(self):
        with tempfile.TemporaryDirectory() as td:
            turns = self._staged_turns(td)
        # Force the cue over the lesson threshold -> skipped.
        units, stats = build_cue_units(
            turns_to_segments(turns), skip_lessons=True, max_lesson_words=2)
        self.assertEqual(len(units), 0)
        self.assertEqual(stats['n_skipped_lesson'], 1)


class TestComputeMechanism(unittest.TestCase):
    def test_per_move_and_cells(self):
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, 'ws2.db')
            conn = open_ws2_db(db)
            units = [
                CueUnit('cA', 'c1', 'P1', 'fA', 'tA', 0, 2, 'f', 'c', 1, '', 1),
                CueUnit('cB', 'c1', 'P1', 'fB', 'tB', 1, 1, 'f', 'c', 1, '', 1),
                CueUnit('cC', 'c1', 'P2', 'fC', 'tC', 2, 0, 'f', 'c', 1, '', 1),
            ]
            persist_cue_units(conn, units)
            # Label: cA move0 (delta +2), cB move0 (delta 0), cC move2 (delta -2),
            # plus an ABSTAIN that must be excluded.
            conn.executemany(
                """INSERT INTO ws2_purer_labels
                   (cue_id, primary_move, secondary_move, primary_confidence,
                    vote, justification, model, raw_response)
                   VALUES (?,?,?,?,?,?,?,?)""",
                [
                    ('cA', 0, None, 0.9, 'CODED', '', 'm', ''),
                    ('cB', 0, None, 0.8, 'CODED', '', 'm', ''),
                    ('cC', 2, None, 0.7, 'CODED', '', 'm', ''),
                ],
            )
            conn.commit()
            mech = compute_mechanism(conn)
            conn.close()

        self.assertEqual(mech['n_labeled'], 3)
        # move 0: deltas [+2, 0] -> mean 1.0, n 2
        self.assertEqual(mech['per_move'][0]['n'], 2)
        self.assertAlmostEqual(mech['per_move'][0]['mean_delta'], 1.0)
        # move 2: delta [-2] -> mean -2.0
        self.assertAlmostEqual(mech['per_move'][2]['mean_delta'], -2.0)
        # populated cells: (0,0),(1,0),(2,2) = 3 distinct
        self.assertEqual(len(mech['cell_counts']), 3)
        self.assertEqual(mech['cell_counts'][(0, 0)], 1)
        self.assertEqual(mech['cell_counts'][(1, 0)], 1)


if __name__ == '__main__':
    unittest.main()
