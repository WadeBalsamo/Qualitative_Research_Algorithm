"""
tests/unit/test_resegment_therapist.py
--------------------------------------
Hermetic tests for THERAPIST-ONLY re-segmentation
(``process/orchestrator.stage_resegment_therapist``) and the building blocks it
relies on.

The real ``stage_resegment_therapist`` reads .vtt inputs, PHI-scrubs via a
transformer (``obi/deid_roberta_i2b2``), and writes ``qra.db``.  To stay
hermetic we:

  * never load the transformer — ``process.orchestrator._scrub_segments`` is
    monkeypatched to an identity pass-through, and the segmenter is built with
    ``skip_embedding_model=True`` so no SentenceTransformer downloads;
  * build a tiny in-memory project: a temp ``output_dir`` with a hand-written
    .vtt under ``01_transcripts_inputs/`` and a ``qra.db`` seeded via
    ``process.segments_io`` (participant + therapist rows) plus a seeded
    ``purer_labels`` overlay row.

Covered:
  1. ``ConversationalSegmenter(skip_embedding_model=True)`` constructs WITHOUT
     loading SentenceTransformer (sentinel that raises if called) and leaves
     ``embedding_model is None``.
  2. ``extract_therapist_segments`` groups therapist sentences by speaker/gap
     and assigns ``speaker='therapist'`` segments with expected boundaries.
  3. End-to-end ``stage_resegment_therapist``: participant rows preserved
     byte-identical (segment_id + segment_index + text), therapist rows
     replaced, orphaned ``purer_labels`` removed, and the return dict counts.
"""

import os
import shutil
import sys
import tempfile
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from classification_tools.data_structures import Segment
from process import segments_io
from process import classifications_io
from process import db as _db
from process import output_paths as _paths
from process import transcript_ingestion as _ti
from process.transcript_ingestion import ConversationalSegmenter


# ── Shared seg builder ──────────────────────────────────────────────────────────

def _seg(segment_id, speaker, segment_index, start_ms, end_ms, text,
         session_id='c1s1'):
    return Segment(
        segment_id=segment_id,
        trial_id='trial_A',
        participant_id='participant_1' if speaker == 'participant' else 'therapist_1',
        session_id=session_id,
        session_number=1,
        cohort_id=1,
        session_variant='',
        segment_index=segment_index,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        total_segments_in_session=4,
        speaker=speaker,
        text=text,
        word_count=len(text.split()),
        speakers_in_segment=[speaker],
        session_file='/data/input/c1s1.vtt',
    )


# ── (a) skip_embedding_model does not load SentenceTransformer ───────────────────

class TestSegmenterSkipsEmbeddingModel(unittest.TestCase):
    def test_skip_embedding_model_does_not_load_sentence_transformer(self):
        class _Boom:
            def __init__(self, *a, **k):
                raise AssertionError(
                    "SentenceTransformer must NOT be constructed when "
                    "skip_embedding_model=True"
                )

        orig = _ti.SentenceTransformer
        _ti.SentenceTransformer = _Boom
        try:
            seg = ConversationalSegmenter({
                'embedding_model': 'Qwen/Qwen3-Embedding-8B',
                'excluded_speakers': ['Dr. T'],
                'speaker_filter_mode': 'exclude',
                'skip_embedding_model': True,
            })
        finally:
            _ti.SentenceTransformer = orig

        self.assertIsNone(seg.embedding_model)
        # speaker_norm still wired up for therapist extraction
        self.assertEqual(seg.speaker_norm.filter_mode, 'exclude')


# ── (b) extract_therapist_segments grouping ──────────────────────────────────────

class TestExtractTherapistSegments(unittest.TestCase):
    def _segmenter(self, excluded):
        orig = _ti.SentenceTransformer

        class _Boom:
            def __init__(self, *a, **k):
                raise AssertionError("no embedding model expected")

        _ti.SentenceTransformer = _Boom
        try:
            return ConversationalSegmenter({
                'excluded_speakers': excluded,
                'speaker_filter_mode': 'exclude',
                'skip_embedding_model': True,
            })
        finally:
            _ti.SentenceTransformer = orig

    def _metadata(self):
        return {
            'trial_id': 'trial_A',
            'session_id': 'c1s1',
            'session_number': 1,
            'cohort_id': 1,
            'session_variant': '',
            'source_file': '/data/input/c1s1.vtt',
        }

    def test_groups_contiguous_therapist_speech_and_splits_on_gap(self):
        seg = self._segmenter(['Dr. T'])
        # Two therapist runs separated by a participant turn AND a temporal gap.
        sentences = [
            {'speaker': 'Dr. T', 'start': 0.0, 'end': 1.0, 'text': 'hello there'},
            {'speaker': 'Dr. T', 'start': 1.0, 'end': 2.0, 'text': 'tell me more'},
            {'speaker': 'P', 'start': 2.0, 'end': 3.0, 'text': 'the pain eased'},
            # large gap from prev therapist end (2.0) -> new therapist run
            {'speaker': 'Dr. T', 'start': 100.0, 'end': 101.0, 'text': 'good observation'},
        ]
        segs = seg.extract_therapist_segments(
            sentences, self._metadata(), max_gap_seconds=30.0,
        )
        # Two therapist segments: [hello there, tell me more] and [good observation]
        self.assertEqual(len(segs), 2)
        self.assertTrue(all(s.speaker == 'therapist' for s in segs))
        self.assertEqual(segs[0].text, 'hello there tell me more')
        self.assertEqual(segs[1].text, 'good observation')
        # Boundaries: first segment spans 0..2s, second 100..101s (ms).
        self.assertEqual(segs[0].start_time_ms, 0)
        self.assertEqual(segs[0].end_time_ms, 2000)
        self.assertEqual(segs[1].start_time_ms, 100000)
        self.assertEqual(segs[1].end_time_ms, 101000)
        # Distinct therapist segment ids.
        self.assertEqual(len({s.segment_id for s in segs}), 2)

    def test_no_excluded_speakers_returns_empty(self):
        seg = self._segmenter([])  # no excluded speakers configured
        sentences = [{'speaker': 'Dr. T', 'start': 0.0, 'end': 1.0, 'text': 'hi'}]
        self.assertEqual(seg.extract_therapist_segments(sentences, self._metadata()), [])

    def test_speaker_change_splits_even_without_gap(self):
        seg = self._segmenter(['Dr. T', 'Coach'])
        sentences = [
            {'speaker': 'Dr. T', 'start': 0.0, 'end': 1.0, 'text': 'one'},
            {'speaker': 'Coach', 'start': 1.0, 'end': 2.0, 'text': 'two'},
        ]
        segs = seg.extract_therapist_segments(sentences, self._metadata(),
                                              max_gap_seconds=30.0)
        self.assertEqual(len(segs), 2)
        self.assertEqual([s.text for s in segs], ['one', 'two'])


# ── (c) End-to-end stage_resegment_therapist on a tiny seeded project ────────────

_VTT = """WEBVTT

1
00:00:00.000 --> 00:00:02.000
Dr. T: welcome back how was your week

2
00:00:02.000 --> 00:00:04.000
participant_1: the tightness in my shoulder eased a lot

3
00:00:04.000 --> 00:00:06.000
Dr. T: wonderful what did you notice while breathing

4
00:00:06.000 --> 00:00:08.000
Dr. T: did the sensation shift at all
"""


class TestStageResegmentTherapistE2E(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # 01_transcripts_inputs/<sid>.vtt
        inputs = _paths.transcripts_diarized_dir(self.tmp)
        os.makedirs(inputs, exist_ok=True)
        with open(os.path.join(inputs, 'c1s1.vtt'), 'w', encoding='utf-8') as fh:
            fh.write(_VTT)

        # Seed qra.db: 2 participant rows + 2 OLD therapist rows.
        self.part_a = _seg('part_A', 'participant', 0, 1000, 2000, 'participant turn A text')
        self.part_b = _seg('part_B', 'participant', 2, 3000, 4000, 'participant turn B text')
        old_th_1 = _seg('old_th_1', 'therapist', 1, 2000, 3000, 'old therapist turn one')
        old_th_2 = _seg('old_th_2', 'therapist', 3, 4000, 5000, 'old therapist turn two')
        segments_io.write_session_segments(
            self.tmp, 'c1s1',
            [self.part_a, old_th_1, self.part_b, old_th_2],
            'seed-hash',
        )

        # Seed a purer_labels overlay row for an OLD therapist segment — this
        # must be removed (orphaned) when the therapist rows are replaced.
        labelled = _seg('old_th_1', 'therapist', 1, 2000, 3000, 'old therapist turn one')
        labelled.purer_primary = 2
        labelled.purer_confidence_primary = 0.8
        labelled.purer_rater_votes = [{'rater': 'r1', 'stage': 2}]
        classifications_io.write_purer_overlay(self.tmp, [labelled])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _config(self):
        from process.config import PipelineConfig
        cfg = PipelineConfig()
        cfg.output_dir = self.tmp
        cfg.trial_id = 'trial_A'
        cfg.speaker_filter.mode = 'exclude'
        cfg.speaker_filter.speakers = ['Dr. T']
        cfg.purer_cue.therapist_max_gap_seconds = 120.0
        return cfg

    def test_resegment_preserves_participants_replaces_therapists(self):
        from process import orchestrator

        # Stub PHI scrub to an identity pass-through (no transformer load).
        orig_scrub = orchestrator._scrub_segments

        def _passthrough(segments, *a, **k):
            return segments, {'n_known': 0, 'n_unknown': 0, 'n_segments_modified': 0}

        orchestrator._scrub_segments = _passthrough
        try:
            result = orchestrator.stage_resegment_therapist(self._config(), self.tmp)
        finally:
            orchestrator._scrub_segments = orig_scrub

        # Return-dict counts.
        self.assertEqual(result['sessions'], 1)
        self.assertEqual(result['old_therapist'], 2)
        self.assertEqual(result['participant_preserved'], 2)
        self.assertGreaterEqual(result['new_therapist'], 1)

        # Re-read frozen segments.
        reloaded = segments_io.read_session_segments(self.tmp, 'c1s1')
        parts = [s for s in reloaded if s.speaker == 'participant']
        ths = [s for s in reloaded if s.speaker == 'therapist']

        # Participants byte-identical (id + index + text preserved).
        self.assertEqual(len(parts), 2)
        by_id = {s.segment_id: s for s in parts}
        self.assertIn('part_A', by_id)
        self.assertIn('part_B', by_id)
        self.assertEqual(by_id['part_A'].segment_index, 0)
        self.assertEqual(by_id['part_A'].text, 'participant turn A text')
        self.assertEqual(by_id['part_B'].segment_index, 2)
        self.assertEqual(by_id['part_B'].text, 'participant turn B text')

        # Therapist rows REPLACED: old ids gone, new ids present.
        th_ids = {s.segment_id for s in ths}
        self.assertNotIn('old_th_1', th_ids)
        self.assertNotIn('old_th_2', th_ids)
        self.assertEqual(len(ths), result['new_therapist'])
        # New therapist indices are above the prior max session index (2).
        for s in ths:
            self.assertGreater(s.segment_index, 2)
        # New therapist text comes from the .vtt content.
        joined = ' '.join(s.text for s in ths)
        self.assertIn('welcome back', joined)

    def test_orphaned_purer_labels_removed(self):
        from process import orchestrator

        orig_scrub = orchestrator._scrub_segments
        orchestrator._scrub_segments = lambda segments, *a, **k: (
            segments, {'n_known': 0, 'n_unknown': 0, 'n_segments_modified': 0}
        )
        try:
            orchestrator.stage_resegment_therapist(self._config(), self.tmp)
        finally:
            orchestrator._scrub_segments = orig_scrub

        # The seeded purer label for old_th_1 must be gone.
        records = classifications_io.read_overlay(self.tmp, 'purer')
        remaining_ids = {r['segment_id'] for r in records}
        self.assertNotIn('old_th_1', remaining_ids)

    def test_total_segments_in_session_updated(self):
        from process import orchestrator

        orig_scrub = orchestrator._scrub_segments
        orchestrator._scrub_segments = lambda segments, *a, **k: (
            segments, {'n_known': 0, 'n_unknown': 0, 'n_segments_modified': 0}
        )
        try:
            orchestrator.stage_resegment_therapist(self._config(), self.tmp)
        finally:
            orchestrator._scrub_segments = orig_scrub

        with _db.open_db(self.tmp) as conn:
            rows = conn.execute(
                "SELECT DISTINCT total_segments_in_session FROM segments "
                "WHERE session_id='c1s1'"
            ).fetchall()
        totals = {r['total_segments_in_session'] for r in rows}
        # All rows agree on a single new total = participants(2) + new therapists.
        self.assertEqual(len(totals), 1)
        reloaded = segments_io.read_session_segments(self.tmp, 'c1s1')
        self.assertEqual(totals.pop(), len(reloaded))


if __name__ == '__main__':
    unittest.main()
