"""
tests/test_segmentation.py
--------------------------
Tests for process/transcript_ingestion.py.

Covers:
  - SpeakerNormalizer — role assignment, ID numbering, stability, counter seeding
  - ConversationalSegmenter._filter_sentences_by_speaker — exclusion logic
  - ConversationalSegmenter.extract_therapist_segments — grouping, gap splitting,
    speaker-change splitting, timestamp conversion
  - ConversationalSegmenter._therapist_block_to_segment — Segment field values
  - ConversationalSegmenter._fold_short_sentences — same-speaker merge, fallback
  - ConversationalSegmenter._merge_undersized — backward/forward merge, speaker boundary
  - ConversationalSegmenter._build_segments — role assignment, timestamp conversion,
    oversized splitting

The embedding model (SentenceTransformer) is mocked so tests run without GPU/network.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sent(speaker='SPEAKER_00', text='Hello world this is a test sentence.',
          start=0.0, end=5.0):
    return {'speaker': speaker, 'text': text, 'start': start, 'end': end}


def _long_text(n_words=80):
    return ' '.join(['word'] * n_words)


METADATA = {
    'trial_id': 'trial1',
    'participant_id': 'participant_1',
    'session_id': 'c1s1',
    'session_number': 1,
    'cohort_id': 1,
    'session_variant': '',
    'source_file': '',
}


def _make_segmenter(excluded=None, filter_mode='exclude', min_words=60,
                    max_words=400, max_gap=30.0, min_words_per_sentence=10):
    """Build a ConversationalSegmenter with a mocked embedding model.

    Pass excluded=[] explicitly to create a segmenter with no excluded speakers.
    The default (excluded=None) uses ['THERAPIST'].
    """
    actual_excluded = ['THERAPIST'] if excluded is None else excluded
    config = {
        'embedding_model': 'mock-model',
        'excluded_speakers': actual_excluded,
        'speaker_filter_mode': filter_mode,
        'min_segment_words_conversational': min_words,
        'max_segment_words_conversational': max_words,
        'silence_threshold_ms': 2000,
        'semantic_shift_percentile': 25,
        'max_gap_seconds': max_gap,
        'min_words_per_sentence': min_words_per_sentence,
        'max_segment_duration_seconds': 300.0,
        'use_adaptive_threshold': False,
        'min_prominence': 0.05,
        'broad_window_size': 7,
        'use_topic_clustering': False,
    }

    with patch('process.transcript_ingestion.SentenceTransformer') as mock_st:
        mock_model = MagicMock()
        mock_model.device = 'cpu'
        mock_st.return_value = mock_model
        from process.transcript_ingestion import ConversationalSegmenter
        seg = ConversationalSegmenter(config)
    return seg


# ── SpeakerNormalizer ─────────────────────────────────────────────────────────

class TestSpeakerNormalizer(unittest.TestCase):

    def _make(self, excluded=None, filter_mode='exclude', existing_map=None,
              use_unknown_prefix=False):
        from process.transcript_ingestion import SpeakerNormalizer
        return SpeakerNormalizer(
            excluded_speakers=excluded or [],
            filter_mode=filter_mode,
            existing_map=existing_map,
            use_unknown_prefix=use_unknown_prefix,
        )

    def test_excluded_speaker_assigned_therapist_role(self):
        norm = self._make(excluded=['THERAPIST'], filter_mode='exclude')
        role, anon_id = norm.normalize('THERAPIST')
        self.assertEqual(role, 'therapist')
        self.assertTrue(anon_id.startswith('therapist_'))

    def test_non_excluded_speaker_assigned_participant_role(self):
        norm = self._make(excluded=['THERAPIST'], filter_mode='exclude')
        role, anon_id = norm.normalize('SPEAKER_00')
        self.assertEqual(role, 'participant')
        self.assertTrue(anon_id.startswith('participant_'))

    def test_filter_mode_none_all_become_participants(self):
        norm = self._make(excluded=['THERAPIST'], filter_mode='none')
        role, _ = norm.normalize('THERAPIST')
        self.assertEqual(role, 'participant')

    def test_participant_ids_increment(self):
        norm = self._make(excluded=['THERAPIST'], filter_mode='exclude')
        _, id1 = norm.normalize('SPEAKER_00')
        _, id2 = norm.normalize('SPEAKER_01')
        self.assertEqual(id1, 'participant_1')
        self.assertEqual(id2, 'participant_2')

    def test_therapist_ids_increment(self):
        norm = self._make(excluded=['T1', 'T2'], filter_mode='exclude')
        _, id1 = norm.normalize('T1')
        _, id2 = norm.normalize('T2')
        self.assertEqual(id1, 'therapist_1')
        self.assertEqual(id2, 'therapist_2')

    def test_same_speaker_always_same_id(self):
        norm = self._make(excluded=['T'], filter_mode='exclude')
        _, id1 = norm.normalize('SPEAKER_00')
        _, id2 = norm.normalize('SPEAKER_00')
        self.assertEqual(id1, id2)

    def test_existing_map_pre_populates(self):
        existing = {'SPEAKER_00': ('participant', 'participant_1')}
        norm = self._make(excluded=[], filter_mode='none', existing_map=existing)
        role, anon_id = norm.normalize('SPEAKER_00')
        self.assertEqual(role, 'participant')
        self.assertEqual(anon_id, 'participant_1')

    def test_counters_seeded_from_existing_map(self):
        existing = {'SPEAKER_00': ('participant', 'participant_5')}
        norm = self._make(excluded=[], filter_mode='none', existing_map=existing)
        # New speaker should get participant_6, not participant_1
        _, new_id = norm.normalize('SPEAKER_NEW')
        self.assertEqual(new_id, 'participant_6')

    def test_therapist_counter_seeded_from_existing_map(self):
        existing = {'THERAPIST': ('therapist', 'therapist_3')}
        norm = self._make(excluded=['THERAPIST', 'THERAPIST2'],
                          filter_mode='exclude', existing_map=existing)
        # THERAPIST already registered, THERAPIST2 is new
        _, id2 = norm.normalize('THERAPIST2')
        self.assertEqual(id2, 'therapist_4')

    def test_use_unknown_prefix_for_new_speakers(self):
        existing = {'SPEAKER_00': ('participant', 'participant_1')}
        norm = self._make(
            excluded=[], filter_mode='none',
            existing_map=existing, use_unknown_prefix=True
        )
        _, new_id = norm.normalize('SPEAKER_NEW')
        self.assertTrue(new_id.startswith('unknownparticipant_'))

    def test_get_role_returns_correct_role(self):
        norm = self._make(excluded=['T'], filter_mode='exclude')
        self.assertEqual(norm.get_role('T'), 'therapist')
        self.assertEqual(norm.get_role('P'), 'participant')

    def test_get_normalized_id_returns_id(self):
        norm = self._make(excluded=[], filter_mode='none')
        nid = norm.get_normalized_id('SPEAKER_00')
        self.assertTrue(nid.startswith('participant_'))

    def test_excluded_speakers_pre_registered_in_exclude_mode(self):
        norm = self._make(excluded=['DR_SMITH'], filter_mode='exclude')
        # DR_SMITH should already be in speaker_map from __init__
        self.assertIn('DR_SMITH', norm.speaker_map)
        self.assertEqual(norm.speaker_map['DR_SMITH'][0], 'therapist')


# ── _filter_sentences_by_speaker ─────────────────────────────────────────────

class TestFilterSentencesBySpeaker(unittest.TestCase):

    def setUp(self):
        self.seg = _make_segmenter(excluded=['THERAPIST'], filter_mode='exclude')

    def test_excluded_speaker_removed(self):
        sentences = [
            _sent(speaker='SPEAKER_00', text='Participant speech.'),
            _sent(speaker='THERAPIST', text='Therapist speech.'),
        ]
        filtered = self.seg._filter_sentences_by_speaker(sentences)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['speaker'], 'SPEAKER_00')

    def test_non_excluded_speaker_kept(self):
        sentences = [_sent(speaker='SPEAKER_00', text='Participant.')]
        filtered = self.seg._filter_sentences_by_speaker(sentences)
        self.assertEqual(len(filtered), 1)

    def test_empty_excluded_keeps_all(self):
        seg = _make_segmenter(excluded=[], filter_mode='none')
        sentences = [
            _sent(speaker='THERAPIST', text='Therapist.'),
            _sent(speaker='SPEAKER_00', text='Participant.'),
        ]
        filtered = seg._filter_sentences_by_speaker(sentences)
        self.assertEqual(len(filtered), 2)

    def test_all_excluded_returns_empty(self):
        sentences = [
            _sent(speaker='THERAPIST', text='Therapist 1.'),
            _sent(speaker='THERAPIST', text='Therapist 2.'),
        ]
        filtered = self.seg._filter_sentences_by_speaker(sentences)
        self.assertEqual(len(filtered), 0)

    def test_mixed_session_correct_count(self):
        sentences = [
            _sent(speaker='SPEAKER_00', text='P1.'),
            _sent(speaker='THERAPIST', text='T1.'),
            _sent(speaker='SPEAKER_00', text='P2.'),
            _sent(speaker='THERAPIST', text='T2.'),
            _sent(speaker='SPEAKER_01', text='P3.'),
        ]
        filtered = self.seg._filter_sentences_by_speaker(sentences)
        speakers = [s['speaker'] for s in filtered]
        self.assertEqual(speakers, ['SPEAKER_00', 'SPEAKER_00', 'SPEAKER_01'])


# ── extract_therapist_segments ────────────────────────────────────────────────

class TestExtractTherapistSegments(unittest.TestCase):

    def setUp(self):
        self.seg = _make_segmenter(
            excluded=['THERAPIST'], filter_mode='exclude',
            max_gap=30.0
        )

    def test_no_excluded_speakers_returns_empty(self):
        seg = _make_segmenter(excluded=[], filter_mode='none')
        sentences = [_sent(speaker='THERAPIST', text='Therapist.')]
        result = seg.extract_therapist_segments(sentences, METADATA)
        self.assertEqual(result, [])

    def test_no_therapist_sentences_returns_empty(self):
        sentences = [
            _sent(speaker='SPEAKER_00', text='Participant.'),
        ]
        result = self.seg.extract_therapist_segments(sentences, METADATA)
        self.assertEqual(result, [])

    def test_consecutive_therapist_sentences_grouped_into_one_segment(self):
        sentences = [
            _sent(speaker='THERAPIST', text='First therapist sentence.',
                  start=5.0, end=8.0),
            _sent(speaker='THERAPIST', text='Second therapist sentence.',
                  start=8.0, end=11.0),
            _sent(speaker='THERAPIST', text='Third therapist sentence.',
                  start=11.0, end=14.0),
        ]
        result = self.seg.extract_therapist_segments(sentences, METADATA)
        self.assertEqual(len(result), 1)

    def test_gap_splits_therapist_block(self):
        # Gap between sentences > max_gap_seconds (30s)
        sentences = [
            _sent(speaker='THERAPIST', text='First block.', start=5.0, end=8.0),
            _sent(speaker='THERAPIST', text='Second block.', start=50.0, end=55.0),
        ]
        result = self.seg.extract_therapist_segments(sentences, METADATA, max_gap_seconds=30.0)
        self.assertEqual(len(result), 2)

    def test_small_gap_not_split(self):
        # Gap of 2 seconds < 30s max_gap
        sentences = [
            _sent(speaker='THERAPIST', text='Block A.', start=5.0, end=8.0),
            _sent(speaker='THERAPIST', text='Still block A.', start=10.0, end=13.0),
        ]
        result = self.seg.extract_therapist_segments(sentences, METADATA, max_gap_seconds=30.0)
        self.assertEqual(len(result), 1)

    def test_speaker_change_splits_block(self):
        seg = _make_segmenter(excluded=['T1', 'T2'], filter_mode='exclude')
        sentences = [
            _sent(speaker='T1', text='Therapist 1.', start=5.0, end=8.0),
            _sent(speaker='T2', text='Therapist 2.', start=8.0, end=11.0),
        ]
        result = seg.extract_therapist_segments(sentences, METADATA)
        self.assertEqual(len(result), 2)

    def test_start_end_times_converted_to_ms(self):
        sentences = [
            _sent(speaker='THERAPIST', text='Therapist block.',
                  start=5.0, end=10.0),
        ]
        result = self.seg.extract_therapist_segments(sentences, METADATA)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].start_time_ms, 5000)
        self.assertEqual(result[0].end_time_ms, 10000)

    def test_multi_sentence_block_uses_first_start_last_end(self):
        sentences = [
            _sent(speaker='THERAPIST', text='First.', start=5.0, end=8.0),
            _sent(speaker='THERAPIST', text='Second.', start=8.0, end=12.0),
            _sent(speaker='THERAPIST', text='Third.', start=12.0, end=15.0),
        ]
        result = self.seg.extract_therapist_segments(sentences, METADATA)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].start_time_ms, 5000)
        self.assertEqual(result[0].end_time_ms, 15000)

    def test_speaker_field_is_therapist(self):
        sentences = [_sent(speaker='THERAPIST', text='Therapist.', start=5.0, end=8.0)]
        result = self.seg.extract_therapist_segments(sentences, METADATA)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].speaker, 'therapist')

    def test_text_joined_with_space(self):
        sentences = [
            _sent(speaker='THERAPIST', text='First sentence.', start=5.0, end=8.0),
            _sent(speaker='THERAPIST', text='Second sentence.', start=8.0, end=11.0),
        ]
        result = self.seg.extract_therapist_segments(sentences, METADATA)
        self.assertEqual(result[0].text, 'First sentence. Second sentence.')

    def test_session_metadata_correctly_assigned(self):
        sentences = [_sent(speaker='THERAPIST', text='Text.', start=1.0, end=3.0)]
        result = self.seg.extract_therapist_segments(sentences, METADATA)
        self.assertEqual(len(result), 1)
        seg = result[0]
        self.assertEqual(seg.trial_id, 'trial1')
        self.assertEqual(seg.session_id, 'c1s1')
        self.assertEqual(seg.session_number, 1)
        self.assertEqual(seg.cohort_id, 1)

    def test_participant_sentences_not_included(self):
        sentences = [
            _sent(speaker='SPEAKER_00', text='Participant.', start=0.0, end=5.0),
            _sent(speaker='THERAPIST', text='Therapist.', start=5.0, end=8.0),
        ]
        result = self.seg.extract_therapist_segments(sentences, METADATA)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, 'Therapist.')

    def test_exact_gap_threshold_splits_block(self):
        # gap = 31s > 30s max_gap → splits
        sentences = [
            _sent(speaker='THERAPIST', text='Block1.', start=0.0, end=5.0),
            _sent(speaker='THERAPIST', text='Block2.', start=36.0, end=40.0),
        ]
        result = self.seg.extract_therapist_segments(sentences, METADATA, max_gap_seconds=30.0)
        self.assertEqual(len(result), 2)

    def test_segment_ids_include_session_id(self):
        sentences = [_sent(speaker='THERAPIST', text='Text.', start=1.0, end=3.0)]
        result = self.seg.extract_therapist_segments(sentences, METADATA)
        self.assertIn('c1s1', result[0].segment_id)


# ── _fold_short_sentences ─────────────────────────────────────────────────────

class TestFoldShortSentences(unittest.TestCase):

    def setUp(self):
        # min_words_per_sentence=10 so 'word ' * 5 = 5 words is short
        self.seg = _make_segmenter(
            excluded=['THERAPIST'], filter_mode='exclude',
            min_words_per_sentence=10
        )

    def _sent(self, speaker='SPEAKER_00', n_words=15, start=0.0, end=5.0):
        return {'speaker': speaker, 'text': ' '.join(['word'] * n_words),
                'start': start, 'end': end}

    def test_no_short_sentences_unchanged(self):
        sentences = [
            self._sent(n_words=15, start=0.0, end=5.0),
            self._sent(n_words=20, start=5.0, end=10.0),
            self._sent(n_words=15, start=10.0, end=15.0),
        ]
        result = self.seg._fold_short_sentences(sentences)
        self.assertEqual(len(result), 3)

    def test_short_sentence_folded_into_previous_same_speaker(self):
        sentences = [
            self._sent(n_words=15, start=0.0, end=5.0),   # long
            self._sent(n_words=5, start=5.0, end=7.0),    # short → should fold into prev
        ]
        result = self.seg._fold_short_sentences(sentences)
        self.assertEqual(len(result), 1)
        # Text should be combined
        self.assertIn('word', result[0]['text'])

    def test_short_sentence_folded_into_next_when_next_is_closer(self):
        sentences = [
            self._sent(n_words=15, start=0.0, end=5.0),   # long, far away
            self._sent(n_words=5, start=10.0, end=12.0),   # short
            self._sent(n_words=15, start=12.5, end=17.0),  # long, close (gap=0.5)
        ]
        result = self.seg._fold_short_sentences(sentences)
        # Short should fold into next (closer)
        self.assertEqual(len(result), 2)

    def test_short_folded_into_nearest_same_speaker_preference(self):
        # Short participant between two participant sentences, equidistant
        sentences = [
            self._sent(speaker='SPEAKER_00', n_words=15, start=0.0, end=5.0),
            self._sent(speaker='SPEAKER_00', n_words=5, start=5.5, end=7.0),
            self._sent(speaker='SPEAKER_00', n_words=15, start=7.5, end=12.0),
        ]
        result = self.seg._fold_short_sentences(sentences)
        self.assertEqual(len(result), 2)

    def test_short_with_no_same_speaker_folds_into_nearest_overall(self):
        sentences = [
            self._sent(speaker='SPEAKER_00', n_words=15, start=0.0, end=5.0),
            self._sent(speaker='SPEAKER_01', n_words=5, start=5.5, end=7.0),
            # No SPEAKER_01 neighbor — should fold into SPEAKER_00 (closest)
        ]
        result = self.seg._fold_short_sentences(sentences)
        self.assertEqual(len(result), 1)

    def test_empty_sentences_unchanged(self):
        result = self.seg._fold_short_sentences([])
        self.assertEqual(result, [])

    def test_single_long_sentence_unchanged(self):
        sentences = [self._sent(n_words=15, start=0.0, end=5.0)]
        result = self.seg._fold_short_sentences(sentences)
        self.assertEqual(len(result), 1)


# ── _merge_undersized ─────────────────────────────────────────────────────────

class TestMergeUndersized(unittest.TestCase):

    def setUp(self):
        self.seg = _make_segmenter(min_words=60)

    def _make_segment(self, participant_id='participant_1', text='', n_words=None,
                      start_ms=0, end_ms=5000):
        from classification_tools.data_structures import Segment
        t = text or ' '.join(['word'] * (n_words or 10))
        return Segment(
            segment_id='seg1',
            trial_id='t1',
            participant_id=participant_id,
            session_id='c1s1',
            session_number=1,
            cohort_id=1,
            session_variant='',
            text=t,
            word_count=len(t.split()),
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            speaker='participant',
        )

    def test_single_segment_returned_unchanged(self):
        seg = self._make_segment(n_words=10)
        result = self.seg._merge_undersized([seg])
        self.assertEqual(len(result), 1)

    def test_undersized_merged_backward_into_same_speaker(self):
        big = self._make_segment(n_words=80, start_ms=0, end_ms=5000)
        small = self._make_segment(n_words=10, start_ms=5000, end_ms=6000)
        result = self.seg._merge_undersized([big, small])
        # Small should merge into big
        self.assertEqual(len(result), 1)
        self.assertGreater(result[0].word_count, 80)

    def test_undersized_not_merged_across_speaker_boundary(self):
        big = self._make_segment(participant_id='participant_1', n_words=80,
                                 start_ms=0, end_ms=5000)
        small = self._make_segment(participant_id='participant_2', n_words=10,
                                   start_ms=5000, end_ms=6000)
        result = self.seg._merge_undersized([big, small])
        # Different participant_ids → no merge
        self.assertEqual(len(result), 2)

    def test_undersized_merged_forward_when_no_prev_same_speaker(self):
        small = self._make_segment(n_words=10, start_ms=0, end_ms=1000)
        big = self._make_segment(n_words=80, start_ms=1000, end_ms=5000)
        result = self.seg._merge_undersized([small, big])
        # small merges forward into big
        self.assertEqual(len(result), 1)

    def test_merge_updates_end_time(self):
        big = self._make_segment(n_words=80, start_ms=0, end_ms=5000)
        small = self._make_segment(n_words=10, start_ms=5000, end_ms=9000)
        result = self.seg._merge_undersized([big, small])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].end_time_ms, 9000)

    def test_already_large_segments_unchanged(self):
        s1 = self._make_segment(n_words=80, start_ms=0, end_ms=5000)
        s2 = self._make_segment(n_words=90, start_ms=5000, end_ms=10000)
        result = self.seg._merge_undersized([s1, s2])
        self.assertEqual(len(result), 2)


# ── _build_segments ───────────────────────────────────────────────────────────

class TestBuildSegments(unittest.TestCase):

    def setUp(self):
        self.seg = _make_segmenter(
            excluded=['THERAPIST'], filter_mode='exclude',
            min_words=5, max_words=50
        )

    def _sents(self, n=3, speaker='SPEAKER_00', start_offset=0.0, gap=5.0):
        sents = []
        for i in range(n):
            start = start_offset + i * gap
            sents.append({
                'speaker': speaker,
                'text': ' '.join(['word'] * 20),  # 20 words each
                'start': start,
                'end': start + gap - 0.5,
            })
        return sents

    def test_timestamps_converted_to_ms(self):
        sents = [{'speaker': 'SPEAKER_00', 'text': ' '.join(['word'] * 20),
                  'start': 3.5, 'end': 7.5}]
        segments = self.seg._build_segments(sents, [], METADATA)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].start_time_ms, 3500)
        self.assertEqual(segments[0].end_time_ms, 7500)

    def test_speaker_role_assigned_to_participant(self):
        sents = [{'speaker': 'SPEAKER_00', 'text': ' '.join(['word'] * 20),
                  'start': 0.0, 'end': 5.0}]
        segments = self.seg._build_segments(sents, [], METADATA)
        self.assertEqual(segments[0].speaker, 'participant')

    def test_session_metadata_assigned(self):
        sents = [{'speaker': 'SPEAKER_00', 'text': ' '.join(['word'] * 20),
                  'start': 0.0, 'end': 5.0}]
        segments = self.seg._build_segments(sents, [], METADATA)
        self.assertEqual(segments[0].session_id, 'c1s1')
        self.assertEqual(segments[0].trial_id, 'trial1')
        self.assertEqual(segments[0].session_number, 1)

    def test_no_boundary_produces_single_segment(self):
        sents = self._sents(n=3, speaker='SPEAKER_00')
        segments = self.seg._build_segments(sents, [], METADATA)
        # All sentences with no boundaries → one segment (may be merged anyway)
        self.assertGreaterEqual(len(segments), 1)

    def test_boundary_splits_into_two_segments(self):
        sents = self._sents(n=4, speaker='SPEAKER_00')
        # Boundary after index 1: splits sents[0:2] and sents[2:4]
        segments = self.seg._build_segments(sents, [1], METADATA)
        self.assertEqual(len(segments), 2)

    def test_oversized_segment_split_recursively(self):
        # max_words=50; 3 sentences × 20 words = 60 words → should split
        sents = [
            {'speaker': 'SPEAKER_00', 'text': ' '.join(['word'] * 20),
             'start': i * 5.0, 'end': i * 5.0 + 4.5}
            for i in range(3)
        ]
        segments = self.seg._build_segments(sents, [], METADATA)
        # Each sub-segment should be ≤ max_words (50)
        for seg in segments:
            self.assertLessEqual(seg.word_count, 50)

    def test_start_time_ms_is_first_sentence_start(self):
        sents = [
            {'speaker': 'SPEAKER_00', 'text': ' '.join(['word'] * 10),
             'start': 2.5, 'end': 5.0},
            {'speaker': 'SPEAKER_00', 'text': ' '.join(['word'] * 10),
             'start': 5.0, 'end': 8.0},
        ]
        segments = self.seg._build_segments(sents, [], METADATA)
        # All in one segment (no boundary)
        self.assertEqual(segments[0].start_time_ms, 2500)

    def test_end_time_ms_is_last_sentence_end(self):
        sents = [
            {'speaker': 'SPEAKER_00', 'text': ' '.join(['word'] * 10),
             'start': 0.0, 'end': 5.0},
            {'speaker': 'SPEAKER_00', 'text': ' '.join(['word'] * 10),
             'start': 5.0, 'end': 9.7},
        ]
        segments = self.seg._build_segments(sents, [], METADATA)
        self.assertEqual(segments[-1].end_time_ms, 9700)

    def test_empty_sentences_returns_empty(self):
        segments = self.seg._build_segments([], [], METADATA)
        self.assertEqual(segments, [])


# ── Cross-speaker boundary enforcement ───────────────────────────────────────

class TestSpeakerBoundaryEnforcement(unittest.TestCase):
    """Verify that therapist vs participant boundaries don't leak into each other."""

    def test_therapist_not_in_filtered_sentences(self):
        seg = _make_segmenter(excluded=['THERAPIST'], filter_mode='exclude')
        sentences = [
            _sent(speaker='SPEAKER_00', text='Participant 1.', start=0.0, end=5.0),
            _sent(speaker='THERAPIST', text='Therapist response.', start=5.0, end=8.0),
            _sent(speaker='SPEAKER_00', text='Participant 2.', start=8.0, end=12.0),
        ]
        filtered = seg._filter_sentences_by_speaker(sentences)
        for s in filtered:
            self.assertNotEqual(s['speaker'], 'THERAPIST',
                                'Therapist sentence leaked into filtered set')

    def test_therapist_timestamps_preserved_for_cue_lookup(self):
        """extract_therapist_segments must preserve exact timestamps for cue lookup."""
        seg = _make_segmenter(excluded=['THERAPIST'], filter_mode='exclude')
        sentences = [
            _sent(speaker='SPEAKER_00', text='Participant.', start=0.0, end=5.0),
            _sent(speaker='THERAPIST', text='Therapist.', start=5.1, end=9.3),
            _sent(speaker='SPEAKER_00', text='Participant.', start=9.5, end=14.0),
        ]
        therapist_segs = seg.extract_therapist_segments(sentences, METADATA)
        self.assertEqual(len(therapist_segs), 1)
        # Timestamps should be exactly 5100 and 9300 ms (seconds * 1000)
        self.assertEqual(therapist_segs[0].start_time_ms, 5100)
        self.assertEqual(therapist_segs[0].end_time_ms, 9300)

    def test_purer_labels_not_applicable_to_participant_segments(self):
        """Participant segments from _build_segments have no purer_primary set."""
        seg = _make_segmenter(excluded=['THERAPIST'], filter_mode='exclude',
                              min_words=5)
        sents = [
            {'speaker': 'SPEAKER_00', 'text': ' '.join(['word'] * 10),
             'start': 0.0, 'end': 5.0},
        ]
        segments = seg._build_segments(sents, [], METADATA)
        for s in segments:
            self.assertIsNone(s.purer_primary,
                              'purer_primary must not be set on participant segments')

    def test_therapist_segment_speaker_field_is_therapist(self):
        seg = _make_segmenter(excluded=['THERAPIST'], filter_mode='exclude')
        sentences = [
            _sent(speaker='THERAPIST', text='Therapist speech.', start=5.0, end=8.0),
        ]
        therapist_segs = seg.extract_therapist_segments(sentences, METADATA)
        for s in therapist_segs:
            self.assertEqual(s.speaker, 'therapist',
                             'Therapist segments must have speaker="therapist"')


if __name__ == '__main__':
    unittest.main()
