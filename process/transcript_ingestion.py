"""
transcript_ingestion.py
-----------------------
Transcript ingestion and segmentation.

Consumes diarized transcripts from the whisper-diarization-batchprocess
pipeline and produces coherent segments for classification.

- Imports Segment/SpeakerRun from classification_tools.data_structures
"""

import json
import os
import glob
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from classification_tools.data_structures import Segment, SpeakerRun


class TranscriptSegmenter:
    """
    Groups diarized sentences into coherent segments for classification.

    Uses embedding-based similarity curves and pause signals with length
    constraints tuned for therapeutic dialogue.
    """

    def __init__(self, config: dict):
        self.embedding_model = SentenceTransformer(
            config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        self.min_words = config.get('min_segment_words', 30)
        self.max_words = config.get('max_segment_words', 200)
        self.silence_threshold_ms = config.get('silence_threshold_ms', 1500)
        self.semantic_shift_percentile = config.get('semantic_shift_percentile', 25)

    def segment_session(
        self, sentences: List[Dict], session_metadata: Dict
    ) -> List[Segment]:
        """
        Takes a list of sentence dicts from the diarization pipeline
        and returns Segment objects suitable for classification.

        Parameters
        ----------
        sentences : list of dict
            Each dict has keys: text, speaker, start, end (seconds).
        session_metadata : dict
            Keys: trial_id, participant_id, session_id, session_number.
        """
        if not sentences:
            return []

        participant_runs, therapist_runs = self._separate_speaker_runs(
            sentences, session_metadata
        )

        all_segments: List[Segment] = []
        seg_counter = 0

        for run in participant_runs + therapist_runs:
            if not run.sentences:
                continue

            texts = [s['text'] for s in run.sentences]
            embeddings = self.embedding_model.encode(
                texts, normalize_embeddings=True
            )

            sim_curve = self._compute_similarity_curve(embeddings, window=3)
            pause_curve = self._compute_pause_curve(run.sentences)

            boundaries = self._find_boundaries(
                sim_curve, pause_curve, run.sentences
            )

            segments = self._build_segments(
                run.sentences, boundaries, run.speaker,
                session_metadata, seg_counter
            )
            seg_counter += len(segments)
            all_segments.extend(segments)

        all_segments.sort(key=lambda s: s.start_time_ms)
        for i, seg in enumerate(all_segments):
            seg.segment_index = i

        return all_segments

    # ------------------------------------------------------------------
    # Similarity and pause curves
    # ------------------------------------------------------------------

    def _compute_similarity_curve(
        self, embeddings: np.ndarray, window: int = 3
    ) -> np.ndarray:
        n = len(embeddings)
        if n <= 1:
            return np.array([])
        sims = np.zeros(n - 1)
        for i in range(n - 1):
            left_start = max(0, i - window + 1)
            left_emb = embeddings[left_start:i + 1].mean(axis=0)
            right_end = min(n, i + 1 + window)
            right_emb = embeddings[i + 1:right_end].mean(axis=0)
            sims[i] = float(np.dot(left_emb, right_emb))
        return sims

    def _compute_pause_curve(self, sentences: List[Dict]) -> np.ndarray:
        n = len(sentences)
        if n <= 1:
            return np.array([])
        pauses = np.zeros(n - 1)
        for i in range(n - 1):
            gap_ms = max(
                0,
                (sentences[i + 1].get('start', 0) - sentences[i].get('end', 0)) * 1000
            )
            pauses[i] = gap_ms
        return pauses

    # ------------------------------------------------------------------
    # Boundary detection
    # ------------------------------------------------------------------

    def _find_boundaries(
        self, sim_curve: np.ndarray, pause_curve: np.ndarray,
        sentences: List[Dict]
    ) -> List[int]:
        if len(sim_curve) == 0:
            return []

        threshold = np.percentile(sim_curve, self.semantic_shift_percentile)
        boundaries: List[int] = []

        for i in range(len(sim_curve)):
            is_sim_dip = sim_curve[i] < threshold
            is_long_pause = (
                i < len(pause_curve) and
                pause_curve[i] > self.silence_threshold_ms
            )
            if is_sim_dip or is_long_pause:
                if self._boundary_valid(i, boundaries, sentences):
                    boundaries.append(i)

        return boundaries

    def _boundary_valid(
        self, idx: int, existing: List[int], sentences: List[Dict]
    ) -> bool:
        if existing:
            prev_boundary = existing[-1]
            words_since = sum(
                len(s['text'].split())
                for s in sentences[prev_boundary + 1:idx + 1]
            )
            if words_since < self.min_words:
                return False
        return True

    # ------------------------------------------------------------------
    # Segment construction
    # ------------------------------------------------------------------

    def _build_segments(
        self, sentences: List[Dict], boundaries: List[int],
        speaker: str, metadata: Dict, offset: int
    ) -> List[Segment]:
        segments: List[Segment] = []
        starts = [0] + [b + 1 for b in boundaries]
        ends = [b + 1 for b in boundaries] + [len(sentences)]

        for start, end in zip(starts, ends):
            seg_sentences = sentences[start:end]
            if not seg_sentences:
                continue

            text = " ".join(s['text'] for s in seg_sentences)
            word_count = len(text.split())

            if word_count > self.max_words:
                mid = len(seg_sentences) // 2
                segments.extend(self._build_segments(
                    seg_sentences[:mid], [], speaker, metadata, offset + len(segments)
                ))
                segments.extend(self._build_segments(
                    seg_sentences[mid:], [], speaker, metadata, offset + len(segments)
                ))
                continue

            segment = Segment(
                segment_id=self._make_segment_id(metadata, offset + len(segments)),
                trial_id=metadata['trial_id'],
                participant_id=metadata['participant_id'],
                session_id=metadata['session_id'],
                session_number=metadata['session_number'],
                start_time_ms=int(seg_sentences[0].get('start', 0) * 1000),
                end_time_ms=int(seg_sentences[-1].get('end', 0) * 1000),
                speaker=speaker,
                text=text,
                word_count=word_count,
            )
            segments.append(segment)

        segments = self._merge_undersized(segments)
        return segments

    def _make_segment_id(self, metadata: Dict, idx: int) -> str:
        return (
            f"{metadata['trial_id']}_{metadata['participant_id']}_"
            f"S{metadata['session_number']:02d}_{idx:04d}"
        )

    def _merge_undersized(self, segments: List[Segment]) -> List[Segment]:
        if len(segments) <= 1:
            return segments
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg.word_count < self.min_words and merged:
                prev = merged[-1]
                prev.text = prev.text + " " + seg.text
                prev.word_count = len(prev.text.split())
                prev.end_time_ms = seg.end_time_ms
            else:
                merged.append(seg)
        return merged

    # ------------------------------------------------------------------
    # Speaker separation
    # ------------------------------------------------------------------

    def _separate_speaker_runs(
        self, sentences: List[Dict], metadata: Dict
    ) -> Tuple[List[SpeakerRun], List[SpeakerRun]]:
        participant_runs: List[SpeakerRun] = []
        therapist_runs: List[SpeakerRun] = []
        current_speaker = None
        current_run: List[Dict] = []

        for sent in sentences:
            spk = sent.get('speaker', 'unknown')
            if spk != current_speaker:
                if current_run and current_speaker is not None:
                    run = SpeakerRun(current_speaker, current_run)
                    if self._is_participant(current_speaker):
                        participant_runs.append(run)
                    else:
                        therapist_runs.append(run)
                current_speaker = spk
                current_run = [sent]
            else:
                current_run.append(sent)

        if current_run and current_speaker is not None:
            run = SpeakerRun(current_speaker, current_run)
            if self._is_participant(current_speaker):
                participant_runs.append(run)
            else:
                therapist_runs.append(run)

        return participant_runs, therapist_runs

    @staticmethod
    def _is_participant(speaker_label: str) -> bool:
        label = speaker_label.lower()
        return 'participant' in label or 'patient' in label or label == 'speaker_0'


# ---------------------------------------------------------------------------
# Session loading utilities
# ---------------------------------------------------------------------------

def load_diarized_session(session_path: str) -> Dict[str, Any]:
    """
    Load a diarized session JSON file produced by the batch diarization repo.

    The diarization pipeline outputs result.json files containing:
      - metadata: {duration, language, num_speakers, ...}
      - sentences: [{text, speaker, start, end, words}, ...]
    """
    with open(session_path, 'r') as f:
        data = json.load(f)

    sentences = data.get('sentences', [])
    if not sentences and 'sections' in data:
        for section in data['sections']:
            sentences.extend(section.get('sentences', []))

    metadata = data.get('metadata', {})
    return {'sentences': sentences, 'metadata': metadata}


def discover_session_files(
    input_dir: str,
    pattern: str = '**/result.json'
) -> List[str]:
    """Find all diarized session JSON files in a directory tree."""
    return sorted(glob.glob(os.path.join(input_dir, pattern), recursive=True))
