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


class ConversationalSegmenter:
    """
    Groups multi-speaker utterances into coherent conversational segments.

    Unlike TranscriptSegmenter (which processes each speaker's run
    independently), this segmenter treats the full transcript as a stream,
    grouping consecutive utterances from ALL speakers into topically coherent
    chunks.  Each output segment's text preserves speaker attributions:

        Bobby: I was thinking about what you said last week...
        Marie: That resonates with me too...

    Excluded speakers' sentences are stripped *before* segmentation so their
    content never enters the semantic grouping or output text.

    This is appropriate for group therapy / coaching sessions where meaning
    emerges from multi-party exchanges rather than individual monologues.
    """

    def __init__(self, config: dict):
        self.embedding_model = SentenceTransformer(
            config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        self.min_words = config.get('min_segment_words_conversational', 60)
        self.max_words = config.get('max_segment_words_conversational', 400)
        self.silence_threshold_ms = config.get('silence_threshold_ms', 2000)
        self.semantic_shift_percentile = config.get('semantic_shift_percentile', 25)
        # Advanced grouping parameters
        self.group_cross_speaker = config.get('group_cross_speaker', False)
        self.max_gap_seconds = config.get('max_gap_seconds', 30.0)
        self.min_words_per_sentence = config.get('min_words_per_sentence', 10)
        self.max_segment_duration_seconds = config.get('max_segment_duration_seconds', 300.0)
        # Speaker filtering (applied before segmentation)
        self.excluded_speakers: List[str] = config.get('excluded_speakers', [])
        self.isolated_speakers: List[str] = config.get('isolated_speakers', [])

    def segment_session(
        self, sentences: List[Dict], session_metadata: Dict
    ) -> List['Segment']:
        """
        Segment a session transcript into multi-speaker conversational chunks.

        Excluded/isolated speakers are filtered at the sentence level *before*
        semantic grouping, so their content never enters segments.

        Parameters
        ----------
        sentences : list of dict
            Each dict: {text, speaker, start, end}.
        session_metadata : dict
            Keys: trial_id, participant_id, session_id, session_number.
        """
        if not sentences:
            return []

        # --- Sentence-level speaker filtering ---
        sentences = self._filter_sentences_by_speaker(sentences)

        # --- Filter short sentences (below min_words_per_sentence) ---
        sentences = [
            s for s in sentences
            if len(s.get('text', '').split()) >= self.min_words_per_sentence
        ]

        if not sentences:
            return []

        texts = [s['text'] for s in sentences]
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)

        sim_curve = self._compute_similarity_curve(embeddings)
        pause_curve = self._compute_pause_curve(sentences)
        boundaries = self._find_boundaries(sim_curve, pause_curve, sentences)

        segments = self._build_segments(sentences, boundaries, session_metadata)
        for i, seg in enumerate(segments):
            seg.segment_index = i
        return segments

    def _filter_sentences_by_speaker(self, sentences: List[Dict]) -> List[Dict]:
        """Remove sentences from excluded speakers or keep only isolated speakers."""
        if self.excluded_speakers:
            excluded_set = set(self.excluded_speakers)
            sentences = [s for s in sentences if s.get('speaker', '') not in excluded_set]
        elif self.isolated_speakers:
            isolated_set = set(self.isolated_speakers)
            sentences = [s for s in sentences if s.get('speaker', '') in isolated_set]
        return sentences

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

    def _find_boundaries(
        self, sim_curve: np.ndarray, pause_curve: np.ndarray,
        sentences: List[Dict]
    ) -> List[int]:
        if len(sim_curve) == 0:
            return []
        threshold = np.percentile(sim_curve, self.semantic_shift_percentile)
        max_gap_ms = self.max_gap_seconds * 1000
        boundaries: List[int] = []
        for i in range(len(sim_curve)):
            is_sim_dip = sim_curve[i] < threshold
            is_long_pause = (
                i < len(pause_curve) and
                pause_curve[i] > self.silence_threshold_ms
            )
            # Force boundary when time gap exceeds max_gap_seconds
            is_gap_exceeded = (
                i < len(pause_curve) and
                pause_curve[i] > max_gap_ms
            )
            # Force boundary on speaker change when group_cross_speaker is False
            is_speaker_change = (
                not self.group_cross_speaker and
                i + 1 < len(sentences) and
                sentences[i].get('speaker', '') != sentences[i + 1].get('speaker', '')
            )
            should_break = is_gap_exceeded or is_speaker_change or is_sim_dip or is_long_pause
            if should_break and self._boundary_valid(i, boundaries, sentences):
                boundaries.append(i)
        return boundaries

    def _boundary_valid(
        self, idx: int, existing: List[int], sentences: List[Dict]
    ) -> bool:
        if existing:
            words_since = sum(
                len(s['text'].split())
                for s in sentences[existing[-1] + 1:idx + 1]
            )
            if words_since < self.min_words:
                return False
        return True

    def _build_segments(
        self, sentences: List[Dict], boundaries: List[int],
        metadata: Dict, offset: int = 0
    ) -> List['Segment']:
        starts = [0] + [b + 1 for b in boundaries]
        ends = [b + 1 for b in boundaries] + [len(sentences)]
        segments: List['Segment'] = []

        for start, end in zip(starts, ends):
            chunk = sentences[start:end]
            if not chunk:
                continue

            # Build speaker-attributed text
            lines = [f"{s['speaker']}: {s['text']}" for s in chunk]
            text = "\n".join(lines)
            word_count = sum(len(s['text'].split()) for s in chunk)

            # Split oversized segments (word count or duration)
            duration_s = (chunk[-1].get('end', 0) - chunk[0].get('start', 0))
            if word_count > self.max_words or duration_s > self.max_segment_duration_seconds:
                mid = len(chunk) // 2
                segments.extend(self._build_segments(
                    chunk[:mid], [], metadata, offset + len(segments)
                ))
                segments.extend(self._build_segments(
                    chunk[mid:], [], metadata, offset + len(segments)
                ))
                continue

            unique_speakers = list(dict.fromkeys(s['speaker'] for s in chunk))
            dominant_speaker = max(
                unique_speakers,
                key=lambda sp: sum(
                    len(s['text'].split()) for s in chunk if s['speaker'] == sp
                )
            )
            speaker_field = dominant_speaker if len(unique_speakers) == 1 else "multiple"

            seg = Segment(
                segment_id=self._make_segment_id(metadata, offset + len(segments)),
                trial_id=metadata['trial_id'],
                participant_id=metadata['participant_id'],
                session_id=metadata['session_id'],
                session_number=metadata['session_number'],
                start_time_ms=int(chunk[0].get('start', 0) * 1000),
                end_time_ms=int(chunk[-1].get('end', 0) * 1000),
                speaker=speaker_field,
                text=text,
                word_count=word_count,
                speakers_in_segment=unique_speakers,
                session_file=metadata.get('source_file', ''),
            )
            segments.append(seg)

        return self._merge_undersized(segments)

    def _make_segment_id(self, metadata: Dict, idx: int) -> str:
        return (
            f"{metadata['trial_id']}_{metadata['participant_id']}_"
            f"S{metadata['session_number']:02d}_{idx:04d}"
        )

    def _merge_undersized(self, segments: List['Segment']) -> List['Segment']:
        if len(segments) <= 1:
            return segments
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg.word_count < self.min_words and merged:
                prev = merged[-1]
                prev.text = prev.text + "\n" + seg.text
                prev.word_count = len(prev.text.split())
                prev.end_time_ms = seg.end_time_ms
                # Update speaker field
                if prev.speaker != seg.speaker:
                    prev.speaker = "multiple"
                # Merge speakers lists
                if prev.speakers_in_segment and seg.speakers_in_segment:
                    seen = set(prev.speakers_in_segment)
                    for sp in seg.speakers_in_segment:
                        if sp not in seen:
                            prev.speakers_in_segment.append(sp)
                            seen.add(sp)
            else:
                merged.append(seg)
        return merged


# ---------------------------------------------------------------------------
# Session loading utilities
# ---------------------------------------------------------------------------

def load_vtt_session(session_path: str) -> Dict[str, Any]:
    """
    Load a WebVTT transcript file where each cue has the format:

        <cue_number>
        HH:MM:SS.mmm --> HH:MM:SS.mmm
        Speaker Name: utterance text

    Returns the same ``{sentences, metadata}`` dict shape as
    ``load_diarized_session`` so the rest of the pipeline is unchanged.
    """
    import re

    def _ts_to_seconds(ts: str) -> float:
        parts = ts.strip().split(':')
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m, s = '0', parts[0], parts[1]
        else:
            return 0.0
        return int(h) * 3600 + int(m) * 60 + float(s.replace(',', '.'))

    ts_re = re.compile(
        r'^([\d:,\.]+)\s*-->\s*([\d:,\.]+)'
    )

    sentences: List[Dict] = []
    with open(session_path, 'r', encoding='utf-8') as fh:
        lines = fh.read().splitlines()

    i = 0
    # skip WEBVTT header
    if lines and lines[0].strip().upper().startswith('WEBVTT'):
        i = 1

    while i < len(lines):
        line = lines[i].strip()

        # skip blank lines and pure-numeric cue identifiers
        if not line or line.isdigit():
            i += 1
            continue

        m = ts_re.match(line)
        if m:
            start_s = _ts_to_seconds(m.group(1))
            end_s = _ts_to_seconds(m.group(2))
            i += 1
            # collect all text lines belonging to this cue
            text_parts: List[str] = []
            while i < len(lines) and lines[i].strip():
                text_parts.append(lines[i].strip())
                i += 1
            raw_text = ' '.join(text_parts)

            # split "Speaker Name: utterance"
            if ':' in raw_text:
                speaker, _, text = raw_text.partition(':')
                speaker = speaker.strip()
                text = text.strip()
            else:
                speaker = 'unknown'
                text = raw_text

            if text:
                sentences.append({
                    'text': text,
                    'speaker': speaker,
                    'start': start_s,
                    'end': end_s,
                })
        else:
            i += 1

    filename = os.path.splitext(os.path.basename(session_path))[0]
    metadata = {
        'source_file': session_path,
        'filename': filename,
    }
    return {'sentences': sentences, 'metadata': metadata}


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
    """Find all diarized session files (JSON or VTT) in a directory tree."""
    json_files = glob.glob(os.path.join(input_dir, pattern), recursive=True)
    vtt_files = glob.glob(os.path.join(input_dir, '**/*.vtt'), recursive=True)
    return sorted(set(json_files) | set(vtt_files))


def scan_speakers(input_dir: str) -> Dict[str, int]:
    """
    Scan all transcript files in *input_dir* and return a dict mapping
    speaker names to the number of utterances attributed to them.

    This is used by the setup wizard to present discovered speakers for
    exclusion/isolation before the pipeline runs.
    """
    session_files = discover_session_files(input_dir)
    speaker_counts: Dict[str, int] = {}

    for path in session_files:
        try:
            if path.lower().endswith('.vtt'):
                data = load_vtt_session(path)
            else:
                data = load_diarized_session(path)
            for sent in data.get('sentences', []):
                spk = sent.get('speaker', 'unknown')
                speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
        except Exception:
            continue

    # Sort by utterance count descending
    return dict(sorted(speaker_counts.items(), key=lambda kv: -kv[1]))
