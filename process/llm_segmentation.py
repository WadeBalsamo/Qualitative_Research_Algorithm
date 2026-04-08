"""
llm_segmentation.py
-------------------
LLM-based segmentation refinement.

A post-processing pass that takes segments from ConversationalSegmenter
and refines boundaries using LLM judgment. Deliberately separate from
the segmenter for composability.

Modes:
  - boundary_review:     Re-evaluates ambiguous boundaries (split/merge)
  - cross_speaker_merge: Merges adjacent cross-speaker segments that form
                         conversational units (Q&A, elaboration)
  - full:                Both passes
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from classification_tools.data_structures import Segment
from classification_tools.llm_client import LLMClient, extract_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_BOUNDARY_REVIEW_PROMPT = """\
You are an expert qualitative researcher analyzing therapeutic group dialogue.

Below are {n_boundaries} potential segment boundaries from a transcript. For each boundary, I provide ~3-4 sentences of context before and after the boundary point.

For each boundary, decide: Is this a genuine topic/idea shift, or a continuation of the same thought?

Respond with a JSON array of objects, one per boundary:
[
  {{"boundary_id": <int>, "decision": "split" | "merge", "reason": "<brief explanation>"}}
]

Boundaries to evaluate:

{boundaries_text}
"""

_CROSS_SPEAKER_MERGE_PROMPT = """\
You are an expert qualitative researcher analyzing therapeutic group dialogue.

Below are {n_pairs} pairs of adjacent single-speaker segments from different speakers. For each pair, decide if they form a conversational unit that should be merged.

Criteria for merging:
- Direct Q&A exchange
- Elaboration or agreement on the same topic
- Emotional continuation or empathic response

Respond with a JSON array of objects, one per pair:
[
  {{"pair_id": <int>, "merge": true | false, "relationship": "qa" | "elaboration" | "agreement" | "empathy" | "unrelated", "reason": "<brief explanation>"}}
]

Pairs to evaluate:

{pairs_text}
"""

_COHERENCE_CHECK_PROMPT = """\
You are an expert qualitative researcher analyzing therapeutic group dialogue.

The following segment appears to contain multiple distinct ideas or topics. Should it be split? If so, at which sentence boundary?

Segment text:
{segment_text}

Sentences are numbered for reference:
{numbered_sentences}

Respond with JSON:
{{
  "should_split": true | false,
  "split_after_sentence": <int or null>,
  "reason": "<brief explanation>"
}}
"""


class LLMSegmentationRefiner:
    """Refines segment boundaries using LLM judgment.

    Takes output from ConversationalSegmenter and applies targeted LLM
    evaluation to ambiguous boundaries and cross-speaker merge candidates.
    """

    def __init__(self, llm_client: LLMClient, config: dict):
        self.client = llm_client
        self.mode = config.get('mode', 'boundary_review')
        self.ambiguity_threshold = config.get('ambiguity_threshold', 0.15)
        self.batch_size = config.get('batch_size', 5)
        self.max_gap_seconds = config.get('max_gap_seconds', 30.0)
        self.merge_sim_threshold = config.get('merge_sim_threshold', 0.5)

    def refine(
        self,
        segments: List[Segment],
        sentences: List[Dict],
        sim_curve: np.ndarray,
        embeddings: np.ndarray,
        boundary_confidence: Optional[Dict[int, str]] = None,
    ) -> List[Segment]:
        """Refine segment boundaries using LLM judgment.

        Parameters
        ----------
        segments : list of Segment
            Segments from ConversationalSegmenter.
        sentences : list of dict
            Original sentence dicts (text, speaker, start, end).
        sim_curve : np.ndarray
            Narrow-window similarity curve.
        embeddings : np.ndarray
            Sentence embeddings.
        boundary_confidence : dict, optional
            Map of boundary_index -> 'confident'|'ambiguous' from the
            segmenter's dual-window analysis.

        Returns
        -------
        list of Segment
            Refined segment list.
        """
        if not segments:
            return segments

        refined = list(segments)

        if self.mode in ('boundary_review', 'full'):
            refined = self._review_boundaries(
                refined, sentences, sim_curve, boundary_confidence
            )

        if self.mode in ('cross_speaker_merge', 'full'):
            refined = self._cross_speaker_merge(
                refined, embeddings
            )

        # Re-check for oversized segments after merging
        refined = self._check_coherence(refined, sentences)

        # Re-index
        for i, seg in enumerate(refined):
            seg.segment_index = i

        return refined

    # ------------------------------------------------------------------
    # Boundary review
    # ------------------------------------------------------------------

    def _review_boundaries(
        self,
        segments: List[Segment],
        sentences: List[Dict],
        sim_curve: np.ndarray,
        boundary_confidence: Optional[Dict[int, str]],
    ) -> List[Segment]:
        """Re-evaluate ambiguous boundaries via LLM."""
        if boundary_confidence is None:
            return segments

        ambiguous = [
            idx for idx, conf in boundary_confidence.items()
            if conf == 'ambiguous'
        ]
        if not ambiguous:
            logger.info("No ambiguous boundaries to review")
            return segments

        logger.info(
            "Reviewing %d ambiguous boundaries (of %d total)",
            len(ambiguous), len(boundary_confidence),
        )

        # Batch ambiguous boundaries
        merge_indices = set()
        for batch_start in range(0, len(ambiguous), self.batch_size):
            batch = ambiguous[batch_start:batch_start + self.batch_size]
            decisions = self._batch_boundary_review(batch, sentences)
            for idx, decision in zip(batch, decisions):
                if decision == 'merge':
                    merge_indices.add(idx)

        if not merge_indices:
            return segments

        # Apply merge decisions: merge segment at boundary with previous
        return self._apply_boundary_merges(segments, merge_indices, sentences)

    def _batch_boundary_review(
        self, boundary_indices: List[int], sentences: List[Dict],
    ) -> List[str]:
        """Send a batch of boundaries to LLM for review.

        Returns a list of decisions ('split' or 'merge') in the same
        order as boundary_indices.
        """
        context_radius = 2  # sentences before/after boundary
        boundaries_text_parts = []

        for bid, idx in enumerate(boundary_indices):
            before_start = max(0, idx - context_radius)
            after_end = min(len(sentences), idx + 1 + context_radius)

            before_text = " | ".join(
                f"[{s.get('speaker', '?')}]: {s['text']}"
                for s in sentences[before_start:idx + 1]
            )
            after_text = " | ".join(
                f"[{s.get('speaker', '?')}]: {s['text']}"
                for s in sentences[idx + 1:after_end]
            )

            boundaries_text_parts.append(
                f"--- Boundary {bid} (sentence index {idx}) ---\n"
                f"BEFORE: {before_text}\n"
                f"AFTER:  {after_text}\n"
            )

        prompt = _BOUNDARY_REVIEW_PROMPT.format(
            n_boundaries=len(boundary_indices),
            boundaries_text="\n".join(boundaries_text_parts),
        )

        response_text, _ = self.client.request(prompt)
        if not response_text:
            return ['split'] * len(boundary_indices)

        try:
            parsed = extract_json(response_text)
            if isinstance(parsed, list):
                decisions = ['split'] * len(boundary_indices)
                for item in parsed:
                    bid = item.get('boundary_id', -1)
                    if 0 <= bid < len(boundary_indices):
                        decisions[bid] = item.get('decision', 'split')
                return decisions
        except Exception:
            pass

        return ['split'] * len(boundary_indices)

    def _apply_boundary_merges(
        self,
        segments: List[Segment],
        merge_boundary_indices: set,
        sentences: List[Dict],
    ) -> List[Segment]:
        """Merge segments at boundary positions the LLM flagged.

        We map boundary sentence indices to segment pairs by checking
        which segment ends near each boundary index.
        """
        if not merge_boundary_indices or len(segments) < 2:
            return segments

        # Build a map: segment index -> set of sentence time ranges
        # We merge segment i with segment i+1 if the boundary between them
        # is in merge_boundary_indices
        segments_to_merge = set()
        for i in range(len(segments) - 1):
            seg_end_ms = segments[i].end_time_ms
            next_start_ms = segments[i + 1].start_time_ms
            # Check if any merge-boundary sentence falls between these segments
            for bidx in merge_boundary_indices:
                if bidx < len(sentences):
                    sent_end = int(sentences[bidx].get('end', 0) * 1000)
                    sent_next_start = int(sentences[min(bidx + 1, len(sentences) - 1)].get('start', 0) * 1000)
                    if abs(sent_end - seg_end_ms) < 2000 or abs(sent_next_start - next_start_ms) < 2000:
                        segments_to_merge.add(i)
                        break

        if not segments_to_merge:
            return segments

        merged = []
        skip = False
        for i, seg in enumerate(segments):
            if skip:
                skip = False
                continue
            if i in segments_to_merge and i + 1 < len(segments):
                next_seg = segments[i + 1]
                seg.text = seg.text + "\n" + next_seg.text
                seg.word_count = len(seg.text.split())
                seg.end_time_ms = next_seg.end_time_ms
                if seg.speaker != next_seg.speaker:
                    seg.speaker = "multiple"
                if seg.speakers_in_segment and next_seg.speakers_in_segment:
                    seen = set(seg.speakers_in_segment)
                    for sp in next_seg.speakers_in_segment:
                        if sp not in seen:
                            seg.speakers_in_segment.append(sp)
                            seen.add(sp)
                skip = True
            merged.append(seg)

        logger.info("Merged %d boundary pairs", len(segments_to_merge))
        return merged

    # ------------------------------------------------------------------
    # Cross-speaker merge
    # ------------------------------------------------------------------

    def _cross_speaker_merge(
        self,
        segments: List[Segment],
        embeddings: np.ndarray,
    ) -> List[Segment]:
        """Evaluate adjacent cross-speaker segment pairs for merging."""
        candidates = self._find_merge_candidates(segments, embeddings)
        if not candidates:
            logger.info("No cross-speaker merge candidates found")
            return segments

        logger.info("Evaluating %d cross-speaker merge candidates", len(candidates))

        merge_set = set()
        for batch_start in range(0, len(candidates), self.batch_size):
            batch = candidates[batch_start:batch_start + self.batch_size]
            decisions = self._batch_cross_speaker_review(batch, segments)
            for (seg_idx, _), should_merge in zip(batch, decisions):
                if should_merge:
                    merge_set.add(seg_idx)

        if not merge_set:
            return segments

        # Apply merges
        merged = []
        skip = False
        for i, seg in enumerate(segments):
            if skip:
                skip = False
                continue
            if i in merge_set and i + 1 < len(segments):
                next_seg = segments[i + 1]
                seg.text = seg.text + "\n" + next_seg.text
                seg.word_count = len(seg.text.split())
                seg.end_time_ms = next_seg.end_time_ms
                seg.speaker = "multiple"
                if seg.speakers_in_segment and next_seg.speakers_in_segment:
                    seen = set(seg.speakers_in_segment)
                    for sp in next_seg.speakers_in_segment:
                        if sp not in seen:
                            seg.speakers_in_segment.append(sp)
                            seen.add(sp)
                elif next_seg.speakers_in_segment:
                    seg.speakers_in_segment = list(next_seg.speakers_in_segment)
                skip = True
            merged.append(seg)

        logger.info("Cross-speaker merged %d pairs", len(merge_set))
        return merged

    def _find_merge_candidates(
        self, segments: List[Segment], embeddings: np.ndarray,
    ) -> List[Tuple[int, float]]:
        """Find pairs of adjacent segments from different speakers
        that are close in embedding space and time."""
        candidates = []
        for i in range(len(segments) - 1):
            seg_a, seg_b = segments[i], segments[i + 1]
            # Must be different speakers
            if seg_a.speaker == seg_b.speaker:
                continue
            # Check temporal proximity
            gap_s = (seg_b.start_time_ms - seg_a.end_time_ms) / 1000.0
            if gap_s > self.max_gap_seconds:
                continue
            # Compute similarity between segment embeddings (average of sentence embeddings)
            # Use the segment text length as a rough proxy for sentence count
            # For a more precise approach, we'd need sentence-to-segment mapping
            sim = self._segment_pair_similarity(seg_a, seg_b, embeddings, segments)
            if sim > self.merge_sim_threshold:
                candidates.append((i, sim))

        return candidates

    def _segment_pair_similarity(
        self, seg_a: Segment, seg_b: Segment,
        embeddings: np.ndarray, all_segments: List[Segment],
    ) -> float:
        """Estimate similarity between two segments using a simple
        embedding centroid approach."""
        # Build approximate sentence index ranges for each segment
        # Since we don't have exact mapping, use a heuristic based on position
        total_segs = len(all_segments)
        total_embs = len(embeddings)
        if total_segs == 0 or total_embs == 0:
            return 0.0

        idx_a = all_segments.index(seg_a) if seg_a in all_segments else -1
        idx_b = all_segments.index(seg_b) if seg_b in all_segments else -1
        if idx_a < 0 or idx_b < 0:
            return 0.0

        # Rough partition of embeddings across segments
        avg_per_seg = total_embs / total_segs
        start_a = int(idx_a * avg_per_seg)
        end_a = int((idx_a + 1) * avg_per_seg)
        start_b = int(idx_b * avg_per_seg)
        end_b = int((idx_b + 1) * avg_per_seg)

        start_a = max(0, min(start_a, total_embs - 1))
        end_a = max(start_a + 1, min(end_a, total_embs))
        start_b = max(0, min(start_b, total_embs - 1))
        end_b = max(start_b + 1, min(end_b, total_embs))

        emb_a = embeddings[start_a:end_a].mean(axis=0)
        emb_b = embeddings[start_b:end_b].mean(axis=0)

        return float(np.dot(emb_a, emb_b))

    def _batch_cross_speaker_review(
        self,
        candidates: List[Tuple[int, float]],
        segments: List[Segment],
    ) -> List[bool]:
        """Send a batch of cross-speaker pairs to LLM for review."""
        pairs_text_parts = []
        for pid, (seg_idx, sim) in enumerate(candidates):
            seg_a = segments[seg_idx]
            seg_b = segments[seg_idx + 1]
            # Truncate long texts for the prompt
            text_a = seg_a.text[:500] if len(seg_a.text) > 500 else seg_a.text
            text_b = seg_b.text[:500] if len(seg_b.text) > 500 else seg_b.text
            pairs_text_parts.append(
                f"--- Pair {pid} ---\n"
                f"Segment A ({seg_a.speaker}):\n{text_a}\n\n"
                f"Segment B ({seg_b.speaker}):\n{text_b}\n"
            )

        prompt = _CROSS_SPEAKER_MERGE_PROMPT.format(
            n_pairs=len(candidates),
            pairs_text="\n".join(pairs_text_parts),
        )

        response_text, _ = self.client.request(prompt)
        if not response_text:
            return [False] * len(candidates)

        try:
            parsed = extract_json(response_text)
            if isinstance(parsed, list):
                decisions = [False] * len(candidates)
                for item in parsed:
                    pid = item.get('pair_id', -1)
                    if 0 <= pid < len(candidates):
                        decisions[pid] = bool(item.get('merge', False))
                return decisions
        except Exception:
            pass

        return [False] * len(candidates)

    # ------------------------------------------------------------------
    # Coherence check for long/merged segments
    # ------------------------------------------------------------------

    def _check_coherence(
        self,
        segments: List[Segment],
        sentences: List[Dict],
    ) -> List[Segment]:
        """Check unusually long segments for coherence and split if needed."""
        result = []
        for seg in segments:
            # Only check segments with >300 words
            if seg.word_count > 300:
                split_result = self._coherence_check_single(seg)
                if split_result:
                    result.extend(split_result)
                    continue
            result.append(seg)
        return result

    def _coherence_check_single(self, seg: Segment) -> Optional[List[Segment]]:
        """Ask LLM whether a large segment should be split."""
        lines = seg.text.split('\n')
        if len(lines) < 3:
            return None

        numbered = "\n".join(f"  {i+1}. {line}" for i, line in enumerate(lines))

        prompt = _COHERENCE_CHECK_PROMPT.format(
            segment_text=seg.text[:1500],
            numbered_sentences=numbered[:1500],
        )

        response_text, _ = self.client.request(prompt)
        if not response_text:
            return None

        try:
            parsed = extract_json(response_text)
            if isinstance(parsed, dict) and parsed.get('should_split'):
                split_after = parsed.get('split_after_sentence')
                if split_after and 0 < split_after < len(lines):
                    return self._split_segment_at_line(seg, split_after)
        except Exception:
            pass

        return None

    def _split_segment_at_line(
        self, seg: Segment, split_after: int,
    ) -> List[Segment]:
        """Split a segment at a given line number."""
        lines = seg.text.split('\n')
        text_a = '\n'.join(lines[:split_after])
        text_b = '\n'.join(lines[split_after:])

        if not text_a.strip() or not text_b.strip():
            return [seg]

        seg_a = Segment(
            segment_id=seg.segment_id,
            trial_id=seg.trial_id,
            participant_id=seg.participant_id,
            session_id=seg.session_id,
            session_number=seg.session_number,
            start_time_ms=seg.start_time_ms,
            end_time_ms=seg.start_time_ms + (seg.end_time_ms - seg.start_time_ms) // 2,
            speaker=seg.speaker,
            text=text_a,
            word_count=len(text_a.split()),
            speakers_in_segment=seg.speakers_in_segment,
            session_file=seg.session_file,
        )
        seg_b = Segment(
            segment_id=seg.segment_id + '_split',
            trial_id=seg.trial_id,
            participant_id=seg.participant_id,
            session_id=seg.session_id,
            session_number=seg.session_number,
            start_time_ms=seg_a.end_time_ms,
            end_time_ms=seg.end_time_ms,
            speaker=seg.speaker,
            text=text_b,
            word_count=len(text_b.split()),
            speakers_in_segment=seg.speakers_in_segment,
            session_file=seg.session_file,
        )
        return [seg_a, seg_b]
