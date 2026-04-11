"""
llm_segmentation.py
-------------------
LLM-based segmentation refinement.

A post-processing pass that takes participant-only segments from
ConversationalSegmenter and refines them using LLM judgment.
Deliberately separate from the segmenter for composability.

Modes:
  - boundary_review:    Re-evaluates ambiguous within-speaker boundaries
  - context_expansion:  Expands segment boundaries by merging relevant
                        surrounding dialogue (therapist AND participant)
  - coherence_check:    Splits oversized segments — embedding-first,
                        LLM only when ambiguous
  - full:               All three passes
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from classification_tools.data_structures import Segment
from classification_tools.llm_client import LLMClient, extract_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_GENERAL_PROMPT_PREFIX = """You are a qualitative research assistant. You are segmenting group psychotherapy session transcripts into coherent segments for qualitative coding and analysis. Each segment should ideally represent a single topic and complete expression, and may include multiple sentences or speakers in dialogue. Use the following information to improve segment coherence by clarifying concept boundaries and grouping necessary context. \n \n """

_BOUNDARY_REVIEW_PROMPT = _GENERAL_PROMPT_PREFIX + """\
\n Decide if each boundary is a topic change or the same idea continuing.
\n 
{boundaries_text}
\n 
Respond with JSON only: \n \
[{{"boundary_id": <int>, "decision": "split" or "merge"}}]"""

# Context expansion — three simple true/false prompts.
# No quoting; the model only answers {"answer": true} or {"answer": false}.

_CTX_BEFORE_NEEDED_PROMPT = _GENERAL_PROMPT_PREFIX + """\
\n A participant said:
\n "{participant_text}"
\n
\n The preceding conversation was:
\n "{preceding_dialogue}"
\n
\n Is the preceding conversation necessary to understand what the participant said?
\n JSON only: {{"answer": true}} or {{"answer": false}}"""

_CTX_AFTER_NEEDED_PROMPT = _GENERAL_PROMPT_PREFIX + """\
\n A participant said:
\n "{participant_text}"

\n The conversation continued:
\n"{following_dialogue}"

\n Does the following conversation need to be included to understand this participant's statement?
\n JSON only: {{"answer": true}} or {{"answer": false}}"""

_CTX_ADD_UTTERANCE_PROMPT = _GENERAL_PROMPT_PREFIX + """\
\n Current segment:
\n "{current_segment}"

\n Should the following utterance be added?
\n "{utterance_line}"

\n JSON only: {{"answer": true}} or {{"answer": false}}"""

_COHERENCE_CHECK_PROMPT = _GENERAL_PROMPT_PREFIX + """\
\n Does this segment contain two different topics? If yes, after which sentence should it split?

\n {numbered_sentences}

\n Respond with JSON only:
\n {{"should_split": true or false, "split_after_sentence": <int or null>}}"""


class LLMSegmentationRefiner:
    """Refines segment boundaries using LLM judgment.

    Takes participant-only segments from ConversationalSegmenter and applies
    targeted refinement. Therapist/excluded speech is accessed via the
    original unfiltered transcript sentences for context expansion only —
    no LLM calls are spent processing therapist-only segments.
    """

    def __init__(self, llm_client: LLMClient, config: dict,
                 speaker_normalizer=None):
        self.client = llm_client
        self.mode = config.get('mode', 'full')
        self.ambiguity_threshold = config.get('ambiguity_threshold', 0.15)
        self.batch_size = config.get('batch_size', 5)
        self.coherence_sim_threshold = config.get('coherence_sim_threshold', 0.3)
        self.excluded_speakers: List[str] = config.get('excluded_speakers', [])
        self._excluded_set: set = set(self.excluded_speakers)
        self.speaker_normalizer = speaker_normalizer
        # Guardrails for adaptive context expansion
        self.max_context_words = config.get('max_context_words', 400)
        self.max_context_duration_s = config.get('max_context_duration_s', 300.0)
        self.max_gap_seconds = config.get('max_gap_seconds', 30.0)
        # Embedding similarity thresholds for context gating
        self.context_attach_threshold = config.get('context_attach_threshold', 0.30)
        self.context_skip_threshold = config.get('context_skip_threshold', 0.22)
        # Embedding model ID for context gating and coherence checks (lazy-loaded)
        self.embedding_model_id = config.get('embedding_model', 'Qwen/Qwen3-Embedding-8B')
        self._embed_model = None
        # Process logger (optional)
        self.plog = config.get('process_logger', None)

    def _get_embed_model(self):
        """Lazy-load and cache the sentence embedding model."""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(
                self.embedding_model_id,
                trust_remote_code=True,
            )
        return self._embed_model

    def refine(
        self,
        segments: List[Segment],
        sentences: List[Dict],
        sim_curve: np.ndarray,
        embeddings: np.ndarray,
        boundary_confidence: Optional[Dict[int, str]] = None,
        original_sentences: Optional[List[Dict]] = None,
    ) -> List[Segment]:
        """Refine segment boundaries using LLM judgment.

        Parameters
        ----------
        segments : list of Segment
            Participant-only segments from ConversationalSegmenter.
        sentences : list of dict
            Filtered sentence dicts used by the segmenter.
        sim_curve : np.ndarray
            Narrow-window similarity curve.
        embeddings : np.ndarray
            Sentence embeddings (filtered sentences only).
        boundary_confidence : dict, optional
            Map of boundary_index -> 'confident'|'ambiguous'.
        original_sentences : list of dict, optional
            Full unfiltered transcript sentences (all speakers).
            Required for context_expansion mode.
        """
        if not segments:
            return segments

        refined = list(segments)

        if self.plog:
            self.plog.section("LLM SEGMENTATION REFINER")
            self.plog.log_segments("INPUT SEGMENTS", refined)

        # Normalize text format before any pass: all segments start with [participant_id]: ...
        # This ensures speaker labels are visible throughout boundary review,
        # context expansion, and coherence check.
        for seg in refined:
            if not seg.text.startswith('['):
                seg.text = f"[{seg.participant_id}]: {seg.text}"

        if self.mode in ('boundary_review', 'full'):
            if self.plog:
                self.plog.subsection("PASS 1: BOUNDARY REVIEW")
            refined = self._review_boundaries(
                refined, sentences, sim_curve, boundary_confidence
            )
            if self.plog:
                self.plog.log_segments("SEGMENTS AFTER BOUNDARY REVIEW", refined)

        if self.mode in ('context_expansion', 'full'):
            if self.plog:
                self.plog.subsection("PASS 2: CONTEXT EXPANSION")
            if original_sentences:
                refined = self._expand_segment_context(refined, original_sentences)
            else:
                logger.warning(
                    "context_expansion requested but no original_sentences provided; skipping"
                )
            if self.plog:
                self.plog.log_segments("SEGMENTS AFTER CONTEXT EXPANSION", refined)

        if self.mode in ('coherence_check', 'full'):
            if self.plog:
                self.plog.subsection("PASS 3: COHERENCE CHECK")
            refined = self._check_coherence(refined)
            if self.plog:
                self.plog.log_segments("SEGMENTS AFTER COHERENCE CHECK", refined)

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
        """Re-evaluate ambiguous within-speaker boundaries via LLM."""
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

        merge_indices = set()
        for batch_start in range(0, len(ambiguous), self.batch_size):
            batch = ambiguous[batch_start:batch_start + self.batch_size]
            decisions = self._batch_boundary_review(batch, sentences)
            for idx, decision in zip(batch, decisions):
                if decision == 'merge':
                    merge_indices.add(idx)

        if not merge_indices:
            return segments

        return self._apply_boundary_merges(segments, merge_indices, sentences)

    def _batch_boundary_review(
        self, boundary_indices: List[int], sentences: List[Dict],
    ) -> List[str]:
        """Send a batch of boundaries to LLM for review."""
        context_radius = 2
        boundaries_text_parts = []

        for bid, idx in enumerate(boundary_indices):
            before_start = max(0, idx - context_radius)
            after_end = min(len(sentences), idx + 1 + context_radius)

            before_text = " | ".join(
                f"[{self._get_anonymized_speaker(s.get('speaker', '?'))}]: {s['text']}"
                for s in sentences[before_start:idx + 1]
            )
            after_text = " | ".join(
                f"[{self._get_anonymized_speaker(s.get('speaker', '?'))}]: {s['text']}"
                for s in sentences[idx + 1:after_end]
            )

            boundaries_text_parts.append(
                f"--- Boundary {bid} (sentence index {idx}) ---\n"
                f"BEFORE: {before_text}\n"
                f"AFTER:  {after_text}\n"
            )

        prompt = _BOUNDARY_REVIEW_PROMPT.format(
            boundaries_text="\n".join(boundaries_text_parts),
        )

        response_text, _ = self.client.request(prompt)

        if self.plog:
            self.plog.log_llm_call(
                "boundary_review",
                prompt,
                response_text or '(empty)',
            )

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
                if self.plog:
                    for i, (idx, dec) in enumerate(zip(boundary_indices, decisions)):
                        self.plog._write(f"  boundary idx={idx} → {dec}")
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

        Only merges same-speaker segments to preserve single-speaker invariant.
        Segments already carry [speaker]: labels so they are joined with a
        newline rather than a space to keep the dialogue format intact.
        """
        if not merge_boundary_indices or len(segments) < 2:
            return segments

        segments_to_merge = set()
        for i in range(len(segments) - 1):
            if segments[i].speaker != segments[i + 1].speaker:
                continue
            seg_end_ms = segments[i].end_time_ms
            next_start_ms = segments[i + 1].start_time_ms
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
                # Join with newline; both sides already carry [speaker]: labels
                seg.text = seg.text + "\n" + next_seg.text
                seg.word_count = len(seg.text.split())
                seg.end_time_ms = next_seg.end_time_ms
                skip = True
            merged.append(seg)

        logger.info("Merged %d boundary pairs", len(segments_to_merge))
        return merged

    # ------------------------------------------------------------------
    # Context expansion
    # ------------------------------------------------------------------

    def _expand_segment_context(
        self,
        segments: List[Segment],
        original_sentences: List[Dict],
    ) -> List[Segment]:
        """Expand participant segments with adjacent dialogue.

        Uses embedding similarity as a first-pass gate:
        - HIGH similarity (>= context_attach_threshold) → attach mechanically
        - LOW similarity (< context_skip_threshold) → skip, unrelated
        - AMBIGUOUS → step-by-step LLM true/false questions

        In the ambiguous band, the model is first asked whether context in
        each direction is needed at all (one true/false call per direction),
        then asked about each candidate utterance one at a time — closest
        to the segment first — until it says false.

        Speaker names and speaker changes are shown in all prompts using
        [normalized_id]: label format, with consecutive same-speaker lines
        collapsed to avoid repeating the label.
        """
        if not original_sentences:
            return segments

        try:
            embed_model = self._get_embed_model()
        except ImportError:
            logger.warning("sentence_transformers not available; skipping context expansion")
            return segments

        stats = {'attached': 0, 'skipped': 0, 'llm': 0}

        for si, seg in enumerate(segments):
            seg_start_ms = seg.start_time_ms
            seg_end_ms = seg.end_time_ms

            # Find the range of original sentences that overlap this segment
            seg_orig_start = None
            seg_orig_end = None
            for oi, osent in enumerate(original_sentences):
                sent_start_ms = int(osent.get('start', 0) * 1000)
                sent_end_ms = int(osent.get('end', 0) * 1000)
                if sent_start_ms >= seg_start_ms and seg_orig_start is None:
                    seg_orig_start = oi
                if sent_end_ms <= seg_end_ms:
                    seg_orig_end = oi

            if seg_orig_start is None:
                continue

            actual_end = seg_orig_end or seg_orig_start

            # Walk backward/forward with guardrails
            seg_words = seg.word_count
            seg_duration = (seg_end_ms - seg_start_ms) / 1000.0
            word_budget = max(0, self.max_context_words - seg_words)
            duration_budget = max(0, self.max_context_duration_s - seg_duration)

            before_sents = self._walk_context(
                original_sentences, seg_orig_start, word_budget,
                duration_budget, seg_start_ms, direction='backward',
                current_participant_id=seg.participant_id,
            )
            after_sents = self._walk_context(
                original_sentences, actual_end, word_budget,
                duration_budget, seg_end_ms, direction='forward',
                current_participant_id=seg.participant_id,
            )

            if not before_sents and not after_sents:
                continue

            # Capture the participant's own text once, before any context is
            # prepended/appended, so LLM calls always see the original speech.
            raw_text = seg.text
            if raw_text.startswith('['):
                colon_pos = raw_text.find(']: ')
                if colon_pos > 0:
                    raw_text = raw_text[colon_pos + 3:]
            seg_emb = embed_model.encode([raw_text], normalize_embeddings=True)[0]

            for direction, context_sents in (('before', before_sents), ('after', after_sents)):
                if not context_sents:
                    continue

                context_text = " ".join(s.get('text', '') for s in context_sents)
                context_emb = embed_model.encode([context_text], normalize_embeddings=True)[0]
                sim = float(np.dot(seg_emb, context_emb))

                if sim >= self.context_attach_threshold:
                    # High similarity — attach mechanically
                    self._attach_context(seg, context_sents, direction)
                    stats['attached'] += 1
                    logger.debug(
                        "Context %s: sim=%.3f → attach (segment %s)", direction, sim, seg.segment_id
                    )
                    if self.plog:
                        self.plog.log_context_gate(
                            seg.segment_id, direction, sim, 'ATTACH (embedding)', len(context_sents)
                        )

                elif sim < self.context_skip_threshold:
                    # Low similarity — skip
                    stats['skipped'] += 1
                    logger.debug(
                        "Context %s: sim=%.3f → skip (segment %s)", direction, sim, seg.segment_id
                    )
                    if self.plog:
                        self.plog.log_context_gate(
                            seg.segment_id, direction, sim, 'SKIP (embedding)', len(context_sents)
                        )

                else:
                    # Ambiguous — step-by-step LLM
                    stats['llm'] += 1
                    logger.debug(
                        "Context %s: sim=%.3f → LLM (segment %s)", direction, sim, seg.segment_id
                    )
                    if self.plog:
                        self.plog.log_context_gate(
                            seg.segment_id, direction, sim, 'LLM (ambiguous)', len(context_sents)
                        )
                    needed = self._ask_context_needed(raw_text, context_sents, direction)
                    if needed:
                        self._add_utterances_incremental(
                            seg, context_sents, direction, raw_text
                        )

            seg.word_count = len(seg.text.split())

        logger.info(
            "Context expansion: %d attached, %d skipped, %d LLM-decided",
            stats['attached'], stats['skipped'], stats['llm'],
        )
        return segments

    def _walk_context(
        self,
        original_sentences: List[Dict],
        anchor_idx: int,
        word_budget: int,
        duration_budget: float,
        ref_time_ms: int,
        direction: str,
        current_participant_id: Optional[str] = None,
    ) -> List[Dict]:
        """Walk adjacent sentences from anchor, respecting guardrails.

        Stops early if a sentence belongs to a different participant so that
        context expansion never pulls in another participant's personal turn.
        Only therapist / facilitator sentences and the current participant's
        own earlier utterances are eligible for context.
        """
        collected = []
        words = 0

        if direction == 'backward':
            indices = range(anchor_idx - 1, -1, -1)
        else:
            indices = range(anchor_idx + 1, len(original_sentences))

        for oi in indices:
            osent = original_sentences[oi]
            osent_words = len(osent.get('text', '').split())

            # Stop at other-participant boundaries
            if current_participant_id and self.speaker_normalizer:
                osent_role = self.speaker_normalizer.get_role(osent.get('speaker', ''))
                if osent_role == 'participant':
                    osent_pid = self.speaker_normalizer.get_normalized_id(
                        osent.get('speaker', '')
                    )
                    if osent_pid != current_participant_id:
                        break

            # Gap check
            if direction == 'backward':
                neighbor = original_sentences[oi + 1]
                gap = neighbor.get('start', 0) - osent.get('end', 0)
            else:
                neighbor = original_sentences[oi - 1]
                gap = osent.get('start', 0) - neighbor.get('end', 0)
            if gap > self.max_gap_seconds:
                break

            if words + osent_words > word_budget:
                break

            # Duration check
            if direction == 'backward':
                elapsed = ref_time_ms / 1000.0 - osent.get('start', 0)
            else:
                elapsed = osent.get('end', 0) - ref_time_ms / 1000.0
            if elapsed > duration_budget:
                break

            words += osent_words
            if direction == 'backward':
                collected.insert(0, osent)
            else:
                collected.append(osent)

        return collected

    def _ask_context_needed(
        self,
        orig_participant_text: str,
        context_sents: List[Dict],
        direction: str,
    ) -> bool:
        """Ask the LLM (true/false) whether context in this direction is needed.

        Shows the full candidate context block with speaker labels so the
        model can assess relevance from dialogue structure, not just content.
        """
        dialogue = self._format_speaker_lines(context_sents)
        if direction == 'before':
            prompt = _CTX_BEFORE_NEEDED_PROMPT.format(
                participant_text=orig_participant_text,
                preceding_dialogue=dialogue,
            )
        else:
            prompt = _CTX_AFTER_NEEDED_PROMPT.format(
                participant_text=orig_participant_text,
                following_dialogue=dialogue,
            )

        response_text, _ = self.client.request(prompt)

        if self.plog:
            self.plog.log_llm_call(
                f"ctx_needed ({direction})", prompt, response_text or '(empty)'
            )

        if not response_text:
            return False
        try:
            parsed = extract_json(response_text)
            return bool(parsed.get('answer', False))
        except Exception:
            return False

    def _add_utterances_incremental(
        self,
        seg: Segment,
        candidate_sents: List[Dict],
        direction: str,
        orig_participant_text: str,
    ) -> None:
        """Add utterances one at a time, closest to segment first, until LLM says stop.

        For 'before': iterates from the sentence immediately before the segment
        outward (i.e., reverse of the chronological candidate list).
        For 'after': iterates from the sentence immediately after outward.

        Shows the growing segment with speaker labels on each call so the model
        has full dialogue context when deciding each incremental addition.
        """
        # candidate_sents is chronological; closest-to-segment depends on direction
        ordered = list(reversed(candidate_sents)) if direction == 'before' else list(candidate_sents)
        attached: List[Dict] = []

        for sent in ordered:
            speaker = self._get_anonymized_speaker(sent.get('speaker', '?'))
            utterance_line = f"[{speaker}]: {sent.get('text', '')}"
            prompt = _CTX_ADD_UTTERANCE_PROMPT.format(
                current_segment=seg.text,
                utterance_line=utterance_line,
            )

            response_text, _ = self.client.request(prompt)

            if self.plog:
                self.plog.log_llm_call(
                    f"ctx_add ({direction})", prompt, response_text or '(empty)'
                )

            if not response_text:
                break
            try:
                parsed = extract_json(response_text)
                should_add = bool(parsed.get('answer', False))
            except Exception:
                break

            if not should_add:
                break

            attached.append(sent)
            # Update seg.text immediately so the next prompt sees the growing segment
            # (attach single sentence as a block to maintain speaker formatting)
            self._attach_context(seg, [sent], direction)

        if attached:
            logger.debug(
                "Incremental context %s: added %d/%d utterances (segment %s)",
                direction, len(attached), len(candidate_sents), seg.segment_id,
            )
        else:
            logger.debug(
                "Incremental context %s: no utterances added (segment %s)",
                direction, seg.segment_id,
            )

    def _get_anonymized_speaker(self, raw_name: str) -> str:
        """Get anonymized speaker ID, falling back to raw name."""
        if self.speaker_normalizer:
            return self.speaker_normalizer.get_normalized_id(raw_name)
        return raw_name

    def _format_speaker_lines(self, sentences: List[Dict]) -> str:
        """Format sentences with anonymized speaker labels at speaker changes.

        Consecutive sentences from the same speaker are joined on the
        same line, so the label only appears once per speaker run.
        Raw speaker names are anonymized via the speaker normalizer.
        """
        if not sentences:
            return ''
        runs: List[Tuple[str, List[str]]] = []
        for s in sentences:
            speaker = self._get_anonymized_speaker(s.get('speaker', '?'))
            text = s.get('text', '')
            if runs and runs[-1][0] == speaker:
                runs[-1][1].append(text)
            else:
                runs.append((speaker, [text]))
        return "\n".join(
            f"[{speaker}]: {' '.join(texts)}" for speaker, texts in runs
        )

    def _attach_context(
        self,
        seg: Segment,
        context_sents: List[Dict],
        direction: str,
    ) -> None:
        """Attach context sentences to segment text.

        Uses anonymized speaker labels (via speaker normalizer),
        collapsed per speaker run. When context is prepended ('before'),
        the segment's own text gets a [participant_id]: prefix so the
        speaker transition is visible in the output.
        """
        context_block = self._format_speaker_lines(context_sents)

        if direction == 'before':
            seg_text = seg.text
            if not seg_text.startswith('['):
                seg_text = f"[{seg.participant_id}]: {seg_text}"
            seg.text = context_block + "\n" + seg_text
            seg.start_time_ms = int(context_sents[0].get('start', 0) * 1000)
        else:
            seg.text = seg.text + "\n" + context_block
            seg.end_time_ms = int(context_sents[-1].get('end', 0) * 1000)

    # ------------------------------------------------------------------
    # Coherence check (embedding-first, LLM fallback)
    # ------------------------------------------------------------------

    def _check_coherence(self, segments: List[Segment]) -> List[Segment]:
        """Check oversized segments for coherence using embeddings first.

        For segments >300 words:
        1. Compute intra-segment sentence similarity
        2. If clear dip found → split mechanically (no LLM call)
        3. If ambiguous dip → ask LLM where to split
        4. If no dip → keep as-is
        """
        result = []
        for seg in segments:
            if seg.word_count <= 300:
                result.append(seg)
                continue

            split_result = self._coherence_check_single(seg)
            if split_result:
                result.extend(split_result)
            else:
                result.append(seg)

        return result

    def _coherence_check_single(self, seg: Segment) -> Optional[List[Segment]]:
        """Check a single segment for coherence using embeddings, LLM fallback."""
        sent_pattern = re.compile(r'(?<=[.!?])\s+')
        text_sentences = sent_pattern.split(seg.text)
        if len(text_sentences) < 4:
            return None

        # Embedding-based detection using cached model
        try:
            model = self._get_embed_model()
            embs = model.encode(text_sentences, normalize_embeddings=True)

            sims = [float(np.dot(embs[i], embs[i + 1])) for i in range(len(embs) - 1)]
            if not sims:
                return None

            min_sim = min(sims)
            min_idx = sims.index(min_sim)

            # Clear dip → split mechanically
            if min_sim < self.coherence_sim_threshold:
                split_after = min_idx + 1
                return self._split_segment(seg, text_sentences, split_after)

            # No dip → keep as-is
            mean_sim = np.mean(sims)
            if min_sim > mean_sim - 0.1:
                return None

        except ImportError:
            pass

        # Ambiguous or import failed → LLM fallback
        return self._coherence_check_llm(seg, text_sentences)

    def _coherence_check_llm(
        self, seg: Segment, text_sentences: List[str],
    ) -> Optional[List[Segment]]:
        """Ask LLM whether a segment should be split."""
        numbered = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(text_sentences))

        prompt = _COHERENCE_CHECK_PROMPT.format(
            numbered_sentences=numbered,
        )

        response_text, _ = self.client.request(prompt)

        if self.plog:
            self.plog.log_llm_call(
                f"coherence_check ({seg.segment_id})",
                prompt,
                response_text or '(empty)',
            )

        if not response_text:
            return None

        try:
            parsed = extract_json(response_text)
            if isinstance(parsed, dict) and parsed.get('should_split'):
                split_after = parsed.get('split_after_sentence')
                if split_after and 0 < split_after < len(text_sentences):
                    if self.plog:
                        self.plog._write(f"  → split after sentence {split_after}")
                    return self._split_segment(seg, text_sentences, split_after)
        except Exception:
            pass

        return None

    def _split_segment(
        self, seg: Segment, text_sentences: List[str], split_after: int,
    ) -> List[Segment]:
        """Split a segment at a sentence boundary.

        Each half retains its speaker label. If the second half does not start
        with a [speaker]: label, the last label found in the first half is
        carried forward so the speaker context is never lost.
        """
        text_a = ' '.join(text_sentences[:split_after])
        text_b = ' '.join(text_sentences[split_after:])

        if not text_a.strip() or not text_b.strip():
            return [seg]

        # Ensure text_b starts with a speaker label
        if not text_b.startswith('['):
            # Find the most recent [speaker]: label in text_a
            label_match = re.findall(r'\[([^\]]+)\]:', text_a)
            if label_match:
                last_speaker = label_match[-1]
                text_b = f"[{last_speaker}]: {text_b}"
            else:
                text_b = f"[{seg.participant_id}]: {text_b}"

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
