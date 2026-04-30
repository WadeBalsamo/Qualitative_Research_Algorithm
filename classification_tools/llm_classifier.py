"""
llm_classifier.py
-----------------
LLM classification of transcript segments (theme and codebook).

Contains both:
- Theme classification: single-label zero-shot using any ThemeFramework
- Codebook classification: multi-label LLM-based using any Codebook

"""

import json
import os
import datetime
from typing import List, Dict, Any, Optional, Tuple

from .data_structures import Segment
from .llm_client import LLMClient, LLMClientConfig, extract_json
from .classification_loop import filter_participant_segments, classify_segments
from .majority_vote import vote_single_label, vote_multi_label
from theme_framework.theme_schema import ThemeFramework
from theme_framework.config import ThemeClassificationConfig
from codebook.codebook_schema import Codebook, CodeAssignment
from codebook.config import LLMCodebookConfig


# ---------------------------------------------------------------------------
# Prompt template (parameterized by framework)
# ---------------------------------------------------------------------------

THEME_PROMPT_TEMPLATE = """You are a qualitative researcher trained in the \
{framework_name} framework for analyzing therapeutic dialogue.

{framework_description}

{codebook_string}

{context_block}Classify the following participant utterance according to these {num_themes} \
themes. The utterance starts and ends with ```.

Utterance:
```
{text}
```

Provide your classification as JSON with these exact fields:
{{
    "primary_stage": "<theme name or null>",
    "primary_confidence": <float 0-1>,
    "secondary_stage": "<theme name or null>",
    "secondary_confidence": <float 0-1 or null>,
    "justification": "<brief explanation referencing specific language>"
}}

Rules:
- Assign exactly one primary theme, or null if the utterance is irrelevant to the study
- Assign a secondary theme ONLY if the utterance clearly expresses two themes
- Confidence should reflect how prototypical the expression is (1.0 = textbook example)
- Reference specific words or phrases in your justification
- Do NOT provide any text outside the JSON

JSON:"""

CONTEXT_PREAMBLE = """For context, here is the preceding conversational exchange \
(do NOT classify this — it is background only):
---
{preceding_context}
---

"""


# ---------------------------------------------------------------------------
# Context window helper
# ---------------------------------------------------------------------------

MAX_CONTEXT_WORDS = 300


def _build_context_block(
    all_segments: List[Segment],
    current_index: int,
    window_size: int,
) -> str:
    """
    Build a context preamble from preceding segments.

    Returns an empty string if window_size is 0 or no preceding segments exist.
    """
    if window_size <= 0 or current_index <= 0:
        return ''

    context_parts = []
    total_words = 0
    start = max(0, current_index - window_size)

    for seg in all_segments[start:current_index]:
        seg_text = seg.text.strip()
        seg_words = len(seg_text.split())
        if total_words + seg_words > MAX_CONTEXT_WORDS:
            # Truncate this segment to fit budget
            remaining = MAX_CONTEXT_WORDS - total_words
            if remaining > 0:
                words = seg_text.split()
                context_parts.append(' '.join(words[-remaining:]) + ' [...]')
            break
        context_parts.append(seg_text)
        total_words += seg_words

    if not context_parts:
        return ''

    preceding_text = '\n\n'.join(context_parts)
    return CONTEXT_PREAMBLE.format(preceding_context=preceding_text)


# ---------------------------------------------------------------------------
# Short segment merging
# ---------------------------------------------------------------------------


def _merge_short_segments(
    segments: List[Segment],
    min_words: int,
) -> List[Segment]:
    """
    Merge segments shorter than *min_words* into their neighbors.

    Short segments are combined with the preceding segment when possible,
    otherwise the following segment.  The merged segment inherits the
    earlier segment's ID and start time, and the later segment's end time.
    Text is joined with a newline separator.
    """
    if not segments or min_words <= 0:
        return segments

    merged: List[Segment] = []
    for seg in segments:
        if seg.word_count >= min_words:
            merged.append(seg)
        elif merged:
            # Merge into preceding segment
            prev = merged[-1]
            prev.text = prev.text + "\n" + seg.text
            prev.word_count = len(prev.text.split())
            prev.end_time_ms = seg.end_time_ms
            if prev.speakers_in_segment and seg.speakers_in_segment:
                seen = set(prev.speakers_in_segment)
                for sp in seg.speakers_in_segment:
                    if sp not in seen:
                        prev.speakers_in_segment.append(sp)
                        seen.add(sp)
        else:
            # No preceding segment yet — hold it and merge forward
            merged.append(seg)

    # Second pass: merge any leading short segment into its follower
    if len(merged) > 1 and merged[0].word_count < min_words:
        first = merged.pop(0)
        merged[0].text = first.text + "\n" + merged[0].text
        merged[0].word_count = len(merged[0].text.split())
        merged[0].start_time_ms = first.start_time_ms
        merged[0].segment_id = first.segment_id

    # Re-index
    for i, seg in enumerate(merged):
        seg.segment_index = i

    return merged


# ---------------------------------------------------------------------------
# Content validity test set construction
# ---------------------------------------------------------------------------

def create_content_validity_test_set(
    framework: ThemeFramework,
) -> List[Dict]:
    """
    Construct a content validity test set from a ThemeFramework.

    Extracts exemplar, subtle, and adversarial utterances at three
    prototypicality tiers for evaluating classifier coverage.
    """
    test_items = []
    item_id = 0

    tier_fields = [
        ('clear', 'exemplar_utterances'),
        ('subtle', 'subtle_utterances'),
        ('adversarial', 'adversarial_utterances'),
    ]

    for theme in framework.themes:
        for difficulty, field_name in tier_fields:
            for utterance in getattr(theme, field_name, []):
                test_items.append({
                    'test_item_id': f'cv_{item_id:04d}',
                    'text': utterance,
                    'expected_stage': theme.theme_id,
                    'difficulty': difficulty,
                    'source': 'codebook',
                })
                item_id += 1

    return test_items


# ---------------------------------------------------------------------------
# Single-run parsing
# ---------------------------------------------------------------------------

def _parse_single_run(
    result: Any,
    name_to_id: Dict[str, int],
) -> Optional[Dict]:
    """Parse a single LLM response into a ballot.

    Returns one of:
    - ``None`` — hard parse failure (no JSON, malformed JSON, or a
      ``primary_stage`` that names something not in the framework).
      These ballots are excluded from the denominator of the vote.
    - ``{'vote': 'ABSTAIN', ...}`` — the rater judged the utterance
      irrelevant to the framework (JSON ``primary_stage``: null).
      ABSTAIN is a *real* ballot and is counted by ``vote_single_label``.
    - ``{'vote': 'CODED', 'primary_stage': <int>, ...}`` — a concrete
      theme ID assignment.
    """
    if result is None:
        return None

    try:
        if isinstance(result, str):
            parsed = extract_json(result)
        elif isinstance(result, dict):
            parsed = result
        else:
            return None

        if 'primary_stage' not in parsed:
            return None

        primary_stage_raw = parsed['primary_stage']

        def _coerce_conf(raw) -> float:
            try:
                return float(raw) if raw is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        confidence = _coerce_conf(parsed.get('primary_confidence'))

        secondary_raw = parsed.get('secondary_stage')
        secondary_id: Optional[int] = None
        if secondary_raw is not None:
            secondary_name = str(secondary_raw).lower().strip()
            if secondary_name and secondary_name not in ('null', 'none', ''):
                secondary_id = name_to_id.get(secondary_name)  # may be None (invalid)

        secondary_conf_raw = parsed.get('secondary_confidence')
        if secondary_conf_raw is not None and str(secondary_conf_raw).lower() not in ('null', 'none'):
            try:
                secondary_conf: Optional[float] = float(secondary_conf_raw)
            except (TypeError, ValueError):
                secondary_conf = None
        else:
            secondary_conf = None

        justification = parsed.get('justification', '') or ''
        evidence = parsed.get('evidence_phrase', '') or ''

        # ABSTAIN: explicit null (or the string "null"/"none")
        if primary_stage_raw is None:
            return {
                'vote': 'ABSTAIN',
                'primary_stage': None,
                'primary_confidence': confidence,
                'secondary_stage': None,
                'secondary_confidence': None,
                'justification': justification,
                'evidence_phrase': evidence,
            }

        primary_name = str(primary_stage_raw).lower().strip()
        if not primary_name or primary_name in ('null', 'none'):
            return {
                'vote': 'ABSTAIN',
                'primary_stage': None,
                'primary_confidence': confidence,
                'secondary_stage': None,
                'secondary_confidence': None,
                'justification': justification,
                'evidence_phrase': evidence,
            }

        primary_id = name_to_id.get(primary_name)
        if primary_id is None:
            # Named a theme we don't recognize — treat as parse failure
            # so it doesn't corrupt the vote.
            return None

        return {
            'vote': 'CODED',
            'primary_stage': primary_id,
            'primary_confidence': confidence,
            'secondary_stage': secondary_id,
            'secondary_confidence': secondary_conf,
            'justification': justification,
            'evidence_phrase': evidence,
        }
    except Exception as e:
        raw_snippet = str(result)[:120] if result else '<empty>'
        print(f"  Parse error: {e} | Response: {raw_snippet}")
        return None


# ---------------------------------------------------------------------------
# Classification loop
# ---------------------------------------------------------------------------

def classify_segments_zero_shot(
    segments: List[Segment],
    framework: ThemeFramework,
    config: ThemeClassificationConfig,
    resume_from: Optional[str] = None,
    process_logger=None,
) -> Tuple[Dict, Dict]:
    """
    Zero-shot classification of transcript segments using a ThemeFramework.

    Interrater reliability design
    -----------------------------
    - If ``config.per_run_models`` is a list of length ``n_runs`` (≥ 2),
      each run uses a *distinct* model: three independent raters, one
      classification pass each. This is the canonical IRR mode.
    - Otherwise if ``config.models`` has ≥ 2 entries, they are used as
      ``per_run_models`` automatically (with ``n_runs`` set to match).
    - Otherwise classification falls back to a single model with
      ``n_runs`` stochastic passes (temperature jitter + randomized
      theme order). ``temperature`` must be > 0 in this case or the
      pipeline raises immediately — identical calls do not give IRR.

    Returns
    -------
    (results_all, metadata_all)
        ``results_all`` maps segment_id -> {'consensus': <vote dict>,
        'rater_votes': [...], 'rater_ids': [...]} (see majority_vote).
    """
    name_to_id = framework.build_name_to_id_map()

    # ------------------------------------------------------------------
    # Resolve rater roster. One of three modes:
    #   per_run_models present  -> use it directly
    #   models has 2+ entries   -> promote to per_run_models; set n_runs
    #   else                    -> single model, n_runs temperature-jitter
    # ------------------------------------------------------------------
    per_run_models = list(getattr(config, 'per_run_models', None) or [])
    multi_models = list(getattr(config, 'models', None) or [])

    if not per_run_models and len(multi_models) >= 2:
        per_run_models = list(multi_models)
        config.n_runs = len(per_run_models)
        print(f"  Promoting {len(per_run_models)} models to per-run raters "
              f"(n_runs={config.n_runs})")

    use_per_run_models = (
        len(per_run_models) == config.n_runs and len(per_run_models) >= 2
    )

    if not use_per_run_models and config.n_runs > 1:
        raise ValueError(
            "Multi-run IRR requires distinct models in per_run_models. "
            "Single-model stochastic IRR at temperature>0 has been removed; "
            "use n_runs=1 or configure per_run_models with >=2 distinct models."
        )

    llm_config = LLMClientConfig(
        backend=config.backend,
        api_key=config.api_key,
        replicate_api_token=config.replicate_api_token,
        model=config.model,
        models=multi_models or [config.model],
        temperature=config.temperature,
        max_new_tokens=config.max_new_tokens,
        ollama_host=getattr(config, 'ollama_host', '0.0.0.0'),
        ollama_port=getattr(config, 'ollama_port', 11434),
        lmstudio_base_url=getattr(config, 'lmstudio_base_url', 'http://127.0.0.1:1234/v1'),
        process_logger=process_logger,
    )
    client = LLMClient(llm_config)

    if use_per_run_models:
        rater_ids = list(per_run_models)
        print(f"  Per-run model assignment ({config.n_runs} distinct raters)")
        for i, model_id in enumerate(per_run_models):
            print(f"    Run {i + 1}: {model_id}")
    else:
        rater_ids = [f'{config.model}#run_{i + 1}' for i in range(config.n_runs)]
        print(f"  Single-model IRR: {config.n_runs} stochastic runs of "
              f"{config.model} (temperature={config.temperature})")

    os.makedirs(config.output_dir, exist_ok=True)
    model_clean = config.model.replace('/', '_')

    target_segments = filter_participant_segments(segments)
    prompt_template = THEME_PROMPT_TEMPLATE

    # Merge trivially short segments into their neighbors before classification
    MIN_CLASSIFIABLE_WORDS = getattr(config, 'min_classifiable_words', 10)
    before_count = len(target_segments)
    target_segments = _merge_short_segments(target_segments, MIN_CLASSIFIABLE_WORDS)
    merged_count = before_count - len(target_segments)
    if merged_count > 0:
        print(f"  Merged {merged_count} short segments (<{MIN_CLASSIFIABLE_WORDS} words) "
              f"into neighbors: {before_count} -> {len(target_segments)} segments")

    context_window = getattr(config, 'context_window_segments', 2)

    def build_prompt(segment: Segment, run: int,
                     all_segments: List[Segment] = None,
                     seg_index: int = 0) -> str:
        codebook_string = framework.to_prompt_string(
            randomize=config.randomize_codebook,
            zero_shot=getattr(config, 'zero_shot_prompt', False),
            n_exemplars=getattr(config, 'prompt_n_exemplars', None),
            include_subtle=getattr(config, 'prompt_include_subtle', True),
            n_subtle=getattr(config, 'prompt_n_subtle', None),
            include_adversarial=getattr(config, 'prompt_include_adversarial', True),
            n_adversarial=getattr(config, 'prompt_n_adversarial', None),
        )
        context_block = ''
        if all_segments and context_window > 0:
            context_block = _build_context_block(all_segments, seg_index, context_window)
        return prompt_template.format(
            framework_name=framework.name,
            framework_description=framework.description,
            codebook_string=codebook_string,
            num_themes=framework.num_themes,
            text=segment.text,
            context_block=context_block,
        )

    def parse_response(result_text: str) -> Optional[Dict]:
        return _parse_single_run(result_text, name_to_id)

    def merge_runs(parsed_runs: List[Optional[Dict]]) -> Dict:
        # classify_segments pads/truncates parsed_runs to n_runs. Each
        # slot lines up with rater_ids by index.
        padded = list(parsed_runs) + [None] * (config.n_runs - len(parsed_runs))
        padded = padded[:config.n_runs]
        consensus = vote_single_label(
            padded,
            rater_ids=rater_ids,
            secondary_weight=getattr(config, 'evidence_secondary_weight', 0.6),
            presence_threshold=getattr(config, 'evidence_presence_threshold', 0.5),
        )
        return {
            'rater_ids': rater_ids,
            'rater_votes': consensus['rater_votes'],
            'consensus': consensus,
        }

    raw_results = classify_segments(
        segments=target_segments,
        client=client,
        n_runs=config.n_runs,
        build_prompt=build_prompt,
        parse_response=parse_response,
        merge_runs=merge_runs,
        output_dir=config.output_dir,
        save_interval=config.save_interval,
        resume_from=resume_from,
        file_prefix='llm_results',
        model_tag=model_clean,
        per_run_models=per_run_models if use_per_run_models else None,
    )

    seg_by_id = {s.segment_id: s for s in target_segments}
    metadata_all: Dict[str, Any] = {}
    for seg_id in raw_results:
        seg = seg_by_id.get(seg_id)
        metadata_all[seg_id] = {
            'segment_id': seg_id,
            'text': seg.text if seg else '',
        }

    return raw_results, metadata_all


# ===========================================================================
# Multi-label codebook classification via LLM (merged from llm_classifier.py)
# ===========================================================================

CODEBOOK_PROMPT_TEMPLATE = """You are a qualitative researcher applying a \
codebook to therapeutic dialogue transcripts.

The codebook contains {n_codes} codes organized into {n_domains} domains:

{codebook_string}

For the following participant utterance, identify ALL codes that apply. \
A segment may have zero, one, or multiple codes.

Utterance:
```
{text}
```

Provide your classification as JSON with these exact fields:
{{
    "applied_codes": [
        {{
            "code": "<code category name>",
            "confidence": <float 0-1>,
            "justification": "<brief explanation>"
        }}
    ],
    "no_codes_apply": <boolean>
}}

Rules:
- Apply ALL codes whose inclusion criteria are met by this utterance
- Do NOT apply codes whose exclusion criteria are met
- Confidence should reflect how clearly the inclusion criteria are met (1.0 = textbook match)
- If no codes apply, set no_codes_apply to true and leave applied_codes empty
- Do NOT provide any text outside the JSON

JSON:"""


class LLMCodebookClassifier:
    """Multi-label codebook classification via LLM zero-shot prompting."""

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[LLMCodebookConfig] = None,
    ):
        self.client = llm_client
        self.config = config or LLMCodebookConfig()

    def classify_segments(
        self,
        segments: List[Segment],
        codebook: Codebook,
        output_dir: Optional[str] = None,
    ) -> Dict[str, List[CodeAssignment]]:
        """
        Classify segments using LLM with multi-label codebook prompt.

        Returns dict mapping segment_id -> list of CodeAssignments.
        """
        valid_categories = {c.category.lower(): c for c in codebook.codes}

        participant_segments = filter_participant_segments(segments)

        def build_prompt(segment: Segment, run: int,
                         all_segments: List[Segment] = None,
                         seg_index: int = 0) -> str:
            return self._build_prompt(segment, codebook)

        def parse_response(response: str) -> Optional[List[CodeAssignment]]:
            return self._parse_response(response, valid_categories)

        def merge_runs(parsed_runs: List[List[CodeAssignment]]) -> List[CodeAssignment]:
            return self._merge_runs(parsed_runs)

        def serialize_result(assignments: List[CodeAssignment]):
            return [
                {
                    'code_id': a.code_id,
                    'category': a.category,
                    'confidence': a.confidence,
                    'justification': a.justification,
                }
                for a in assignments
            ]

        effective_output = output_dir or self.config.output_dir or None

        raw_results = classify_segments(
            segments=participant_segments,
            client=self.client,
            n_runs=self.config.n_runs,
            build_prompt=build_prompt,
            parse_response=parse_response,
            merge_runs=merge_runs,
            output_dir=effective_output,
            save_interval=self.config.save_interval,
            file_prefix='codebook_llm_results',
            serialize_result=serialize_result,
        )

        return raw_results

    def _build_prompt(self, segment: Segment, codebook: Codebook) -> str:
        """Build the multi-label classification prompt."""
        codebook_string = codebook.to_prompt_string(
            randomize=self.config.randomize_codebook
        )
        return CODEBOOK_PROMPT_TEMPLATE.format(
            n_codes=len(codebook.codes),
            n_domains=len(codebook.domain_names),
            codebook_string=codebook_string,
            text=segment.text,
        )

    def _parse_response(
        self,
        response: str,
        valid_categories: Dict[str, 'CodeDefinition'],
    ) -> List[CodeAssignment]:
        """Parse a single LLM response into code assignments."""
        try:
            parsed = extract_json(response)
        except (ValueError, json.JSONDecodeError):
            return []

        assignments = []
        for entry in parsed.get('applied_codes', []):
            code_name = str(entry.get('code', '')).strip()
            code_lower = code_name.lower()

            code_def = valid_categories.get(code_lower)
            if code_def is None:
                continue

            confidence = float(entry.get('confidence') or 0.5)
            if confidence < self.config.confidence_threshold:
                continue

            assignments.append(CodeAssignment(
                code_id=code_def.code_id,
                category=code_def.category,
                confidence=confidence,
                justification=str(entry.get('justification', '')),
                method='llm',
            ))

        return assignments

    def _merge_runs(
        self, all_assignments: List[List[CodeAssignment]]
    ) -> List[CodeAssignment]:
        """Merge assignments across multiple runs via strict-majority voting."""
        if not all_assignments or not any(a is not None for a in all_assignments):
            return []

        voted = vote_multi_label(all_assignments)

        return [
            CodeAssignment(
                code_id=exemplar.code_id,
                category=exemplar.category,
                confidence=round(avg_conf, 4),
                justification=exemplar.justification,
                method='llm',
            )
            for exemplar, avg_conf in voted['assignments']
        ]
