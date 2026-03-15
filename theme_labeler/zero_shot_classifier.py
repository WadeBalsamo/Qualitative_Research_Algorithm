"""
zero_shot_classifier.py
-----------------------
Zero-shot LLM classification of transcript segments using any
ThemeFramework, with triplicate-and-flag consistency checking.

- Uses ThemeFramework.to_prompt_string() instead of hardcoded VA-MR codebook
- Uses shared.LLMClient instead of inline API calls
- Prompt template is parameterized by framework name/description
- name_to_id lookup is built from the framework, not a global constant
"""

import json
import os
import datetime
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

from shared.data_structures import Segment
from shared.llm_client import LLMClient, LLMClientConfig, extract_json
from shared.classification_loop import filter_participant_segments, classify_segments
from shared.majority_vote import single_label_majority_vote
from .theme_schema import ThemeFramework
from .config import ThemeClassificationConfig


# ---------------------------------------------------------------------------
# Prompt template (parameterized by framework)
# ---------------------------------------------------------------------------

THEME_PROMPT_TEMPLATE = """You are a qualitative researcher trained in the \
{framework_name} framework for analyzing therapeutic dialogue.

{framework_description}

{codebook_string}

Classify the following participant utterance according to these {num_themes} \
themes. The utterance starts and ends with ```.

Utterance:
```
{text}
```

Provide your classification as JSON with these exact fields:
{{
    "primary_stage": "<theme name>",
    "primary_confidence": <float 0-1>,
    "secondary_stage": "<theme name or null>",
    "secondary_confidence": <float 0-1 or null>,
    "justification": "<brief explanation referencing specific language>"
}}

Rules:
- Assign exactly one primary theme
- Assign a secondary theme ONLY if the utterance clearly expresses two themes
- Confidence should reflect how prototypical the expression is (1.0 = textbook example)
- Reference specific words or phrases in your justification
- Do NOT provide any text outside the JSON

JSON:"""


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
    """Parse a single LLM response into structured fields."""
    if result is None:
        return None

    try:
        if isinstance(result, str):
            parsed = extract_json(result)
        elif isinstance(result, dict):
            parsed = result
        else:
            return None

        primary_name = str(parsed.get('primary_stage', '')).lower().strip()
        primary_id = name_to_id.get(primary_name)

        secondary_name = parsed.get('secondary_stage')
        secondary_id = None
        if secondary_name and str(secondary_name).lower().strip() not in ('null', 'none', ''):
            secondary_id = name_to_id.get(str(secondary_name).lower().strip())

        secondary_conf = parsed.get('secondary_confidence')
        if secondary_conf is not None and str(secondary_conf).lower() not in ('null', 'none'):
            secondary_conf = float(secondary_conf)
        else:
            secondary_conf = None

        return {
            'primary_stage': primary_id,
            'primary_confidence': float(parsed.get('primary_confidence', 0)),
            'secondary_stage': secondary_id,
            'secondary_confidence': secondary_conf,
            'justification': parsed.get('justification', ''),
        }
    except Exception as e:
        print(f"  Parse error: {e}")
        return None


# ---------------------------------------------------------------------------
# Multi-model cross-referencing
# ---------------------------------------------------------------------------

def _classify_multi_model(
    segment: Segment,
    framework: ThemeFramework,
    client: LLMClient,
    config: ThemeClassificationConfig,
    name_to_id: Dict[str, int],
) -> Dict:
    """
    Classify a segment using multiple models and cross-reference results.

    Returns a dictionary with per-model results and cross-model consensus.
    """
    from shared.model_loader import get_model_display_name

    models = client.config.models
    model_results = {}

    # Classify with each model
    for model_id in models:
        model_runs = []
        for run in range(config.n_runs):
            codebook_string = framework.to_prompt_string(
                randomize=config.randomize_codebook,
            )
            prompt = THEME_PROMPT_TEMPLATE.format(
                framework_name=framework.name,
                framework_description=framework.description,
                codebook_string=codebook_string,
                num_themes=framework.num_themes,
                text=segment.text,
            )
            try:
                # Override model for this request
                original_model = client.config.model
                client.config.model = model_id
                result_text, meta = client.request(prompt)
                client.config.model = original_model

                model_runs.append(result_text)
            except Exception as e:
                print(f"  Error on {segment.segment_id}, model {model_id}, run {run}: {e}")
                model_runs.append(None)

        # Parse and compute consistency for this model
        parsed_runs = [
            _parse_single_run(r, name_to_id)
            for r in model_runs if r is not None
        ]
        consistency_result = single_label_majority_vote(parsed_runs)

        model_results[model_id] = {
            'runs': model_runs,
            'parsed_runs': [r for r in parsed_runs if r is not None],
            'consistency': consistency_result,
        }

    # Cross-reference models
    cross_model_consensus = _compute_cross_model_consensus(model_results)

    return {
        'model_results': model_results,
        'cross_model_consensus': cross_model_consensus,
        'runs': [],  # For backward compatibility
        'parsed_runs': [],
        'consistency': cross_model_consensus,  # Use consensus as final result
    }


def _compute_cross_model_consensus(model_results: Dict[str, Dict]) -> Dict:
    """
    Compute consensus across multiple models.

    Flags segments where models disagree.
    """
    from shared.model_loader import get_model_display_name

    # Extract primary stage predictions from each model
    model_predictions = {}
    for model_id, result in model_results.items():
        consistency = result.get('consistency', {})
        primary_stage = consistency.get('primary_stage')
        confidence = consistency.get('confidence', 0.0)
        model_predictions[model_id] = {
            'primary_stage': primary_stage,
            'confidence': confidence,
            'display_name': get_model_display_name(model_id),
        }

    # Count agreements
    valid_predictions = [
        p for p in model_predictions.values()
        if p['primary_stage'] is not None
    ]

    if not valid_predictions:
        return {
            'primary_stage': None,
            'consistency': 0,
            'confidence': 0.0,
            'secondary_stage': None,
            'secondary_confidence': None,
            'justification': 'All models failed to classify',
            'model_agreement': 'none',
            'model_predictions': model_predictions,
        }

    # Find most common prediction
    stage_counts = Counter(p['primary_stage'] for p in valid_predictions)
    majority_stage, majority_count = stage_counts.most_common(1)[0]

    # Compute agreement level
    total_models = len(valid_predictions)
    if majority_count == total_models:
        agreement = 'unanimous'
    elif majority_count >= (total_models / 2 + 1):
        agreement = 'majority'
    else:
        agreement = 'split'

    # Average confidence among agreeing models
    agreeing_models = [
        p for p in valid_predictions
        if p['primary_stage'] == majority_stage
    ]
    avg_confidence = sum(p['confidence'] for p in agreeing_models) / len(agreeing_models)

    # Generate justification
    model_names = ', '.join([p['display_name'] for p in agreeing_models])
    if agreement == 'unanimous':
        justification = f"All models agree ({model_names})"
    elif agreement == 'majority':
        justification = f"Majority agree: {model_names} ({majority_count}/{total_models})"
    else:
        justification = f"Models disagree: {majority_count}/{total_models} for this label"

    return {
        'primary_stage': majority_stage,
        'consistency': majority_count,  # Number of models agreeing
        'confidence': avg_confidence,
        'secondary_stage': None,
        'secondary_confidence': None,
        'justification': justification,
        'model_agreement': agreement,
        'model_predictions': model_predictions,
        'total_models': total_models,
    }


# ---------------------------------------------------------------------------
# Classification loop
# ---------------------------------------------------------------------------

def classify_segments_zero_shot(
    segments: List[Segment],
    framework: ThemeFramework,
    config: ThemeClassificationConfig,
    resume_from: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """
    Zero-shot classification of transcript segments using a ThemeFramework.

    Supports multi-model cross-referencing when multiple models are configured.
    Each segment is classified by all models, and results are cross-referenced
    for consensus or disagreement flagging.
    """
    name_to_id = framework.build_name_to_id_map()

    # Support both single model and multi-model configuration
    models = getattr(config, 'models', None)
    if not models:
        models = [config.model] if config.model else []

    llm_config = LLMClientConfig(
        backend=config.backend,
        api_key=config.api_key,
        replicate_api_token=config.replicate_api_token,
        model=config.model,
        models=models,
        temperature=config.temperature,
        max_new_tokens=config.max_new_tokens,
        ollama_host=getattr(config, 'ollama_host', '0.0.0.0'),
        ollama_port=getattr(config, 'ollama_port', 11434),
    )
    client = LLMClient(llm_config)

    use_multi_model = len(models) > 1
    if use_multi_model:
        print(f"  Using multi-model cross-referencing with {len(models)} models")
        from shared.model_loader import get_model_display_name
        for model_id in models:
            print(f"    - {get_model_display_name(model_id)}")

    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')
    model_clean = config.model.replace('/', '_')

    if use_multi_model:
        # Multi-model path: manual loop (uses its own per-model N-run logic)
        results_all: Dict[str, Any] = {}
        metadata_all: Dict[str, Any] = {}
        if resume_from and os.path.exists(resume_from):
            with open(resume_from, 'r') as f:
                results_all = json.load(f)
            print(f"  Resumed from checkpoint: {len(results_all)} segments already classified")

        participant_segments = filter_participant_segments(segments)
        total = len(participant_segments)

        for i, segment in enumerate(participant_segments):
            if segment.segment_id in results_all:
                continue

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Classifying segment {i + 1}/{total}: {segment.segment_id}")

            segment_results = _classify_multi_model(
                segment, framework, client, config, name_to_id
            )

            results_all[segment.segment_id] = segment_results
            metadata_all[segment.segment_id] = {
                'segment_id': segment.segment_id,
                'text': segment.text,
                'runs_raw': segment_results.get('runs', []),
            }

            if i % config.save_interval == 0:
                _save_intermediate(results_all, metadata_all, config.output_dir, model_clean, timestamp)

        _save_intermediate(results_all, metadata_all, config.output_dir, model_clean, timestamp)
        return results_all, metadata_all

    # ----- Single-model path: delegate to shared loop -----
    participant_segments = filter_participant_segments(segments)

    def build_prompt(segment: Segment, run: int) -> str:
        codebook_string = framework.to_prompt_string(
            randomize=config.randomize_codebook,
        )
        return THEME_PROMPT_TEMPLATE.format(
            framework_name=framework.name,
            framework_description=framework.description,
            codebook_string=codebook_string,
            num_themes=framework.num_themes,
            text=segment.text,
        )

    def parse_response(result_text: str) -> Optional[Dict]:
        return _parse_single_run(result_text, name_to_id)

    def merge_runs(parsed_runs: List[Optional[Dict]]) -> Dict:
        consistency_result = single_label_majority_vote(parsed_runs)
        return {
            'parsed_runs': [r for r in parsed_runs if r is not None],
            'consistency': consistency_result,
        }

    raw_results = classify_segments(
        segments=participant_segments,
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
    )

    # Build metadata_all for backward compatibility
    seg_by_id = {s.segment_id: s for s in participant_segments}
    metadata_all: Dict[str, Any] = {}
    for seg_id, result in raw_results.items():
        seg = seg_by_id.get(seg_id)
        metadata_all[seg_id] = {
            'segment_id': seg_id,
            'text': seg.text if seg else '',
            'runs_raw': [],
        }

    return raw_results, metadata_all


def _save_intermediate(
    results: Dict, metadata: Dict,
    output_dir: str, model_clean: str, timestamp: str,
):
    """Save intermediate results to JSON files (multi-model path only)."""
    for name, data in [('results', results), ('metadata', metadata)]:
        path = os.path.join(
            output_dir, f'llm_{name}_{model_clean}_{timestamp}.json'
        )
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
