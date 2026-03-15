"""
zero_shot_classifier.py
-----------------------
Stage 3: LLM classification of transcript segments using the
VA-MR framework with triplicate-and-flag consistency checking.

Uses openrouter_request() and process_api_output() as the
API client layer.
"""

import json
import os
import datetime
import random
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

from .data_structures import Segment
from .vamr_constructs import stage_definitions, STAGE_NAME_TO_ID


# ---------------------------------------------------------------------------
# Codebook string construction
# ---------------------------------------------------------------------------

def create_vamr_codebook_string(
    definitions: Optional[Dict] = None,
    randomize: bool = False,
) -> str:
    """
    Build the codebook text block that gets inserted into the LLM prompt.

    Adapted from create_codebook_string() in classify_reddit_llms.py, which
    iterates over codebook_dict and formats each construct as:
        prompt_name.capitalize(): definition. Examples: examples

    Extended to include distinguishing features and exemplar utterances,
    which are critical for differentiating adjacent VA-MR stages.
    """
    if definitions is None:
        definitions = stage_definitions

    stages = list(definitions.keys())
    if randomize:
        random.shuffle(stages)

    codebook_string = ""
    for stage_key in stages:
        stage = definitions[stage_key]
        prompt_name = stage['prompt_name']
        definition = stage['definition']
        features = '; '.join(stage['prototypical_features'])
        exemplars = ' | '.join(stage['exemplar_utterances'][:3])
        distinguishing = stage['distinguishing_from_next']

        construct_str = (
            f"{prompt_name.capitalize()}: {definition} "
            f"Prototypical features: {features}. "
            f"Key distinction: {distinguishing}. "
            f"Examples: {exemplars}"
        ).replace('\n', ' ').replace('  ', ' ')

        codebook_string += construct_str + '\n\n'

    return codebook_string


# ---------------------------------------------------------------------------
# Prompt template
# Adapted from classify_reddit_llms.py's prompt_template
# ---------------------------------------------------------------------------

VAMR_PROMPT_TEMPLATE = """You are a qualitative researcher trained in the \
Vigilance-Avoidance Metacognition-Reappraisal (VA-MR) framework for analyzing \
therapeutic dialogue from Mindfulness-Oriented Recovery Enhancement sessions.

The VA-MR framework describes four stages of contemplative transformation \
that participants express in their language during therapy:

{codebook_string}

Classify the following participant utterance from a MORE therapy session \
according to these four stages. The utterance starts and ends with ```.

Utterance:
```
{text}
```

Provide your classification as JSON with these exact fields:
{{
    "primary_stage": "<stage name>",
    "primary_confidence": <float 0-1>,
    "secondary_stage": "<stage name or null>",
    "secondary_confidence": <float 0-1 or null>,
    "justification": "<brief explanation referencing specific language>"
}}

Rules:
- Assign exactly one primary stage
- Assign a secondary stage ONLY if the utterance clearly expresses two stages
- Confidence should reflect how prototypical the expression is (1.0 = textbook example)
- Reference specific words or phrases in your justification
- Do NOT provide any text outside the JSON

JSON:"""


# ---------------------------------------------------------------------------
# Content validity test set construction
# ---------------------------------------------------------------------------

def create_content_validity_test_set(
    definitions: Optional[Dict] = None,
) -> List[Dict]:
    """
    Construct the content validity test set for evaluating whether
    the classifier captures the full range of each stage's expressions.

    Adapted from ctl_feature_extraction.py's content validity creation which
    builds test sets at two prototypicality tiers by extracting tokens from
    the validated lexicon. We use full utterances instead of single tokens
    since VA-MR stages require sentential context to distinguish.
    """
    if definitions is None:
        definitions = stage_definitions

    test_items = []
    item_id = 0

    tier_fields = [
        ('clear', 'exemplar_utterances'),
        ('subtle', 'subtle_utterances'),
        ('adversarial', 'adversarial_utterances'),
    ]

    for stage_key, stage in definitions.items():
        stage_id = stage['stage_id']

        for difficulty, field_name in tier_fields:
            for utterance in stage.get(field_name, []):
                test_items.append({
                    'test_item_id': f'cv_{item_id:04d}',
                    'text': utterance,
                    'expected_stage': stage_id,
                    'difficulty': difficulty,
                    'source': 'codebook',
                })
                item_id += 1

    return test_items


# ---------------------------------------------------------------------------
# LLM API client (adapted from llm.py)
# ---------------------------------------------------------------------------

def _openrouter_request(
    prompt: str,
    api_key: str,
    model: str = 'openai/gpt-4o',
    temperature: float = 0.0,
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Send a prompt to OpenRouter API and return the response text + metadata.

    Adapted from openrouter_request() in llm.py. We import requests here
    rather than depending on llm.py directly so this module is self-contained.
    """
    import requests

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )

    try:
        metadata = response.json()
        result_text = metadata['choices'][0]['message']['content']
        return result_text, metadata
    except Exception as e:
        print(f"API error: {e}")
        return None, response.text if hasattr(response, 'text') else None


def _replicate_request(
    prompt: str,
    api_token: str,
    model: str = 'google-deepmind/gemma-2b',
    temperature: float = 0.1,
    max_new_tokens: int = 512,
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Send a prompt to Replicate API and return the response text.

    Adapted from replicate_request() in llm.py, which supports open-source
    models (Gemma, Mistral, etc.) for local/private inference.
    """
    try:
        import replicate
        client = replicate.Client(api_token=api_token)

        output = client.run(
            model,
            input={
                "prompt": prompt,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            },
        )
        result_text = ''.join(output)
        return result_text, None
    except Exception as e:
        print(f"Replicate API error: {e}")
        return None, None


def _process_api_output(output_str: str) -> Dict:
    """
    Extract JSON from LLM output, handling extra text.

    Direct adaptation of process_api_output() from llm.py:
        start_index = output_str.find('{')
        end_index = output_str.rfind('}') + 1
        json_part = output_str[start_index:end_index]
        data = json.loads(json_part)
    """
    try:
        return json.loads(output_str)
    except json.JSONDecodeError:
        start = output_str.find('{')
        end = output_str.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(output_str[start:end])
        raise ValueError("Could not extract JSON from LLM output.")


# ---------------------------------------------------------------------------
# Single-run parsing
# ---------------------------------------------------------------------------

def _parse_single_run(result: Any) -> Optional[Dict]:
    """Parse a single LLM response into structured fields."""
    if result is None:
        return None

    try:
        if isinstance(result, str):
            parsed = _process_api_output(result)
        elif isinstance(result, dict):
            parsed = result
        else:
            return None

        primary_name = str(parsed.get('primary_stage', '')).lower().strip()
        primary_id = STAGE_NAME_TO_ID.get(primary_name)

        secondary_name = parsed.get('secondary_stage')
        secondary_id = None
        if secondary_name and str(secondary_name).lower().strip() not in ('null', 'none', ''):
            secondary_id = STAGE_NAME_TO_ID.get(str(secondary_name).lower().strip())

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
        print(f"Parse error: {e}")
        return None


# ---------------------------------------------------------------------------
# Run consistency (triplicate-and-flag)
# ---------------------------------------------------------------------------

def _compute_run_consistency(parsed_runs: List[Optional[Dict]]) -> Dict:
    """
    Determine how many of the n runs agree on the primary stage.
    Segments with consistency < n are flagged for human review.
    """
    valid_runs = [
        r for r in parsed_runs
        if r is not None and r['primary_stage'] is not None
    ]

    if not valid_runs:
        return {
            'primary_stage': None,
            'consistency': 0,
            'confidence': 0.0,
            'secondary_stage': None,
            'secondary_confidence': None,
            'justification': '',
        }

    primary_counts = Counter(r['primary_stage'] for r in valid_runs)
    majority_stage, majority_count = primary_counts.most_common(1)[0]

    agreeing_runs = [r for r in valid_runs if r['primary_stage'] == majority_stage]
    avg_confidence = sum(r['primary_confidence'] for r in agreeing_runs) / len(agreeing_runs)

    # Secondary stage: most common non-None secondary
    secondary_stages = [
        r['secondary_stage'] for r in valid_runs if r['secondary_stage'] is not None
    ]
    secondary_stage = None
    secondary_confidence = None
    if secondary_stages:
        sec_counts = Counter(secondary_stages)
        secondary_stage = sec_counts.most_common(1)[0][0]
        sec_runs = [r for r in valid_runs if r['secondary_stage'] == secondary_stage]
        sec_confs = [r['secondary_confidence'] for r in sec_runs if r['secondary_confidence'] is not None]
        if sec_confs:
            secondary_confidence = sum(sec_confs) / len(sec_confs)

    justification = agreeing_runs[0]['justification'] if agreeing_runs else ''

    return {
        'primary_stage': majority_stage,
        'consistency': majority_count,
        'confidence': avg_confidence,
        'secondary_stage': secondary_stage,
        'secondary_confidence': secondary_confidence,
        'justification': justification,
    }


# ---------------------------------------------------------------------------
# Classification loop
# ---------------------------------------------------------------------------

def classify_segments_zero_shot(
    segments: List[Segment],
    api_key: str,
    model: str = 'openai/gpt-4o',
    temperature: float = 0.0,
    n_runs: int = 3,
    output_dir: str = './data/output/llm_labels/',
    randomize_codebook: bool = True,
    definitions: Optional[Dict] = None,
    backend: str = 'openrouter',
    replicate_api_token: str = '',
    max_new_tokens: int = 512,
    resume_from: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """
    Zero-shot classification of transcript segments using VA-MR framework.

    Adapted from loop_through_documents() in classify_reddit_llms.py:
        for i, (document, post_id) in tqdm(enumerate(zip(documents, post_ids))):
            codebook_string = create_codebook_string(codebook_dict, randomize=randomize)
            prompt = prompt_template.format(...)
            final_result, metadata = llm.openrouter_request(...)
            results_all[post_id] = final_result
            if i%20 == 0:  # Save intermediate results

    Key adaptations:
    1. Triplicate-and-flag: each segment classified n_runs times
    2. Randomized codebook order on each call to reduce order bias
    3. Temperature=0 for maximum determinism
    4. Intermediate saves every 20 segments (same cadence as original)
    """
    if definitions is None:
        definitions = stage_definitions

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')
    model_clean = model.replace('/', '_')

    # Resume from checkpoint if provided
    results_all: Dict[str, Any] = {}
    metadata_all: Dict[str, Any] = {}
    if resume_from and os.path.exists(resume_from):
        with open(resume_from, 'r') as f:
            results_all = json.load(f)
        print(f"  Resumed from checkpoint: {len(results_all)} segments already classified")

    participant_segments = [s for s in segments if s.speaker == 'participant']
    total = len(participant_segments)

    for i, segment in enumerate(participant_segments):
        # Skip already-classified segments when resuming
        if segment.segment_id in results_all:
            continue

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Classifying segment {i + 1}/{total}: {segment.segment_id}")

        segment_results = []
        for run in range(n_runs):
            codebook_string = create_vamr_codebook_string(
                definitions, randomize=randomize_codebook
            )
            prompt = VAMR_PROMPT_TEMPLATE.format(
                codebook_string=codebook_string,
                text=segment.text,
            )
            try:
                if backend == 'replicate':
                    result_text, meta = _replicate_request(
                        prompt, replicate_api_token, model=model,
                        temperature=temperature, max_new_tokens=max_new_tokens,
                    )
                else:
                    result_text, meta = _openrouter_request(
                        prompt, api_key, model=model, temperature=temperature
                    )
                segment_results.append(result_text)
            except Exception as e:
                print(f"  Error on {segment.segment_id}, run {run}: {e}")
                segment_results.append(None)

        parsed_runs = [_parse_single_run(r) for r in segment_results if r is not None]
        consistency_result = _compute_run_consistency(parsed_runs)

        results_all[segment.segment_id] = {
            'runs': segment_results,
            'parsed_runs': [r for r in parsed_runs if r is not None],
            'consistency': consistency_result,
        }
        metadata_all[segment.segment_id] = {
            'segment_id': segment.segment_id,
            'text': segment.text,
            'runs_raw': segment_results,
        }

        # Periodic saving (same cadence as classify_reddit_llms.py)
        if i % 20 == 0:
            _save_intermediate(results_all, metadata_all, output_dir, model_clean, timestamp)

    # Final save
    _save_intermediate(results_all, metadata_all, output_dir, model_clean, timestamp)

    return results_all, metadata_all


def _save_intermediate(
    results: Dict, metadata: Dict,
    output_dir: str, model_clean: str, timestamp: str
):
    """Save intermediate results to JSON files."""
    for name, data in [('results', results), ('metadata', metadata)]:
        path = os.path.join(
            output_dir, f'llm_{name}_{model_clean}_{timestamp}.json'
        )
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
