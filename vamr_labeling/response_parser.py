"""
response_parser.py
------------------
Stage 4: Response Parsing and Error Handling.

Directly adapts the error-handling cascade from classify_reddit_llms.py,
which handles the messy reality of LLM outputs through multiple parsing
passes:

    # First pass: try direct eval
    for post_id in metadata_all_i.keys():
        try:
            response = metadata_all_i[post_id]['choices'][0]['message']['content']
            responses_all[post_id] = eval(response)
        except:
            could_not_eval[post_id] = metadata_all_i[post_id]

    # Second pass: handle edge cases
    for key, value in could_not_eval.items():
        if 'error' in str(value) or 'SAFETY' in str(value):
            request_errors[key] = value
        else:
            # try truncation fixes, JSON extraction, etc.

We replicate this exact pattern with VA-MR-specific adaptations.
"""

import json
from typing import List, Dict, Any, Tuple

from .data_structures import Segment
from .zero_shot_classifier import _parse_single_run, _process_api_output


def parse_all_results(
    results_all: Dict[str, Any],
    segments: List[Segment],
) -> Tuple[List[Segment], Dict[str, Any]]:
    """
    Parse and clean all LLM results, handling errors gracefully.

    Returns the updated segments list and a stats dictionary.
    """
    parsed_segments: Dict[str, Dict] = {}
    could_not_parse: Dict[str, Any] = {}
    request_errors: Dict[str, Any] = {}
    safety_filtered: Dict[str, Any] = {}

    # First pass: use already-parsed consistency results
    for segment_id, result in results_all.items():
        consistency = result.get('consistency', {})
        if consistency.get('primary_stage') is not None:
            parsed_segments[segment_id] = consistency
        else:
            could_not_parse[segment_id] = result

    # Second pass: attempt recovery on failures
    for segment_id, result in could_not_parse.items():
        raw_runs = result.get('runs', [])
        is_error = False

        for run in raw_runs:
            if run is None:
                continue
            run_str = str(run)
            if 'error' in run_str.lower():
                request_errors[segment_id] = result
                is_error = True
                break
            if 'SAFETY' in run_str or 'content_filter' in run_str:
                safety_filtered[segment_id] = result
                is_error = True
                break

        if is_error:
            continue

        # Aggressive JSON extraction (from llm.py's process_api_output)
        for run in raw_runs:
            if run is None:
                continue
            try:
                run_str = str(run)
                start = run_str.find('{')
                end = run_str.rfind('}') + 1
                if start != -1 and end > start:
                    json_part = run_str[start:end]
                    parsed = json.loads(json_part)
                    parsed_result = _parse_single_run(parsed)
                    if parsed_result and parsed_result['primary_stage'] is not None:
                        parsed_segments[segment_id] = {
                            **parsed_result,
                            'consistency': 1,
                        }
                        break
            except Exception:
                continue

    # Apply parsed results to segment objects
    segment_lookup = {s.segment_id: s for s in segments}

    for segment_id, parsed in parsed_segments.items():
        seg = segment_lookup.get(segment_id)
        if seg is None:
            continue
        seg.primary_stage = parsed.get('primary_stage')
        seg.secondary_stage = parsed.get('secondary_stage')
        seg.llm_confidence_primary = parsed.get('confidence', 0.0)
        seg.llm_confidence_secondary = parsed.get('secondary_confidence')
        seg.llm_justification = parsed.get('justification', '')
        seg.llm_run_consistency = parsed.get('consistency', 0)

    # Stats
    total = len(results_all)
    parsed_count = len(parsed_segments)
    error_count = len(request_errors)
    safety_count = len(safety_filtered)
    unrecoverable = total - parsed_count - error_count - safety_count

    stats = {
        'parsed': parsed_count,
        'errors': error_count,
        'safety_filtered': safety_count,
        'unrecoverable': unrecoverable,
        'error_ids': list(request_errors.keys()),
        'safety_ids': list(safety_filtered.keys()),
    }

    print(f"Parsing results: {parsed_count}/{total} parsed successfully")
    print(f"  Request errors: {error_count}")
    print(f"  Safety filtered: {safety_count}")
    print(f"  Unrecoverable: {unrecoverable}")

    return segments, stats
