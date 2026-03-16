"""
response_parser.py
------------------
Multi-pass parsing and error classification for LLM theme classification
responses.

Generalized from vamr_labeling/response_parser.py. Key changes:
- Uses extract_json from classification_tools.llm_client
- Accepts a name_to_id map parameter instead of importing a global constant
- _parse_single_run imported from this package's llm_classifier
"""

import json
from typing import List, Dict, Any, Tuple

from .data_structures import Segment
from .llm_classifier import _parse_single_run


class ErrorCategory:
    """Expanded error taxonomy for LLM response classification."""
    API_ERROR = 'api_error'
    TIMEOUT = 'timeout'
    SAFETY_FILTER = 'safety_filter'
    MALFORMED_JSON = 'malformed_json'
    MISSING_FIELDS = 'missing_fields'
    INVALID_STAGE = 'invalid_stage'
    QUOTA_EXCEEDED = 'quota_exceeded'
    NULL_RESPONSE = 'null_response'


def _classify_error(run_str: str) -> str:
    """Classify an error run into a specific error category."""
    run_lower = run_str.lower()

    if 'timeout' in run_lower or 'timed out' in run_lower:
        return ErrorCategory.TIMEOUT
    if 'rate_limit' in run_lower or 'rate limit' in run_lower or '429' in run_str:
        return ErrorCategory.QUOTA_EXCEEDED
    if 'quota' in run_lower or 'billing' in run_lower or 'insufficient' in run_lower:
        return ErrorCategory.QUOTA_EXCEEDED
    if 'safety' in run_lower or 'content_filter' in run_lower or 'blocked' in run_lower:
        return ErrorCategory.SAFETY_FILTER
    if 'error' in run_lower:
        return ErrorCategory.API_ERROR
    return ErrorCategory.MALFORMED_JSON


def parse_all_results(
    results_all: Dict[str, Any],
    segments: List[Segment],
    name_to_id: Dict[str, int],
) -> Tuple[List[Segment], Dict[str, Any]]:
    """
    Parse and clean all LLM results, handling errors gracefully.

    Returns the updated segments list and a stats dictionary with
    expanded error taxonomy.

    Parameters
    ----------
    results_all : dict
        Raw LLM results keyed by segment_id.
    segments : list[Segment]
        Segment objects to update with parsed labels.
    name_to_id : dict
        Mapping from lowercase theme names/keys/aliases to integer IDs.
        Built from ThemeFramework.build_name_to_id_map().
    """
    parsed_segments: Dict[str, Dict] = {}
    could_not_parse: Dict[str, Any] = {}
    error_categories: Dict[str, List[str]] = {
        ErrorCategory.API_ERROR: [],
        ErrorCategory.TIMEOUT: [],
        ErrorCategory.SAFETY_FILTER: [],
        ErrorCategory.MALFORMED_JSON: [],
        ErrorCategory.MISSING_FIELDS: [],
        ErrorCategory.INVALID_STAGE: [],
        ErrorCategory.QUOTA_EXCEEDED: [],
        ErrorCategory.NULL_RESPONSE: [],
    }

    # First pass: use already-parsed consistency results
    for segment_id, result in results_all.items():
        # Handle multi-model results
        if 'cross_model_consensus' in result:
            consistency = result.get('cross_model_consensus', {})
        else:
            consistency = result.get('consistency', {})

        if consistency.get('primary_stage') is not None:
            parsed_segments[segment_id] = consistency
        else:
            could_not_parse[segment_id] = result

    # Second pass: attempt recovery on failures with granular error classification
    for segment_id, result in could_not_parse.items():
        raw_runs = result.get('runs', [])
        classified_error = None

        # Check for null responses (all runs returned None)
        non_null_runs = [r for r in raw_runs if r is not None]
        if not non_null_runs:
            error_categories[ErrorCategory.NULL_RESPONSE].append(segment_id)
            continue

        # Classify error type from run content
        for run in raw_runs:
            if run is None:
                continue
            run_str = str(run)
            error_type = _classify_error(run_str)

            if error_type in (ErrorCategory.SAFETY_FILTER, ErrorCategory.TIMEOUT,
                              ErrorCategory.QUOTA_EXCEEDED, ErrorCategory.API_ERROR):
                run_lower = run_str.lower()
                if any(indicator in run_lower for indicator in
                       ['error', 'safety', 'content_filter', 'timeout',
                        'rate_limit', '429', 'blocked']):
                    classified_error = error_type
                    break

        if classified_error:
            error_categories[classified_error].append(segment_id)
            continue

        # Aggressive JSON extraction
        recovered = False
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
                    parsed_result = _parse_single_run(parsed, name_to_id)
                    if parsed_result and parsed_result['primary_stage'] is not None:
                        parsed_segments[segment_id] = {
                            **parsed_result,
                            'consistency': 1,
                        }
                        recovered = True
                        break
                    elif parsed_result and parsed_result['primary_stage'] is None:
                        error_categories[ErrorCategory.INVALID_STAGE].append(segment_id)
                        recovered = True
                        break
                    else:
                        error_categories[ErrorCategory.MISSING_FIELDS].append(segment_id)
                        recovered = True
                        break
            except json.JSONDecodeError:
                continue
            except Exception:
                continue

        if not recovered:
            error_categories[ErrorCategory.MALFORMED_JSON].append(segment_id)

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

        # Store multi-model metadata if available
        if 'model_agreement' in parsed:
            seg.model_agreement = parsed.get('model_agreement')
        if 'model_predictions' in parsed:
            seg.model_predictions = parsed.get('model_predictions')
        if 'total_models' in parsed:
            seg.total_models = parsed.get('total_models')

    # Stats with expanded error taxonomy
    total = len(results_all)
    parsed_count = len(parsed_segments)
    total_errors = sum(len(ids) for ids in error_categories.values())

    stats = {
        'parsed': parsed_count,
        'total_errors': total_errors,
        'error_breakdown': {
            category: {
                'count': len(ids),
                'segment_ids': ids,
            }
            for category, ids in error_categories.items()
            if len(ids) > 0
        },
        # Backward-compatible keys
        'errors': (
            len(error_categories[ErrorCategory.API_ERROR])
            + len(error_categories[ErrorCategory.TIMEOUT])
            + len(error_categories[ErrorCategory.QUOTA_EXCEEDED])
        ),
        'safety_filtered': len(error_categories[ErrorCategory.SAFETY_FILTER]),
        'unrecoverable': (
            len(error_categories[ErrorCategory.MALFORMED_JSON])
            + len(error_categories[ErrorCategory.MISSING_FIELDS])
            + len(error_categories[ErrorCategory.INVALID_STAGE])
            + len(error_categories[ErrorCategory.NULL_RESPONSE])
        ),
        'error_ids': (
            error_categories[ErrorCategory.API_ERROR]
            + error_categories[ErrorCategory.TIMEOUT]
            + error_categories[ErrorCategory.QUOTA_EXCEEDED]
        ),
        'safety_ids': error_categories[ErrorCategory.SAFETY_FILTER],
    }

    print(f"Parsing results: {parsed_count}/{total} parsed successfully")
    if total_errors > 0:
        print(f"  Total errors: {total_errors}")
        for category, ids in error_categories.items():
            if ids:
                print(f"    {category}: {len(ids)}")

    return segments, stats
