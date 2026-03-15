"""
classification_loop.py
----------------------
Shared classification loop used by both theme_labeler (single-label)
and codebook_classifier (multi-label) modules.

Provides:
- ``filter_participant_segments`` — extract participant-only segments
- ``classify_segments`` — the shared N-run-per-segment loop with
  periodic checkpointing and optional resume
- ``_save_checkpoint`` — write intermediate results to JSON
"""

import json
import os
import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .data_structures import Segment
from .llm_client import LLMClient

T = TypeVar('T')


def filter_participant_segments(segments: List[Segment]) -> List[Segment]:
    """Return only segments where ``speaker == 'participant'``."""
    return [s for s in segments if s.speaker == 'participant']


def classify_segments(
    segments: List[Segment],
    client: LLMClient,
    n_runs: int,
    build_prompt: Callable[[Segment, int], str],
    parse_response: Callable[[str], Any],
    merge_runs: Callable[[List[Any]], Any],
    output_dir: Optional[str] = None,
    save_interval: int = 20,
    resume_from: Optional[str] = None,
    file_prefix: str = 'classification',
    model_tag: Optional[str] = None,
    serialize_result: Optional[Callable[[Any], Any]] = None,
) -> Dict[str, Any]:
    """
    Shared classification loop for N LLM runs per segment.

    Parameters
    ----------
    segments : list of Segment
        Already filtered to participant-only segments.
    client : LLMClient
        Configured LLM API client.
    n_runs : int
        Number of independent LLM runs per segment.
    build_prompt : callable(segment, run_index) -> str
        Builds the prompt for a given segment and run.
    parse_response : callable(response_text) -> parsed
        Parses a single LLM response string into a structured result.
        Should return None on parse failure.
    merge_runs : callable(list_of_parsed) -> merged
        Aggregates parsed results across runs (majority vote, etc.).
    output_dir : str or None
        Directory for checkpoint files.  Created if it doesn't exist.
    save_interval : int
        Save a checkpoint every *save_interval* segments.
    resume_from : str or None
        Path to a JSON checkpoint to resume from.
    file_prefix : str
        Prefix for checkpoint filenames.
    model_tag : str or None
        Model identifier used in checkpoint filenames.
    serialize_result : callable or None
        Optional function to make a result JSON-serializable for
        checkpointing.  If None, results are stored as-is.

    Returns
    -------
    dict mapping segment_id -> merged result
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')

    # Resume from checkpoint if provided
    results: Dict[str, Any] = {}
    if resume_from and os.path.exists(resume_from):
        with open(resume_from, 'r') as f:
            results = json.load(f)
        print(f"  Resumed from checkpoint: {len(results)} segments already classified")

    total = len(segments)

    for i, segment in enumerate(segments):
        if segment.segment_id in results:
            continue

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Classifying segment {i + 1}/{total}: {segment.segment_id}")

        run_results: List[Any] = []
        for run in range(n_runs):
            prompt = build_prompt(segment, run)
            try:
                result_text, _ = client.request(prompt)
                if result_text is not None:
                    parsed = parse_response(result_text)
                    if parsed is not None:
                        run_results.append(parsed)
            except Exception as e:
                print(f"  Error on {segment.segment_id}, run {run}: {e}")

        merged = merge_runs(run_results)
        results[segment.segment_id] = merged

        if output_dir and i % save_interval == 0:
            _save_checkpoint(
                results, output_dir, file_prefix, model_tag,
                timestamp, serialize_result,
            )

    if output_dir:
        _save_checkpoint(
            results, output_dir, file_prefix, model_tag,
            timestamp, serialize_result,
        )

    return results


def _save_checkpoint(
    results: Dict[str, Any],
    output_dir: str,
    file_prefix: str,
    model_tag: Optional[str],
    timestamp: str,
    serialize_fn: Optional[Callable[[Any], Any]] = None,
):
    """Write intermediate results to a JSON checkpoint file."""
    tag = f"_{model_tag}" if model_tag else ''
    path = os.path.join(output_dir, f'{file_prefix}{tag}_{timestamp}.json')

    if serialize_fn is not None:
        serializable = {
            seg_id: serialize_fn(val) for seg_id, val in results.items()
        }
    else:
        serializable = results

    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
