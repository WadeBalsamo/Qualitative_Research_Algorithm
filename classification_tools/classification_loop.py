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
- ``_write_status_entry`` — append live segment status to llm_status.txt
"""

import json
import os
import datetime
import textwrap
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

from .data_structures import Segment
from .llm_client import LLMClient

T = TypeVar('T')

# Stage mapping: index to name (VAMMR framework — display only, not classification logic)
STAGE_NAMES = {
    0: 'Vigilance',
    1: 'Avoidance',
    2: 'Mindfulness',
    3: 'Metacognition',
    4: 'Reappraisal',
}


def _ms_to_timecode(ms: int) -> str:
    """Convert milliseconds to SRT timecode format (HH:MM:SS.mmm)."""
    total_seconds = ms // 1000
    milliseconds = ms % 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _stage_name(stage_id: Any) -> str:
    """Convert stage ID (int) to stage name, or return original if not found."""
    if isinstance(stage_id, int) and stage_id in STAGE_NAMES:
        return STAGE_NAMES[stage_id]
    return str(stage_id)


def filter_participant_segments(segments: List[Segment]) -> List[Segment]:
    """Return only segments where ``speaker == 'participant'``."""
    return [s for s in segments if s.speaker == 'participant']


def classify_segments(
    segments: List[Segment],
    client: LLMClient,
    n_runs: int,
    build_prompt: Callable[..., str],
    parse_response: Callable[[str], Any],
    merge_runs: Callable[[List[Any]], Any],
    output_dir: Optional[str] = None,
    save_interval: int = 20,
    resume_from: Optional[str] = None,
    file_prefix: str = 'classification',
    model_tag: Optional[str] = None,
    serialize_result: Optional[Callable[[Any], Any]] = None,
    per_run_models: Optional[List[str]] = None,
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
    build_prompt : callable(segment, run_index, all_segments, seg_index) -> str
        Builds the prompt for a given segment and run.  Receives the full
        segment list and current index so it can include surrounding context.
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
    per_run_models : list of str or None
        When provided and ``len == n_runs``, run *i* uses ``per_run_models[i]``
        instead of ``client.config.model``.  Enables distinct-model interrater
        reliability: each run is an independent rater.  Uses a model-first sweep
        — all segments are classified with model 0, then all with model 1, etc.
        — so each model is loaded only once per pass rather than reloaded for
        every segment.  Early-exit optimisation is disabled (all raters always
        run).

    Returns
    -------
    dict mapping segment_id -> merged result
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')

    total = len(segments)

    use_per_run_models = (
        per_run_models is not None and len(per_run_models) == n_runs
    )

    # Prepare live status file
    status_path = None
    if output_dir:
        status_path = os.path.join(output_dir, 'llm_classification_log.txt')
        with open(status_path, 'a') as sf:
            sf.write(f"\n{'=' * 80}\n")
            sf.write(f"Classification Run: {file_prefix}\n")
            sf.write(f"Started: {datetime.datetime.utcnow().isoformat()}Z\n")
            sf.write(f"Total segments: {total}\n")
            if use_per_run_models:
                sf.write(f"Mode: model-first ({n_runs} models, 1 sweep each)\n")
            sf.write("=" * 80 + "\n\n")
        print(f"  Live status log: llm_classification_log.txt")

    # Model-first path: process all segments with each model before switching
    if use_per_run_models:
        return _classify_segments_model_first(
            segments=segments,
            client=client,
            per_run_models=per_run_models,
            n_runs=n_runs,
            build_prompt=build_prompt,
            parse_response=parse_response,
            merge_runs=merge_runs,
            output_dir=output_dir,
            save_interval=save_interval,
            resume_from=resume_from,
            file_prefix=file_prefix,
            model_tag=model_tag,
            timestamp=timestamp,
            status_path=status_path,
            serialize_result=serialize_result,
        )

    # --- Single-model (segment-first) path below ---

    # Resume from checkpoint if provided
    results: Dict[str, Any] = {}
    if resume_from and os.path.exists(resume_from):
        with open(resume_from, 'r') as f:
            results = json.load(f)
        print(f"  Resumed from checkpoint: {len(results)} segments already classified")

    ok_count = 0
    error_count = 0

    for i, segment in enumerate(segments):
        if segment.segment_id in results:
            continue

        # Print progress for every segment so the terminal stays alive
        pct = f" ({error_count}/{ok_count + error_count} errors)" if (ok_count + error_count) > 0 else ""
        snippet = segment.text.replace('\n', ' ')
        if len(segment.text) > 80:
            snippet += "..."
        print(f"  [{i + 1}/{total}] {segment.segment_id}{pct}")
        print(f"           \"{snippet}\"")

        # Preserve slot positions: run_results[k] is the ballot from rater k,
        # or None when that run failed to produce a parseable response.
        # All n_runs always execute — no early-exit — so every rater gets a
        # chance to cast a ballot. Early-exit would bias IRR estimates.
        run_results: List[Any] = [None] * n_runs
        for run in range(n_runs):
            prompt = build_prompt(segment, run, segments, i)
            try:
                result_text, _ = client.request(prompt)
                if result_text is not None:
                    parsed = parse_response(result_text)
                    run_results[run] = parsed
            except Exception as e:
                print(f"  Error on {segment.segment_id}, run {run}: {e}")

        if any(r is not None for r in run_results):
            ok_count += 1
        else:
            error_count += 1

        merged = merge_runs(run_results)
        results[segment.segment_id] = merged

        # Write live status entry
        if status_path:
            _write_status_entry(status_path, segment, i, total, merged, run_results)

        if output_dir and i % save_interval == 0:
            _save_checkpoint(
                results, output_dir, file_prefix, model_tag,
                timestamp, serialize_result,
            )

    if total > 0:
        print(f"  Classification complete: {ok_count} ok, {error_count} errors out of {total}")
        if status_path:
            with open(status_path, 'a') as sf:
                sf.write("\n" + "=" * 80 + "\n")
                sf.write(f"COMPLETE: {ok_count} ok, {error_count} errors out of {total}\n")
                sf.write(f"Finished: {datetime.datetime.utcnow().isoformat()}Z\n")

    if output_dir:
        _save_checkpoint(
            results, output_dir, file_prefix, model_tag,
            timestamp, serialize_result,
        )

    return results


def _classify_segments_model_first(
    segments: List['Segment'],
    client: 'LLMClient',
    per_run_models: List[str],
    n_runs: int,
    build_prompt: Callable[..., str],
    parse_response: Callable[[str], Any],
    merge_runs: Callable[[List[Any]], Any],
    output_dir: Optional[str],
    save_interval: int,
    resume_from: Optional[str],
    file_prefix: str,
    model_tag: Optional[str],
    timestamp: str,
    status_path: Optional[str],
    serialize_result: Optional[Callable[[Any], Any]],
) -> Dict[str, Any]:
    """
    Model-first classification: classify all segments with model 0, then model 1,
    then model 2 (etc.), so each model is loaded only once per pass.

    Intermediate per-run results are checkpointed to ``*_runs.json`` after each
    full sweep.  The final merged checkpoint is written in the same format as the
    single-model path.
    """
    total = len(segments)
    original_model = client.config.model

    # -- Resume --
    run_results: Dict[str, Dict[str, Any]] = {}   # seg_id → {run_idx_str → parsed}
    completed_runs: Set[int] = set()
    if resume_from and os.path.exists(resume_from):
        run_results, completed_runs = _load_runs_checkpoint(resume_from, n_runs)
        if completed_runs:
            print(f"  Resumed: runs {sorted(completed_runs)} already complete")

    # -- Sweep phase: one full pass per model --
    for run_idx, model in enumerate(per_run_models):
        if run_idx in completed_runs:
            print(f"  Run {run_idx + 1}/{n_runs} ({model}): already complete, skipping")
            continue

        client.config.model = model
        print(f"  Run {run_idx + 1}/{n_runs}: {model}")

        for i, segment in enumerate(segments):
            seg_id = segment.segment_id
            if str(run_idx) in run_results.get(seg_id, {}):
                continue   # already classified in a prior attempt

            prompt = build_prompt(segment, run_idx, segments, i)
            parsed = None
            try:
                result_text, _ = client.request(prompt)
                if result_text is not None:
                    parsed = parse_response(result_text)
            except Exception as e:
                print(f"  Error on {seg_id}, run {run_idx}: {e}")

            run_results.setdefault(seg_id, {})[str(run_idx)] = parsed

            snippet = segment.text#[:60].replace('\n', ' ')
            print(f"  [Run {run_idx + 1}/{n_runs} | Seg {i + 1}/{total}] {seg_id}: \"{snippet}...\"")
            # print the result of this run for the current segment, if parseable
            if parsed is not None:
                print(f"    → Parsed result: {parsed}")
            else:
                print(f"    → No parseable result")
            if output_dir and i % save_interval == 0:
                _save_runs_checkpoint(
                    run_results, completed_runs, n_runs, per_run_models,
                    output_dir, file_prefix, model_tag, timestamp,
                )

        completed_runs.add(run_idx)
        if output_dir:
            _save_runs_checkpoint(
                run_results, completed_runs, n_runs, per_run_models,
                output_dir, file_prefix, model_tag, timestamp,
            )

    client.config.model = original_model

    # -- Merge phase --
    # Preserve per-rater slot alignment: slot k always corresponds to
    # per_run_models[k], with None marking rater k's parse failure. The
    # merge_runs callback in llm_classifier wraps vote_single_label with
    # the pre-configured rater_ids, so ordering matters.
    results: Dict[str, Any] = {}
    ok_count = 0
    error_count = 0
    for i, segment in enumerate(segments):
        seg_id = segment.segment_id
        seg_run_data = run_results.get(seg_id, {})
        slot_ballots = [seg_run_data.get(str(r)) for r in range(n_runs)]
        if any(p is not None for p in slot_ballots):
            ok_count += 1
        else:
            error_count += 1
        merged = merge_runs(slot_ballots)
        results[seg_id] = merged
        if status_path:
            _write_status_entry(status_path, segment, i, total, merged,
                                slot_ballots, run_model_names=per_run_models)

    print(f"  Classification complete: {ok_count} ok, {error_count} errors out of {total}")
    if status_path:
        with open(status_path, 'a') as sf:
            sf.write("\n" + "=" * 80 + "\n")
            sf.write(f"COMPLETE: {ok_count} ok, {error_count} errors out of {total}\n")
            sf.write(f"Finished: {datetime.datetime.utcnow().isoformat()}Z\n")

    if output_dir:
        _save_checkpoint(results, output_dir, file_prefix, model_tag, timestamp, serialize_result)

    return results


def _save_runs_checkpoint(
    run_results: Dict[str, Dict[str, Any]],
    completed_runs: Set[int],
    n_runs: int,
    per_run_models: List[str],
    output_dir: str,
    file_prefix: str,
    model_tag: Optional[str],
    timestamp: str,
) -> None:
    """Write per-run intermediate results for the model-first path."""
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    tag = f"_{model_tag}" if model_tag else ''
    path = os.path.join(checkpoint_dir, f'{file_prefix}{tag}_{timestamp}_runs.json')
    payload = {
        "_meta": {
            "format": "model_first_v1",
            "n_runs": n_runs,
            "per_run_models": per_run_models,
            "completed_runs": sorted(completed_runs),
        },
        "run_results": run_results,
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)


def _load_runs_checkpoint(
    path: str,
    n_runs: int,
) -> Tuple[Dict[str, Dict[str, Any]], Set[int]]:
    """
    Load a runs checkpoint written by ``_save_runs_checkpoint``.

    Returns ``(run_results, completed_runs)``.  If the file is in the legacy
    merged format (not model_first_v1), warns and returns empty state so the
    caller restarts from scratch.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and data.get('_meta', {}).get('format') == 'model_first_v1':
        run_results = data.get('run_results', {})
        completed_runs = set(data['_meta'].get('completed_runs', []))
        return run_results, completed_runs
    # Legacy merged checkpoint — cannot restore per-run state
    print(f"  Warning: {os.path.basename(path)} is a legacy merged checkpoint; "
          f"per-run resume not available. Re-classifying.")
    return {}, set()


def _save_checkpoint(
    results: Dict[str, Any],
    output_dir: str,
    file_prefix: str,
    model_tag: Optional[str],
    timestamp: str,
    serialize_fn: Optional[Callable[[Any], Any]] = None,
):
    """Write intermediate results to a JSON checkpoint file."""
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    tag = f"_{model_tag}" if model_tag else ''
    path = os.path.join(checkpoint_dir, f'{file_prefix}{tag}_{timestamp}.json')

    if serialize_fn is not None:
        serializable = {
            seg_id: serialize_fn(val) for seg_id, val in results.items()
        }
    else:
        serializable = results

    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)


def _write_status_entry(
    status_path: str,
    segment: 'Segment',
    index: int,
    total: int,
    merged: Any,
    run_results: List[Any],
    run_model_names: Optional[List[str]] = None,
):
    """Append a human-readable status entry for one segment to the status file.

    Reads the unified merge-result shape:
        {'rater_ids': [...], 'rater_votes': [...], 'consensus': {...}}
    """
    with open(status_path, 'a') as sf:
        start_tc = _ms_to_timecode(segment.start_time_ms)
        end_tc = _ms_to_timecode(segment.end_time_ms)
        sf.write(f" {segment.segment_id}\n\n")
        sf.write(f"  Session: {segment.session_id}  |  [{index + 1}/{total}]  |  "
                 f"Time: {start_tc} --> {end_tc}\n")
        sf.write("-" * 60 + "\n")

        sf.write("SEGMENT TEXT:\n")
        for line in textwrap.wrap(segment.text, width=76, initial_indent="  ", subsequent_indent="  "):
            sf.write(line + "\n")
        sf.write("\n")

        if not isinstance(merged, dict):
            sf.write(f"  {str(merged)[:200]}\n")
            sf.write("\n" + "=" * 80 + "\n\n")
            return

        rater_votes = merged.get('rater_votes') or []
        rater_ids = merged.get('rater_ids') or run_model_names or []
        consensus = merged.get('consensus') or {}

        if rater_votes:
            sf.write(f"RATER BALLOTS ({len(rater_votes)}):\n")
            for r, rv in enumerate(rater_votes):
                rid = rv.get('rater') or (rater_ids[r] if r < len(rater_ids) else f'run_{r + 1}')
                vote_kind = rv.get('vote', '?')
                stage = rv.get('stage')
                conf = rv.get('confidence')
                sec = rv.get('secondary_stage')
                just = rv.get('justification') or ''
                if vote_kind == 'CODED':
                    conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf)
                    sec_s = f"  secondary={_stage_name(sec)}" if sec is not None else ""
                    sf.write(f"  [{rid}] CODED stage={_stage_name(stage)}  conf={conf_s}{sec_s}\n")
                elif vote_kind == 'ABSTAIN':
                    sf.write(f"  [{rid}] ABSTAIN (irrelevant to framework)\n")
                else:
                    sf.write(f"  [{rid}] ERROR (no parseable response)\n")
                if just:
                    for line in textwrap.wrap(just, width=72, initial_indent="    → ", subsequent_indent="      "):
                        sf.write(line + "\n")

        sf.write("\nCONSENSUS:\n")
        agreement = consensus.get('agreement_level', '?')
        n_agree = consensus.get('n_agree', 0)
        n_raters = consensus.get('n_raters', len(rater_votes))
        consensus_vote = consensus.get('consensus_vote')
        needs_review = consensus.get('needs_review', False)

        if consensus_vote == 'ABSTAIN':
            sf.write(f"  Result: UNCLASSIFIED (consensus ABSTAIN)\n")
        elif consensus.get('primary_stage') is None:
            sf.write(f"  Result: UNCLASSIFIED ({agreement})\n")
        else:
            conf = consensus.get('primary_confidence', 0.0)
            sf.write(f"  Result: CLASSIFIED as {_stage_name(consensus['primary_stage'])}\n")
            sf.write(f"  Mean confidence: {conf:.3f}\n")
            sec = consensus.get('secondary_stage')
            if sec is not None:
                sec_conf = consensus.get('secondary_confidence')
                sec_conf_s = f" ({sec_conf:.2f})" if isinstance(sec_conf, (int, float)) else ""
                sf.write(f"  Secondary: {_stage_name(sec)}{sec_conf_s}\n")
            just = consensus.get('justification') or ''
            if just:
                for line in textwrap.wrap(just, width=72,
                                          initial_indent="  Justification: ",
                                          subsequent_indent="    "):
                    sf.write(line + "\n")

        sf.write(f"  Agreement: {agreement}  ({n_agree}/{n_raters} raters)\n")
        if consensus.get('tie_broken_by_confidence'):
            sf.write(f"  ↳ TIE BROKEN BY CONFIDENCE\n")
        if needs_review:
            sf.write(f"  ↳ FLAGGED FOR HUMAN REVIEW\n")

        sf.write("\n" + "=" * 80 + "\n\n")
