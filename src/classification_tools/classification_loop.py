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


# Transient empty/unparseable responses are recoverable.  Under LM Studio
# model-rotation + context pressure a capable model intermittently returns a
# bare ``{}`` (no content, no reasoning) for a long prompt; a re-request of the
# SAME prompt almost always yields valid JSON (verified empirically against the
# PURER cue prompt).  We therefore retry a *parse failure* — a successful HTTP
# call whose ``parse_response`` returns ``None`` — a bounded number of times
# before recording a permanent ``None``.  (Network/HTTP errors are already
# retried inside ``LLMClient.request``; this covers the parse layer, which was
# the cause of the all-NULL PURER overlay incident.)
_PARSE_RETRY_ATTEMPTS = 3


def _request_and_parse(client, prompt, parse_response, seg_id, run_idx,
                       attempts: int = _PARSE_RETRY_ATTEMPTS):
    """Request + parse with bounded retries on a parse failure.

    Returns the parsed ballot, or ``None`` if every attempt failed to parse.
    """
    parsed = None
    for attempt in range(1, attempts + 1):
        try:
            result_text, _ = client.request(prompt)
            if result_text is not None:
                parsed = parse_response(result_text)
        except Exception as e:
            print(f"  Error on {seg_id}, run {run_idx} "
                  f"(attempt {attempt}/{attempts}): {e}")
            parsed = None
        if parsed is not None:
            return parsed
        if attempt < attempts:
            print(f"  Unparseable response for {seg_id}, run {run_idx} "
                  f"(attempt {attempt}/{attempts}) — retrying")
    return parsed


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
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    checkpoint_path: Optional[str] = None,
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
    on_progress : callable({seg_id: ballot|None}) or None
        Optional callback invoked right after every checkpoint save with the
        per-rater cells accumulated *since the last call* (``{segment_id: parsed
        ballot | None}``; ``None`` is a parse failure → an ERROR ballot).  Lets a
        caller (the M3 run executor) flush durable ballots mid-sweep without
        ``classification_tools`` importing ``process.db``.  Also called once more
        at sweep end.  No-op when None (the legacy inline path passes nothing).
    checkpoint_path : str or None
        When given (the M3 run executor's per-run mode), this EXACT path is used
        for both resume-read and every model-first checkpoint write — a stable,
        deterministic ``{prefix}_run{run_id:04d}_runs.json`` with no timestamp, so
        ``resume_from`` always matches the file the prior attempt wrote (no glob
        races).  When None the legacy timestamped filename is used.

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
            on_progress=on_progress,
            checkpoint_path=checkpoint_path,
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

    # Crash-safety: a try/finally guarantees a checkpoint write even on an
    # abnormal exit (KeyboardInterrupt, unrecoverable error) so the trailing
    # (< save_interval) cells of an in-flight sweep are not lost — the run
    # resumes from the checkpoint instead of re-fetching completed segments.
    try:
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
                run_results[run] = _request_and_parse(
                    client, prompt, parse_response, segment.segment_id, run)

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
    finally:
        if output_dir:
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
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Model-first classification: classify all segments with model 0, then model 1,
    then model 2 (etc.), so each model is loaded only once per pass.

    Intermediate per-run results are checkpointed to ``*_runs.json`` after each
    full sweep.  The final merged checkpoint is written in the same format as the
    single-model path.

    ``on_progress`` (when given) is invoked right after every checkpoint save
    with the per-rater ballots produced *since the last call* (``{seg_id:
    parsed|None}`` for the current run).  A ``try/finally`` guarantees a
    checkpoint save (and a final flush) even on an abnormal exit, so an
    interrupted sweep never loses up to ``save_interval - 1`` trailing cells.
    """
    total = len(segments)
    original_model = client.config.model

    def _save():
        """Persist the per-run checkpoint (deterministic path in executor mode)."""
        if not output_dir:
            return
        _save_runs_checkpoint(
            run_results, completed_runs, n_runs, per_run_models,
            output_dir, file_prefix, model_tag, timestamp,
            checkpoint_path=checkpoint_path,
        )

    # -- Resume --
    run_results: Dict[str, Dict[str, Any]] = {}   # seg_id → {run_idx_str → parsed}
    completed_runs: Set[int] = set()
    # In executor mode the resume source is the deterministic per-run checkpoint
    # (matches exactly what _save writes); otherwise the caller's resume_from.
    _resume_src = checkpoint_path if checkpoint_path else resume_from
    if _resume_src and os.path.exists(_resume_src):
        run_results, completed_runs = _load_runs_checkpoint(_resume_src, n_runs)
        if completed_runs:
            print(f"  Resumed: runs {sorted(completed_runs)} already complete")

    # Track which (seg_id, run_idx) cells on_progress has already seen so each
    # callback gets only the delta since the previous save.  Cells already in the
    # resumed checkpoint are considered already-flushed (the executor upserted
    # them on the prior attempt).
    flushed: Set[tuple] = set()
    if on_progress is not None:
        for sid, by_run in run_results.items():
            for r_key in by_run:
                flushed.add((sid, r_key))

    def _flush_progress(run_idx: int) -> None:
        """Emit ballots for run_idx produced since the last flush, then mark them."""
        if on_progress is None:
            return
        r_key = str(run_idx)
        delta: Dict[str, Any] = {}
        for sid, by_run in run_results.items():
            if r_key in by_run and (sid, r_key) not in flushed:
                delta[sid] = by_run[r_key]
                flushed.add((sid, r_key))
        if delta:
            on_progress(delta)

    # -- Sweep phase: one full pass per model --
    try:
        for run_idx, model in enumerate(per_run_models):
            if run_idx in completed_runs:
                print(f"  Run {run_idx + 1}/{n_runs} ({model}): already complete, skipping")
                continue

            client.config.model = model
            print(f"  Run {run_idx + 1}/{n_runs}: {model}")
            if client.config.backend == 'lmstudio' and not client.check_loaded_model(model):
                print(
                    f"\n  *** MODEL MISMATCH WARNING ***\n"
                    f"  Requested : {model}\n"
                    f"  LMStudio does not appear to have this model loaded.\n"
                    f"  Load '{model}' in LMStudio, then recover with:\n"
                    f"    qra reclassify-run --output-dir <output_dir> --run {run_idx + 1} --model {model}\n"
                    f"  Continuing — results for this run will use whatever LMStudio has loaded.\n"
                )

            for i, segment in enumerate(segments):
                seg_id = segment.segment_id
                if str(run_idx) in run_results.get(seg_id, {}):
                    continue   # already classified in a prior attempt

                prompt = build_prompt(segment, run_idx, segments, i)
                parsed = _request_and_parse(
                    client, prompt, parse_response, seg_id, run_idx)

                run_results.setdefault(seg_id, {})[str(run_idx)] = parsed

                snippet = segment.text#[:60].replace('\n', ' ')
                print(f"  [Run {run_idx + 1}/{n_runs} | Seg {i + 1}/{total}] {seg_id}: \"{snippet}...\"")
                # print the result of this run for the current segment, if parseable
                if parsed is not None:
                    print(f"    → Parsed result: {parsed}")
                else:
                    print(f"    → No parseable result")
                if output_dir and i % save_interval == 0:
                    _save()
                    _flush_progress(run_idx)

            completed_runs.add(run_idx)
            _save()
            _flush_progress(run_idx)
    finally:
        # Crash-safety: ALWAYS persist whatever has been classified so an
        # abnormal exit (KeyboardInterrupt, network failure) is resumable from
        # the checkpoint rather than discarding the trailing (< save_interval)
        # cells of the in-flight sweep.  Flush them to the executor too.
        _save()
        if on_progress is not None:
            for run_idx in range(n_runs):
                _flush_progress(run_idx)
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


def patch_runs_checkpoint(
    checkpoint_path: str,
    run_idx: int,
    new_model: Optional[str] = None,
) -> List[str]:
    """
    Remove one run's results from a model_first_v1 checkpoint so it can be
    re-classified without disturbing the other runs.

    Removes *run_idx* from ``completed_runs``, deletes the ``str(run_idx)``
    key from every segment's result dict, and (if *new_model* is given)
    updates ``per_run_models[run_idx]``.  Writes the patched data back to
    *checkpoint_path* in place and returns the (possibly updated)
    ``per_run_models`` list.

    Raises ``ValueError`` if the file is not in ``model_first_v1`` format or
    if *run_idx* is out of range.
    """
    with open(checkpoint_path, 'r') as f:
        data = json.load(f)

    meta = data.get('_meta', {})
    if meta.get('format') != 'model_first_v1':
        raise ValueError(
            f"{os.path.basename(checkpoint_path)} is not a model_first_v1 checkpoint "
            f"(format={meta.get('format')!r}). Only model-first runs checkpoints can be patched."
        )

    per_run_models: List[str] = meta.get('per_run_models', [])
    n_runs: int = meta.get('n_runs', len(per_run_models))

    if not (0 <= run_idx < n_runs):
        raise ValueError(
            f"run_idx {run_idx} is out of range for a {n_runs}-run checkpoint "
            f"(valid: 0–{n_runs - 1})."
        )

    if new_model is not None:
        per_run_models[run_idx] = new_model
        meta['per_run_models'] = per_run_models

    completed_runs: List[int] = meta.get('completed_runs', [])
    meta['completed_runs'] = [r for r in completed_runs if r != run_idx]

    run_key = str(run_idx)
    cleared = 0
    for seg_data in data.get('run_results', {}).values():
        if run_key in seg_data:
            del seg_data[run_key]
            cleared += 1

    with open(checkpoint_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(
        f"  Patched checkpoint: run {run_idx + 1} removed from completed_runs, "
        f"{cleared} segment result(s) cleared."
    )
    if new_model is not None:
        print(f"  Updated per_run_models[{run_idx}] → {new_model}")

    return per_run_models


def _save_runs_checkpoint(
    run_results: Dict[str, Dict[str, Any]],
    completed_runs: Set[int],
    n_runs: int,
    per_run_models: List[str],
    output_dir: str,
    file_prefix: str,
    model_tag: Optional[str],
    timestamp: str,
    checkpoint_path: Optional[str] = None,
) -> None:
    """Write per-run intermediate results for the model-first path.

    ``checkpoint_path``, when given, is the exact destination (executor per-run
    mode); otherwise a timestamped filename under ``output_dir/checkpoints/`` is
    used (legacy inline behavior).
    """
    if checkpoint_path is not None:
        path = checkpoint_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
    else:
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
