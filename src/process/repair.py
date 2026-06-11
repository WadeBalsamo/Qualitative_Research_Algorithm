"""
process/repair.py
-----------------
M4 — surgical error repair for classification overlays.

``fix_errors(run_dir, config, ...)`` is the single entry point.  For each
overlays in the request it:

1. Detects error cells (``error_detection.repair_targets`` for registry overlays;
   empty-ensemble rows for codebook).
2. Skips dead raters (≥ ``dead_rater_error_fraction`` fraction of total cells are
   errors) unless ``force=True``.
3. For *registry* overlays (theme/purer): patches the per-run checkpoint
   (``reclassify_ops.patch_run_errors_only``) so only the error cells are re-run,
   then calls ``run_executor.execute_single_run`` (retries=0 inside repair — the
   pass loop IS the retry budget) to re-fetch them.
4. For *codebook*: calls ``stage_classify_codebook`` with ``only_segment_ids``
   to fill just the empty rows.
5. Repeats up to ``max_passes`` times; stops early on no-progress (error count
   not strictly decreasing between passes).
6. Scoped ``consensus_rebuild.rebuild_overlay`` for changed segments.
7. Writes flagged remainder to ``02_meta/flagged_for_review_repair.json``.

**dry_run=True**: prints detection tables only; makes NO mutations, NO LLM calls.

Auto-repair hook (called by ``run_executor.execute_queue`` and ``cmd_classify``)
reads config.auto_repair tolerantly (missing → enabled=True, max_passes=2).
"""

import datetime
import json
import os
from typing import Dict, List, Optional, Tuple

from . import output_paths as _paths
from . import run_registry as _rr
from . import error_detection as _ed
from . import reclassify_ops as _reclassify
from . import consensus_rebuild as _crebuild
from . import run_executor as _rx

# Module-level defaults (read tolerantly so config absence is safe).
# The dead-rater fraction default lives in run_executor (single-homed).
_DEFAULT_ENABLED = True
_DEFAULT_MAX_PASSES = 2

# Registry overlays that use the run executor + checkpoint patching.
_REGISTRY_OVERLAYS = frozenset({'theme', 'purer'})


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _auto_repair_enabled(config) -> bool:
    ar = getattr(config, 'auto_repair', None)
    if ar is None:
        return _DEFAULT_ENABLED
    val = getattr(ar, 'enabled', None)
    if val is None:
        return _DEFAULT_ENABLED
    return bool(val)


def _auto_repair_max_passes(config) -> int:
    ar = getattr(config, 'auto_repair', None)
    val = getattr(ar, 'max_passes', None) if ar is not None else None
    try:
        return int(val) if val is not None else _DEFAULT_MAX_PASSES
    except (TypeError, ValueError):
        return _DEFAULT_MAX_PASSES


def _dead_rater_fraction(config) -> float:
    # Single-homed in run_executor (M4 config knob: auto_repair.dead_rater_error_fraction).
    return _rx._dead_rater_fraction(config)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fix_errors(
    run_dir: str,
    config,
    *,
    overlays: Tuple[str, ...] = ('theme', 'purer', 'codebook'),
    run_ids: Optional[List[int]] = None,
    max_passes: int = _DEFAULT_MAX_PASSES,
    dry_run: bool = False,
    force: bool = False,
    observer=None,
) -> dict:
    """Detect and repair classification errors in ``overlays``.

    Parameters
    ----------
    run_dir : str
        Project output directory (holds ``qra.db`` + checkpoints).
    config : PipelineConfig
        Pipeline config (read tolerantly: auto_repair sub-config, dead_rater_error_fraction).
    overlays : tuple of str
        ``'theme'``, ``'purer'``, and/or ``'codebook'`` — which overlays to repair.
    run_ids : list of int, optional
        Only repair these specific runs (all eligible non-archived/failed when None).
    max_passes : int
        Maximum repair iterations per run (bounded retry budget).
    dry_run : bool
        When True, print detection tables and return results without any mutation.
    force : bool
        Bypass the dead-rater skip (repair even a run where ≥50% cells are errors).
    observer : optional
        Unused (reserved for future TUI progress notifications).

    Returns
    -------
    dict
        ``{'overlays': {overlay: {'repaired': n, 'remaining': n,
           'flagged': [...], 'passes': n}}, 'dry_run': bool}``
    """
    dead_frac = _dead_rater_fraction(config)
    result: Dict[str, dict] = {}

    for overlay in overlays:
        if overlay in _REGISTRY_OVERLAYS:
            ov_result = _repair_registry_overlay(
                run_dir, config, overlay,
                run_ids=run_ids, max_passes=max_passes,
                dry_run=dry_run, force=force, dead_frac=dead_frac,
            )
        elif overlay == 'codebook':
            ov_result = _repair_codebook_overlay(
                run_dir, config,
                max_passes=max_passes, dry_run=dry_run,
            )
        else:
            print(f"  [repair] overlay {overlay!r} not supported — skipping.")
            continue
        result[overlay] = ov_result

    if not dry_run and result:
        _write_flagged_remainder(run_dir, result)

    return {'overlays': result, 'dry_run': dry_run}


# ---------------------------------------------------------------------------
# Registry overlay repair (theme / purer)
# ---------------------------------------------------------------------------

def _repair_registry_overlay(
    run_dir, config, overlay, *,
    run_ids, max_passes, dry_run, force, dead_frac,
) -> dict:
    """Repair ERROR ballots in a registry overlay by re-sweeping the bad cells."""
    targets = _ed.repair_targets(run_dir, overlay, run_ids)

    if dry_run:
        summary = _ed.detect_overlay_errors(run_dir, overlay)
        print(f"\n  [dry-run {overlay}] overlay error summary: "
              f"{summary['summary']}")
        per_run_counts = {rid: len(segs) for rid, segs in targets.items()}
        print(f"  [dry-run {overlay}] per-run ERROR cells: {per_run_counts}")
        return {
            'repaired': 0,
            'remaining': sum(len(s) for s in targets.values()),
            'flagged': [],
            'passes': 0,
        }

    total_repaired = 0
    flagged: List[dict] = []
    passes_used = 0
    repair_sweep_failed: List[int] = []

    for run_id, error_seg_ids in list(targets.items()):
        run = _rr.get_run(run_dir, run_id)
        if run is None:
            print(f"  [repair {overlay}] run {run_id} not found — skipping.")
            continue

        # Status BEFORE repair: if a repair sweep fails (execute_single_run sets
        # status='failed', retries=0), a run that was completed/with-errors and
        # still carries valid ballots must NOT be demoted to 'failed' — that would
        # drop it from selection eligibility and silently remove its good votes
        # from consensus.  We restore it below.
        prior_status = run.get('status')

        # ---- Dead-rater guard ----
        n_total = run.get('n_total') or 0
        n_initial_errors = len(error_seg_ids)
        if n_total > 0 and (n_initial_errors / n_total) >= dead_frac and not force:
            print(
                f"  [repair {overlay}] run {run_id} ({run.get('rater_label')!r}): "
                f"{n_initial_errors}/{n_total} cells are errors "
                f"(≥{int(dead_frac * 100)}% dead-rater threshold). "
                f"Skipping — consider swapping the model with "
                f"`qra runs archive --run-id {run_id}` + `qra runs queue --model NEW`. "
                f"Pass --force to repair anyway."
            )
            flagged.append({
                'overlay': overlay, 'run_id': run_id,
                'rater_label': run.get('rater_label'),
                'reason': 'dead_rater_skipped',
                'n_errors': n_initial_errors,
                'n_total': n_total,
                'segment_ids': error_seg_ids,
            })
            continue

        # ---- Per-run checkpoint path ----
        checkpoint_path = _rx._per_run_checkpoint_path(run_dir, overlay, run_id)
        if not os.path.isfile(checkpoint_path):
            print(
                f"  [repair {overlay}] run {run_id}: per-run checkpoint not found "
                f"({os.path.basename(checkpoint_path)}). Cannot re-fetch ballots; "
                f"skipping this run."
            )
            flagged.append({
                'overlay': overlay, 'run_id': run_id,
                'rater_label': run.get('rater_label'),
                'reason': 'missing_checkpoint',
                'segment_ids': error_seg_ids,
            })
            continue

        # ---- Pass loop (≤ max_passes) ----
        current_errors = list(error_seg_ids)
        run_repaired = 0
        run_passes = 0
        prev_count = len(current_errors)

        for pass_n in range(1, max_passes + 1):
            run_passes = pass_n
            passes_used = max(passes_used, pass_n)

            print(
                f"  [repair {overlay}] run {run_id} ({run.get('rater_label')!r}) "
                f"pass {pass_n}/{max_passes}: clearing {len(current_errors)} error cell(s)..."
            )

            # Patch the per-run checkpoint to remove only the error cells.
            try:
                patch_result = _reclassify.patch_run_errors_only(
                    checkpoint_path, 0,  # run_idx=0 for per-run n_runs=1 checkpoints
                    segment_ids=set(current_errors),
                )
                print(
                    f"    cleared {patch_result['cleared_errors']} error(s), "
                    f"preserved {patch_result['preserved']} valid ballot(s)."
                )
            except Exception as e:
                print(f"    checkpoint patch failed: {e} — aborting this run's repair.")
                break

            # Re-sweep the run (retries=0 inside repair; the pass loop IS the retry).
            run_row = dict(run)
            run_row['__run_dir'] = run_dir
            try:
                status = _rx.execute_single_run(
                    run_dir, config, run_row, retries=0, force=True,
                )
            except KeyboardInterrupt:
                print(f"    interrupted during repair of run {run_id}.")
                raise
            except Exception as e:
                print(f"    execute_single_run failed: {e} — checking remaining errors.")

            # Recount error cells.
            new_errors = _ed.detect_run_error_cells(run_dir, run_id)
            n_fixed = prev_count - len(new_errors)
            run_repaired += max(0, n_fixed)
            print(
                f"    after pass {pass_n}: {len(new_errors)} error(s) remain "
                f"({n_fixed} fixed)."
            )

            if not new_errors:
                current_errors = []
                break

            # No-progress break: stop if error count didn't strictly decrease.
            if len(new_errors) >= prev_count:
                print(
                    f"    no progress (count {prev_count} → {len(new_errors)}); "
                    f"stopping repair for run {run_id}."
                )
                current_errors = new_errors
                break

            prev_count = len(new_errors)
            current_errors = new_errors

        total_repaired += run_repaired

        if current_errors:
            flagged.append({
                'overlay': overlay, 'run_id': run_id,
                'rater_label': run.get('rater_label'),
                'reason': 'passes_exhausted',
                'n_errors_remaining': len(current_errors),
                'segment_ids': current_errors,
            })

        # ---- Restore a run a failed repair sweep wrongly demoted to 'failed' ----
        # If the run was completed (or completed_with_errors) before repair and
        # still carries valid (non-error) ballots, an execute_single_run failure
        # during repair must not strand it as 'failed' — that would drop it from
        # `qra runs select` eligibility and silently pull its good votes out.
        after = _rr.get_run(run_dir, run_id) or {}
        if (after.get('status') == _rx.STATUS_FAILED
                and prior_status in (_rx.STATUS_COMPLETED, _rx.STATUS_COMPLETED_WITH_ERRORS)
                and ((after.get('n_coded') or 0) + (after.get('n_abstain') or 0)) > 0):
            _rr.update_run(run_dir, run_id, status=prior_status)
            repair_sweep_failed.append(run_id)
            print(
                f"  [repair {overlay}] run {run_id} ({run.get('rater_label')!r}): "
                f"repair sweep failed, but it still has valid ballots — restoring "
                f"status {prior_status!r} (NOT demoting to 'failed' / dropping it "
                f"from consensus)."
            )

    # ---- Scoped consensus rebuild for any changed segments ----
    all_changed_seg_ids: set = set()
    for rid, segs in targets.items():
        all_changed_seg_ids.update(segs)
    if all_changed_seg_ids and not dry_run:
        try:
            _crebuild.rebuild_overlay(
                run_dir, overlay, config,
                only_segment_ids=all_changed_seg_ids,
            )
        except Exception as e:
            print(f"  [repair {overlay}] scoped rebuild failed: {e}")

    n_remaining = sum(f.get('n_errors_remaining', len(f.get('segment_ids', [])))
                      for f in flagged if f.get('reason') == 'passes_exhausted')

    result = {
        'repaired': total_repaired,
        'remaining': n_remaining,
        'flagged': flagged,
        'passes': passes_used,
    }
    if repair_sweep_failed:
        result['repair_sweep_failed'] = True
        result['repair_sweep_failed_run_ids'] = repair_sweep_failed
    return result


# ---------------------------------------------------------------------------
# Codebook overlay repair
# ---------------------------------------------------------------------------

def _repair_codebook_overlay(run_dir, config, *, max_passes, dry_run) -> dict:
    """Re-classify participant segments with empty ensemble labels."""
    errors = _ed.detect_overlay_errors(run_dir, 'codebook')
    empty_ids = {sid for sid, status in errors['by_segment'].items()
                 if status == 'repairable_error'}

    if dry_run:
        print(f"\n  [dry-run codebook] overlay error summary: "
              f"{errors['summary']}")
        print(f"  [dry-run codebook] empty-ensemble segments: {len(empty_ids)}")
        return {
            'repaired': 0,
            'remaining': len(empty_ids),
            'flagged': [],
            'passes': 0,
        }

    if not empty_ids:
        return {'repaired': 0, 'remaining': 0, 'flagged': [], 'passes': 0}

    print(f"  [repair codebook] {len(empty_ids)} empty-ensemble segment(s) to re-classify.")

    codebook = _load_codebook(config)
    if codebook is None:
        print("  [repair codebook] could not load codebook — skipping.")
        return {
            'repaired': 0,
            'remaining': len(empty_ids),
            'flagged': [{'overlay': 'codebook', 'reason': 'codebook_load_failed',
                         'segment_ids': list(empty_ids)}],
            'passes': 0,
        }

    prev_count = len(empty_ids)
    repaired = 0
    passes_used = 0
    current_empty = set(empty_ids)

    for pass_n in range(1, max_passes + 1):
        passes_used = pass_n
        print(f"  [repair codebook] pass {pass_n}/{max_passes}: "
              f"re-classifying {len(current_empty)} segment(s)...")
        try:
            from .orchestrator import stage_classify_codebook
            stage_classify_codebook(
                config, codebook,
                output_dir=run_dir,
                only_segment_ids=current_empty,
            )
        except Exception as e:
            print(f"    stage_classify_codebook failed: {e}")
            break

        # Recount empty segments.
        new_errors = _ed.detect_overlay_errors(run_dir, 'codebook')
        new_empty = {sid for sid, status in new_errors['by_segment'].items()
                     if status == 'repairable_error'}
        n_fixed = prev_count - len(new_empty)
        repaired += max(0, n_fixed)
        print(f"    after pass {pass_n}: {len(new_empty)} empty remain ({n_fixed} fixed).")

        if not new_empty:
            current_empty = set()
            break

        if len(new_empty) >= prev_count:
            print(f"    no progress; stopping codebook repair.")
            current_empty = new_empty
            break

        prev_count = len(new_empty)
        current_empty = new_empty

    flagged: List[dict] = []
    if current_empty:
        flagged.append({
            'overlay': 'codebook',
            'reason': 'passes_exhausted',
            'n_errors_remaining': len(current_empty),
            'segment_ids': list(current_empty),
        })

    return {
        'repaired': repaired,
        'remaining': len(current_empty),
        'flagged': flagged,
        'passes': passes_used,
    }


def _load_codebook(config):
    """Load the VCE phenomenology codebook from config or the default preset."""
    try:
        codebook_preset = getattr(config, 'codebook', None)
        from constructs.codebook.phenomenology_codebook import get_phenomenology_codebook
        return get_phenomenology_codebook()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Flagged-remainder persistence
# ---------------------------------------------------------------------------

def _flagged_path(run_dir: str) -> str:
    """Absolute path to the flagged-for-review repair JSON."""
    return os.path.join(_paths.meta_dir(run_dir), 'flagged_for_review_repair.json')


def _write_flagged_remainder(run_dir: str, ov_results: dict) -> None:
    """Merge-update the flagged_for_review_repair.json with newly-flagged items.

    The file is keyed by ``"{overlay}/{run_id}"`` (or ``"{overlay}/codebook"`` for
    the codebook path) so successive repair runs accumulate rather than overwrite.
    If nothing was flagged across all overlays, the function does nothing (avoids
    creating an empty file).
    """
    any_flagged = any(
        ov_results[ov].get('flagged') for ov in ov_results
    )
    if not any_flagged:
        return

    flagged_path = _flagged_path(run_dir)
    existing: dict = {}
    if os.path.isfile(flagged_path):
        try:
            with open(flagged_path, 'r') as f:
                existing = json.load(f)
        except (OSError, json.JSONDecodeError):
            existing = {}

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for overlay, ov_data in ov_results.items():
        for entry in ov_data.get('flagged', []):
            run_id = entry.get('run_id', 'codebook')
            key = f"{overlay}/{run_id}"
            existing[key] = dict(entry)
            existing[key]['overlay'] = overlay
            existing[key]['flagged_at'] = now

    os.makedirs(os.path.dirname(flagged_path), exist_ok=True)
    with open(flagged_path, 'w') as f:
        json.dump(existing, f, indent=2, default=str)

    n_flagged = sum(len(v.get('flagged', [])) for v in ov_results.values())
    print(
        f"\n  [repair] {n_flagged} run(s)/overlay(s) remain with unresolved errors "
        f"→ {flagged_path}"
    )
    for overlay, ov_data in ov_results.items():
        for entry in ov_data.get('flagged', []):
            print(
                f"    {overlay} / run {entry.get('run_id', '?')} "
                f"({entry.get('rater_label', '?')}): "
                f"reason={entry.get('reason')} "
                f"n_errors={entry.get('n_errors_remaining', len(entry.get('segment_ids', [])))}"
            )


# ---------------------------------------------------------------------------
# Auto-repair hook (called by execute_queue + cmd_classify)
# ---------------------------------------------------------------------------

def maybe_auto_repair(run_dir: str, config, overlays: tuple) -> Optional[dict]:
    """Auto-repair hook for ``execute_queue`` and ``cmd_classify``.

    Reads config.auto_repair tolerantly.  Returns None when auto-repair is
    disabled; returns the ``fix_errors`` result dict otherwise.  Never raises
    (failures are printed but swallowed so the queue does not abort).
    """
    if not _auto_repair_enabled(config):
        return None
    mp = _auto_repair_max_passes(config)
    try:
        result = fix_errors(
            run_dir, config,
            overlays=tuple(ov for ov in overlays if ov in ('theme', 'purer', 'codebook')),
            max_passes=mp,
            dry_run=False,
            force=False,
        )
        _print_auto_repair_summary(result)
        return result
    except KeyboardInterrupt:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"  [auto-repair] unexpected error: {e} — skipping and continuing.")
        return None


def _print_auto_repair_summary(result: dict) -> None:
    """One-line summary printed by the auto-repair hook."""
    parts = []
    for overlay, ov in result.get('overlays', {}).items():
        repaired = ov.get('repaired', 0)
        remaining = ov.get('remaining', 0)
        parts.append(f"{overlay}: repaired={repaired} remaining={remaining}")
    if parts:
        print(f"  [auto-repair] {', '.join(parts)}")
