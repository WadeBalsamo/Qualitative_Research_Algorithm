"""
process/reclassify_ops.py
-------------------------
"From scratch" reset helpers shared by the CLI (``qra classify --fresh``) and the
interactive TUI reclassify sub-menus.

Re-classifying a framework *from scratch* means two pieces of prior state must be
cleared before the classifier re-runs, or it would resume rather than start over:

  * the LLM run checkpoints in ``02_meta/auditable_logs/checkpoints/`` — the
    classifiers resume from these ``*_runs.json`` files; and
  * the classifier's overlay table in ``qra.db`` — so stale labels for segments
    that are no longer produced cannot linger.  (The overlay writers already do a
    full ``DELETE`` + re-insert, so this is belt-and-braces for the LLM keys and
    the only reset for the checkpoint-less ``cross-validation`` key.)

This generalises the PURER "re-run all" logic that previously lived inline in the
TUI, so VAAMR / PURER / codebook all share one implementation.

Checkpoint prefixes are the ``file_prefix`` values passed to
``classification_tools.classification_loop`` (verified in
``classification_tools/llm_classifier.py``):
``llm_results`` (VAAMR), ``purer_cue_results`` (PURER),
``codebook_llm_results`` (codebook).  ``cross-validation`` has no LLM checkpoint.
"""
import glob
import json
import os

from . import classifications_io as _cio
from . import output_paths as _paths

# `qra classify --what` value -> checkpoint filename prefix(es).
_CHECKPOINT_PREFIXES = {
    'vaamr': ('llm_results',),
    'purer': ('purer_cue_results',),
    'codebook': ('codebook_llm_results',),
    'cross-validation': (),
}

# `qra classify --what` value -> overlay key (classifications_io table key).
# Public (single-home): the ONLY place the vaamr<->theme naming split lives.
OVERLAY_FOR_WHAT = {
    'vaamr': 'theme',
    'purer': 'purer',
    'codebook': 'codebook',
    'cross-validation': 'cv',
}
# Reverse: overlay key -> `--what` value (cv has no --what equivalent → omitted).
WHAT_FOR_OVERLAY = {v: k for k, v in OVERLAY_FOR_WHAT.items() if v != 'cv'}

# Back-compat alias (kept so existing imports keep working).
_OVERLAY_KEY = OVERLAY_FOR_WHAT

# `qra classify --what` value -> registry overlay name (theme/purer have runs).
_REGISTRY_OVERLAY = {'vaamr': 'theme', 'purer': 'purer'}


def checkpoint_prefix(overlay_or_what: str) -> str:
    """The single LLM checkpoint file_prefix for an overlay key or `--what` value.

    Accepts either a registry overlay name (``'theme'`` / ``'purer'``) or a
    ``qra classify --what`` value (``'vaamr'`` / ``'purer'`` / ``'codebook'``).
    Raises ``ValueError`` for an unknown/checkpoint-less key (e.g. cross-validation).
    """
    what = WHAT_FOR_OVERLAY.get(overlay_or_what, overlay_or_what)
    prefixes = _CHECKPOINT_PREFIXES.get(what, ())
    if not prefixes:
        raise ValueError(
            f"checkpoint_prefix: no LLM checkpoint prefix for {overlay_or_what!r}")
    return prefixes[0]


def delete_checkpoints(output_dir: str, what: str) -> int:
    """Delete the LLM run checkpoints for ``what``.  Returns the number removed."""
    ckpt_dir = _paths.llm_checkpoints_dir(output_dir)
    removed = 0
    for prefix in _CHECKPOINT_PREFIXES.get(what, ()):
        for path in glob.glob(os.path.join(ckpt_dir, f'{prefix}_*')):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    return removed


def clear_overlay(output_dir: str, what: str) -> bool:
    """Clear the overlay table for ``what`` in qra.db.  Returns True if a table was cleared."""
    key = _OVERLAY_KEY.get(what)
    if key is None:
        return False
    _cio.clear_overlay(output_dir, key)
    return True


def patch_run_errors_only(
    checkpoint_path: str,
    run_idx: int,
    new_model: str | None = None,
    *,
    segment_ids=None,
) -> dict:
    """Clear only the *parse-error* (None) entries for ``run_idx`` in a model-first
    checkpoint, preserving valid ABSTAIN / CODED ballots so they are not re-run.

    This is the surgical counterpart to
    ``classification_loop.patch_runs_checkpoint`` (which wipes every entry for the
    run).  It exists for the case where a rater produced a *mix* of valid ballots
    and parse errors (e.g. PURER qwen: 437 ABSTAIN + 83 CODED + 24 ERROR): a full
    wipe would discard 520 valid votes to re-fetch 24 failures.

    For each segment, an entry is "an error" iff ``run_results[seg_id][str(run_idx)]``
    is ``None``.  Those keys are deleted; non-None ballots are left intact.  ``run_idx``
    is removed from ``completed_runs`` so the next sweep re-fills only the cleared
    (now-missing) keys, and (if given) ``per_run_models[run_idx]`` is updated.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``model_first_v1`` JSON checkpoint.
    run_idx : int
        0-indexed run slot to patch (per-run checkpoints always use 0).
    new_model : str or None
        When given, update ``per_run_models[run_idx]`` to this model string.
    segment_ids : set or None
        When given, clear null cells **only** for segments whose id is in this set.
        When None (default), clear all null cells for ``run_idx`` (original behavior).

    Returns
    -------
    dict
        ``{'cleared_errors', 'preserved', 'per_run_models',
        'cleared_segment_ids'}`` — ``cleared_segment_ids`` is the list of
        segment ids whose error entry was actually removed.
    Raises ``ValueError`` if the file is not ``model_first_v1`` or ``run_idx`` is
    out of range.
    """
    with open(checkpoint_path, 'r') as f:
        data = json.load(f)

    meta = data.get('_meta', {})
    if meta.get('format') != 'model_first_v1':
        raise ValueError(
            f"{os.path.basename(checkpoint_path)} is not a model_first_v1 checkpoint "
            f"(format={meta.get('format')!r}). Only model-first runs checkpoints can be patched."
        )

    per_run_models = meta.get('per_run_models', [])
    n_runs = meta.get('n_runs', len(per_run_models))
    if not (0 <= run_idx < n_runs):
        raise ValueError(
            f"run_idx {run_idx} is out of range for a {n_runs}-run checkpoint "
            f"(valid: 0–{n_runs - 1})."
        )

    if new_model is not None:
        if run_idx < len(per_run_models):
            per_run_models[run_idx] = new_model
        meta['per_run_models'] = per_run_models

    completed_runs = meta.get('completed_runs', [])
    meta['completed_runs'] = [r for r in completed_runs if r != run_idx]

    run_key = str(run_idx)
    cleared = preserved = 0
    cleared_segment_ids: list = []
    for seg_id, seg_data in data.get('run_results', {}).items():
        if run_key not in seg_data:
            continue
        if seg_data[run_key] is None:
            # Apply segment_ids filter when provided.
            if segment_ids is not None and seg_id not in segment_ids:
                # Error cell but not in our target set — leave it as-is.
                preserved += 1
                continue
            del seg_data[run_key]
            cleared += 1
            cleared_segment_ids.append(seg_id)
        else:
            preserved += 1

    with open(checkpoint_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    return {
        'cleared_errors': cleared,
        'preserved': preserved,
        'per_run_models': per_run_models,
        'cleared_segment_ids': cleared_segment_ids,
    }


def archive_runs(output_dir: str, what: str) -> int:
    """Archive (status='archived', selected=0) every registry run for ``what``.

    Used by ``--fresh`` so a from-scratch classify starts a brand-new run lineage
    instead of resuming the old runs.  Their durable ballots stay in the DB (for
    κ-history) but the runs are terminal/archived and excluded from selection and
    from the executor's resumable set.  Theme/PURER only (the only overlays with
    a run registry); a no-op (returns 0) for codebook / cross-validation.
    """
    overlay = _REGISTRY_OVERLAY.get(what)
    if overlay is None:
        return 0
    from . import db
    if not db.db_exists(output_dir):
        return 0
    # One UPDATE for every not-already-archived run in this overlay (a row counts
    # as "to archive" if it isn't yet status='archived' OR is still selected).
    with db.open_db(output_dir) as conn:
        cur = conn.execute(
            "UPDATE classification_runs SET status = 'archived', selected = 0 "
            "WHERE overlay = ? AND NOT (status = 'archived' AND selected = 0)",
            (overlay,),
        )
        return cur.rowcount


def reset_for_fresh(output_dir: str, what: str) -> dict:
    """Clear checkpoints + overlay for ``what`` so its classifier starts from scratch.

    Also archives the overlay's registry runs (theme/purer) and deletes their
    per-run checkpoints (already covered by the ``{prefix}_*`` glob in
    ``delete_checkpoints``) so a ``--fresh`` classify opens a new run lineage
    rather than resuming the prior runs.

    Returns ``{'what', 'checkpoints_removed', 'overlay_cleared', 'runs_archived'}``.
    """
    removed = delete_checkpoints(output_dir, what)
    cleared = clear_overlay(output_dir, what)
    archived = archive_runs(output_dir, what)
    return {'what': what, 'checkpoints_removed': removed,
            'overlay_cleared': cleared, 'runs_archived': archived}
