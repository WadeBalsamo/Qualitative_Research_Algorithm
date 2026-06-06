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
_OVERLAY_KEY = {
    'vaamr': 'theme',
    'purer': 'purer',
    'codebook': 'codebook',
    'cross-validation': 'cv',
}


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


def reset_for_fresh(output_dir: str, what: str) -> dict:
    """Clear checkpoints + overlay for ``what`` so its classifier starts from scratch.

    Returns ``{'what', 'checkpoints_removed', 'overlay_cleared'}``.
    """
    removed = delete_checkpoints(output_dir, what)
    cleared = clear_overlay(output_dir, what)
    return {'what': what, 'checkpoints_removed': removed, 'overlay_cleared': cleared}
