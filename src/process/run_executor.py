"""
process/run_executor.py
-----------------------
Run queue executor (schema-v2 registry).  Productizes the hand-rolled shell
watchdog (``run_purer_gemma_watchdog.sh``): a single-flock-guarded loop that
processes every queued / stale-running run for the requested overlays, resuming
each from its own per-run checkpoint, honoring a STOP sentinel + Ctrl-C, retrying
failures cell-wise, and flushing durable ballots mid-sweep.

State machine (one run)::

    queued ──start──▶ running ──{ok, 0 err}──────────▶ completed
                          │   └─{ok, >0 err}──────────▶ completed_with_errors
                          │   └─{fail, attempt<retries}▶ running  (cell-wise resume)
                          │   └─{retries exhausted}─────▶ failed
                          │   └─{dead-rater ≥50% err}───▶ failed   (no retry storm)
                          │   └─{model-mismatch preflight}▶ queued (skipped, actionable msg)
                          └─{Ctrl-C | STOP}─────────────▶ running  (checkpoint saved)

This module is a LIBRARY: it must NOT import ``qra.py`` (the CLI chains the
downstream assemble→testset→analyze itself).  It writes ballots directly through
``run_registry`` (the inline classify paths use ``orchestrator._persist_ballots_
from_results`` instead); consensus is rebuilt at queue end via
``consensus_rebuild.rebuild_overlay``.

The executor sweeps via the SAME entry points the inline path uses
(``classify_segments_zero_shot`` for theme; ``classify_purer_cue_units`` for
turn-mode PURER) so prompts / parsing / checkpoint format are byte-compatible —
the M3 equivalence gate (``tests/unit/test_run_executor.py``) asserts a queued
3-run sweep reproduces an inline 3-model classify record-for-record.
"""

import contextlib
import copy
import fcntl
import os
from typing import Callable, Dict, List, Optional

from . import output_paths as _paths
from . import run_registry as _rr
from . import consensus_rebuild as _crebuild
from . import segments_io
from . import reclassify_ops as _reclassify


# Overlay -> classification sub-config attribute. The checkpoint file prefix is
# single-homed in ``reclassify_ops`` (see ``overlay_prefix`` below).
_OVERLAY_SUBCFG = {'theme': 'theme_classification', 'purer': 'purer_classification'}

# A run is "dead" (skip the retry storm) when at least this fraction of its
# attempted cells are hard parse errors after the first full sweep.  Overridable
# via ``config.auto_repair.dead_rater_error_fraction`` (M4) when present.
DEFAULT_DEAD_RATER_ERROR_FRACTION = 0.5

# Statuses an executor pass will (re)process: fresh queue + resumable stale runs.
_RESUMABLE_STATUSES = ('queued', 'running')

# Terminal/in-progress run statuses (informational).
STATUS_COMPLETED = 'completed'
STATUS_COMPLETED_WITH_ERRORS = 'completed_with_errors'
STATUS_FAILED = 'failed'
STATUS_RUNNING = 'running'
STATUS_QUEUED = 'queued'


class RunnerBusy(RuntimeError):
    """Raised when another executor already holds the per-project runner lock."""


# ---------------------------------------------------------------------------
# Lock + STOP sentinel
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def acquire_runner_lock(run_dir: str):
    """Hold an exclusive, non-blocking ``flock`` on ``<run_dir>/.qra_runs.lock``.

    Prevents two executors from sweeping the same project concurrently (the WAL
    handles row-level writes; this guards the queue-processing loop).  Raises
    :class:`RunnerBusy` immediately if the lock is already held.
    """
    os.makedirs(run_dir, exist_ok=True)
    lock_path = os.path.join(run_dir, '.qra_runs.lock')
    fh = open(lock_path, 'w')
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError):
            raise RunnerBusy(
                f"another `qra runs` executor is already running for {run_dir} "
                f"(lock held: {lock_path}). Wait for it to finish, or stop it with "
                f"`touch {os.path.join(run_dir, 'STOP_QRA_RUNS')}`."
            )
        yield fh
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except (OSError, IOError):
            pass
        fh.close()


def _stop_requested(run_dir: str) -> bool:
    """True when the ``<run_dir>/STOP_QRA_RUNS`` sentinel file exists."""
    return os.path.exists(os.path.join(run_dir, 'STOP_QRA_RUNS'))


# ---------------------------------------------------------------------------
# Per-run config + checkpoint helpers
# ---------------------------------------------------------------------------

def _per_run_checkpoint_path(run_dir: str, overlay: str, run_id: int) -> str:
    """Stable per-run checkpoint: ``{prefix}_run{run_id:04d}_runs.json``.

    A dedicated path per run means exact cell-wise resume with no glob races
    (the shell watchdog's re-glob-latest hazard) — ``model_first_v1`` with
    n_runs=1, so ``patch_run_errors_only`` works unchanged with run_idx=0.
    """
    ckpt_dir = _paths.llm_checkpoints_dir(run_dir)
    return os.path.join(ckpt_dir, f'{overlay_prefix(overlay)}_run{run_id:04d}_runs.json')


def overlay_prefix(overlay: str) -> str:
    """checkpoint file_prefix for an overlay's per-run sweep (single-homed in
    ``reclassify_ops.checkpoint_prefix``)."""
    if overlay not in _OVERLAY_SUBCFG:
        raise ValueError(f"run_executor: unsupported overlay {overlay!r}")
    return _reclassify.checkpoint_prefix(overlay)


def _build_per_run_config(config, overlay: str, run_row: dict, *, checkpoint_path: str):
    """Return a deep-copied PipelineConfig whose overlay sub-config is pinned to
    this single run (model, no_reasoning, temperature, n_runs=1, per_run_models,
    output_dir=auditable_logs, resume_from=checkpoint when present).

    For PURER the VAAMR-inheritance is resolved first (so a bare PURER config
    still gets a backend/url), then the run's own model overrides it.
    """
    cfg = copy.deepcopy(config)
    run_dir_logs = _paths.auditable_logs_dir(run_row['__run_dir'])

    if overlay == 'purer':
        # Resolve inheritance into cfg.purer_classification (mutates the COPY).
        from .orchestrator import _resolve_purer_framework_and_config
        _resolve_purer_framework_and_config(cfg, run_row['__run_dir'])

    sub_attr = _OVERLAY_SUBCFG[overlay]
    sub = getattr(cfg, sub_attr)
    sub.model = run_row['model']
    sub.per_run_models = [run_row['model']]
    sub.n_runs = 1
    # thinking: 'off' -> True, 'on' -> False, None/'' -> leave inherited from base.
    _thinking = run_row.get('thinking')
    if _thinking == 'off':
        sub.no_reasoning = True
    elif _thinking == 'on':
        sub.no_reasoning = False
    if run_row.get('temperature') is not None:
        sub.temperature = run_row['temperature']
    if run_row.get('backend') not in (None, ''):
        sub.backend = run_row['backend']
    # Honor the run-execution save_interval knob (M6) when configured.
    re_cfg = getattr(config, 'run_execution', None)
    si = getattr(re_cfg, 'save_interval', None) if re_cfg is not None else None
    if si:
        sub.save_interval = si
    sub.output_dir = run_dir_logs
    # The sweep resumes via the explicit checkpoint_path arg (deterministic
    # per-run file), not cfg.resume_from — kept None so nothing double-resumes.
    cfg.resume_from = None
    return cfg, sub


def _dead_rater_fraction(config) -> float:
    """Read the dead-rater error-fraction threshold tolerantly (M4 config)."""
    ar = getattr(config, 'auto_repair', None)
    val = getattr(ar, 'dead_rater_error_fraction', None) if ar is not None else None
    try:
        return float(val) if val is not None else DEFAULT_DEAD_RATER_ERROR_FRACTION
    except (TypeError, ValueError):
        return DEFAULT_DEAD_RATER_ERROR_FRACTION


# ---------------------------------------------------------------------------
# Single-run sweep
# ---------------------------------------------------------------------------

def execute_single_run(
    run_dir: str,
    config,
    run_row: dict,
    *,
    retries: int = 2,
    observer=None,
    force: bool = False,
) -> str:
    """Sweep one run end-to-end and return its final status string.

    Parameters
    ----------
    run_dir : str
        Project output directory (holds ``qra.db`` + checkpoints).
    config : PipelineConfig
        Base config; the run's model/thinking/temperature override the overlay
        sub-config for this sweep only.
    run_row : dict
        A ``run_registry`` row (must carry ``run_id``, ``overlay``, ``model``).
    retries : int
        Max retry attempts on an exception (each resumes cell-wise from the
        per-run checkpoint).
    force : bool
        Bypass the LM Studio model-mismatch pre-flight.

    Returns one of ``completed`` / ``completed_with_errors`` / ``failed`` /
    ``queued`` (the last = pre-flight skip; the run is left queued).
    """
    overlay = run_row['overlay']
    run_id = run_row['run_id']

    if overlay not in _OVERLAY_SUBCFG:
        _rr.update_run(run_dir, run_id, status=STATUS_FAILED)
        print(f"  [run {run_id}] FAILED: overlay {overlay!r} is not executor-supported "
              f"(theme/purer only).")
        return STATUS_FAILED

    # PURER cue_block guard: registry v1 supports turn mode only (cue_block's
    # consensus-driven bisection creates run-dependent synthetic unit ids).
    if overlay == 'purer':
        unit = getattr(getattr(config, 'purer_cue', None), 'classification_unit', 'turn')
        if unit != 'turn':
            _rr.update_run(run_dir, run_id, status=STATUS_FAILED)
            print(f"  [run {run_id}] FAILED: PURER runs require "
                  f"purer_cue.classification_unit='turn' (got {unit!r}); cue_block mode "
                  f"stays on the legacy `qra classify --what purer` inline path.")
            return STATUS_FAILED

    run_row = dict(run_row)
    run_row['__run_dir'] = run_dir
    checkpoint_path = _per_run_checkpoint_path(run_dir, overlay, run_id)
    cfg, sub = _build_per_run_config(config, overlay, run_row, checkpoint_path=checkpoint_path)

    # ---- Pre-flight (lmstudio): require the model loaded unless --force ----
    if not force and sub.backend == 'lmstudio':
        from classification_tools.llm_client import LLMClient, LLMClientConfig
        probe = LLMClient(LLMClientConfig(
            backend='lmstudio', model=run_row['model'],
            lmstudio_base_url=getattr(sub, 'lmstudio_base_url', 'http://127.0.0.1:1234/v1'),
        ))
        if not probe.check_loaded_model(run_row['model']):
            print(
                f"\n  [run {run_id}] SKIPPED — model not loaded in LM Studio.\n"
                f"    Requested : {run_row['model']}\n"
                f"    Load it in LM Studio, then re-run `qra runs start` "
                f"(or pass --force to proceed with whatever is loaded).\n"
            )
            return STATUS_QUEUED

    # ---- Stamp running + checkpoint path before sweeping ----
    os.makedirs(_paths.llm_checkpoints_dir(run_dir), exist_ok=True)
    _rr.update_run(run_dir, run_id, status=STATUS_RUNNING,
                   checkpoint_path=checkpoint_path, started_at=_rr._now_iso())

    # ---- Ballot flush closure (short transaction per save) ----
    # ``applies_to`` is set by the PURER sweep builder (cue-unit -> constituent
    # therapist ids); theme units are 1:1 with their segment so it stays None.
    _applies_to_box = {'map': None}

    def _flush(cells: Dict[str, Optional[dict]]):
        if not cells:
            return
        _rr.upsert_ballots(run_dir, overlay, run_id, cells,
                           applies_to=(_applies_to_box['map'] or None))

    # ---- Build the sweep callable (theme vs purer) ----
    sweep, applies_to_built = _build_sweep(cfg, sub, run_dir, overlay, checkpoint_path, _flush)
    if applies_to_built is not None:
        _applies_to_box['map'] = applies_to_built

    attempt = 0
    last_error: Optional[Exception] = None
    while True:
        try:
            sweep()
            break
        except KeyboardInterrupt:
            # Save/flush already happened in classify_segments' finally + the
            # on_progress flushes; leave status 'running' (resumable) and re-raise.
            _rr.refresh_counters(run_dir, run_id)
            print(f"\n  [run {run_id}] interrupted — checkpoint saved. Resume with "
                  f"`qra runs start` (it continues this run cell-wise).")
            raise
        except Exception as e:  # noqa: BLE001
            last_error = e
            _rr.refresh_counters(run_dir, run_id)
            # Dead-rater guard on EVERY caught exception: if the run is mostly
            # parse errors, fail fast (no retry storm — the watchdog's pathology).
            if _is_dead_rater(run_dir, run_id, _dead_rater_fraction(config)):
                _rr.update_run(run_dir, run_id, status=STATUS_FAILED,
                               completed_at=_rr._now_iso())
                print(f"  [run {run_id}] FAILED (dead rater: ≥"
                      f"{int(_dead_rater_fraction(config) * 100)}% parse errors). "
                      f"Not retrying. Last error: {e}")
                return STATUS_FAILED
            if attempt >= retries:
                _rr.update_run(run_dir, run_id, status=STATUS_FAILED,
                               completed_at=_rr._now_iso())
                print(f"  [run {run_id}] FAILED after {attempt + 1} attempt(s): {e}")
                return STATUS_FAILED
            attempt += 1
            print(f"  [run {run_id}] error (attempt {attempt}/{retries}): {e} — "
                  f"resuming from checkpoint.")
            # The sweep already reads checkpoint_path on every invocation (the
            # model-first path resumes the completed cells), so the retry is
            # cell-wise with no extra wiring.

    # ---- Sweep finished cleanly: final flush + counters + status ----
    _rr.refresh_counters(run_dir, run_id)
    run_after = _rr.get_run(run_dir, run_id) or {}
    n_error = run_after.get('n_error') or 0
    status = STATUS_COMPLETED_WITH_ERRORS if n_error > 0 else STATUS_COMPLETED
    _rr.update_run(run_dir, run_id, status=status, completed_at=_rr._now_iso())
    print(f"  [run {run_id}] {status}: "
          f"coded {run_after.get('n_coded') or 0}  abstain {run_after.get('n_abstain') or 0}  "
          f"error {n_error}  (total {run_after.get('n_total') or 0})")
    # A run that completed but is mostly errors is NOT failed (its valid ballots
    # still count) — but make the dead rater impossible to miss.
    if status == STATUS_COMPLETED_WITH_ERRORS and _is_dead_rater(
            run_dir, run_id, _dead_rater_fraction(config)):
        pct = int(_dead_rater_fraction(config) * 100)
        print(
            f"\n  ************************************************************\n"
            f"  *** WARNING: run {run_id} ({run_after.get('rater_label')!r}) is a DEAD RATER:\n"
            f"  ***   {n_error}/{run_after.get('n_total') or 0} cells are parse errors (≥{pct}%).\n"
            f"  ***   Its ballots still feed consensus, but this rater is degrading it.\n"
            f"  ***   Fix it: `qra fix-errors -o <dir>` to re-fetch the error cells,\n"
            f"  ***   or swap the model (`qra runs archive --run-id {run_id}` + queue a new one).\n"
            f"  ************************************************************\n"
        )
    return status


def _build_sweep(cfg, sub, run_dir, overlay, checkpoint_path, flush):
    """Return ``(sweep_callable, applies_to_map_or_None)`` for the overlay.

    The sweep, when called, runs the SAME inline entry point with n_runs=1,
    per_run_models=[model], resume_from=checkpoint_path, and on_progress=flush.
    The PURER builder also returns the cue-unit -> constituents ``applies_to``
    mapping so each ballot records its propagation targets.
    """
    from .process_logger import ProcessLogger
    plog = ProcessLogger(None, llm_log_path=_paths.llm_prompts_path(run_dir))

    if overlay == 'theme':
        from constructs.registry import load as _load_fw
        from classification_tools.theme_llm.llm_classifier import classify_segments_zero_shot
        from .orchestrator import _load_classify_targets
        framework = _load_fw(getattr(cfg, 'participant_framework', 'vaamr') or 'vaamr')
        targets = _load_classify_targets(cfg, run_dir, 'theme')

        def sweep():
            try:
                classify_segments_zero_shot(
                    segments=targets, framework=framework, config=sub,
                    process_logger=plog, on_progress=flush,
                    checkpoint_path=checkpoint_path,
                )
            finally:
                plog.close_llm_log()
        return sweep, None

    # ---- PURER turn-mode ----
    from classification_tools.theme_llm.llm_classifier import classify_purer_cue_units
    from .orchestrator import build_purer_turn_cue_units
    segments = _load_classify_targets_purer(cfg, run_dir)
    cue_units, purer_framework, _ = build_purer_turn_cue_units(cfg, segments, run_dir)
    # Pin the per-run model onto the resolved purer config (build_* resolves
    # inheritance but the executor's single-run model must win).
    purer_cfg = sub
    purer_cfg.per_run_models = [purer_cfg.model]
    purer_cfg.n_runs = 1

    # cue-unit segment_id -> its constituent therapist ids (applies_to).
    applies_to_map: Dict[str, list] = {
        cu['segment'].segment_id: [c.segment_id for c in cu.get('_constituents') or []]
        for cu in cue_units
    }
    stripped = [{k: v for k, v in cu.items() if not k.startswith('_')} for cu in cue_units]

    def sweep():
        try:
            classify_purer_cue_units(
                cue_units=stripped, framework=purer_framework, config=purer_cfg,
                process_logger=plog, on_progress=flush,
                checkpoint_path=checkpoint_path,
            )
        finally:
            plog.close_llm_log()
    return sweep, applies_to_map


def _load_classify_targets_purer(cfg, run_dir):
    """Frozen segments with the theme overlay applied (turn-mode PURER anchor)."""
    from .orchestrator import _load_classify_targets
    return _load_classify_targets(cfg, run_dir, 'purer')


def _is_dead_rater(run_dir: str, run_id: int, fraction: float) -> bool:
    """True when ≥``fraction`` of the run's attempted cells are ERROR ballots."""
    run = _rr.get_run(run_dir, run_id) or {}
    total = run.get('n_total') or 0
    n_error = run.get('n_error') or 0
    return total > 0 and (n_error / total) >= fraction


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------

def _selection_is_effectively_all(run_dir: str, overlay: str) -> bool:
    """v1 selection-policy heuristic.

    Auto-select newly-completed runs only when selection currently behaves like
    'all': every non-archived completed/with-errors run is selected, OR zero runs
    are selected (a fresh project).  Otherwise the operator has curated a subset
    (top-n) and the executor must not clobber it — it prints a hint instead.
    """
    runs = _rr.list_runs(run_dir, overlay=overlay)
    eligible = [r for r in runs
                if r['status'] in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_ERRORS)]
    selected = [r for r in runs if r['selected']]
    if not selected:
        return True
    eligible_ids = {r['run_id'] for r in eligible}
    selected_ids = {r['run_id'] for r in selected}
    # 'all' iff the selection is exactly the eligible (completed) set.
    return selected_ids == eligible_ids and bool(eligible_ids)


def execute_queue(
    run_dir: str,
    config,
    *,
    overlays=('theme', 'purer'),
    retries: Optional[int] = None,
    observer=None,
    force: bool = False,
) -> dict:
    """Process every queued / stale-running run for ``overlays`` under the lock.

    For each overlay, in ``run_id`` order, sweep its resumable runs (honoring the
    STOP sentinel *between* runs).  Then, per overlay that gained or changed
    ballots: if selection is effectively 'all', auto-select the newly-completed
    runs; otherwise leave selection untouched and print a `qra runs select` hint.
    Finally rebuild each touched overlay's consensus from the selected ballots.

    Returns ``{'per_run': {run_id: status}, 'overlays_rebuilt': [...],
    'stopped_early': bool, 'skipped_queued': [run_id, ...]}``.
    """
    if retries is None:
        retries = _resolve_default_retries(config)

    summary = {'per_run': {}, 'overlays_rebuilt': [], 'stopped_early': False,
               'skipped_queued': []}

    with acquire_runner_lock(run_dir):
        touched_overlays: List[str] = []
        stop = False
        for overlay in overlays:
            if overlay not in _OVERLAY_SUBCFG:
                continue
            runs = _rr.list_runs(run_dir, overlay=overlay, statuses=list(_RESUMABLE_STATUSES))
            if not runs:
                continue
            overlay_touched = False
            for run_row in runs:
                if _stop_requested(run_dir):
                    print(f"  STOP_QRA_RUNS present — stopping before run "
                          f"{run_row['run_id']} ({overlay}). Remove the file to resume.")
                    summary['stopped_early'] = True
                    stop = True
                    break
                status = execute_single_run(
                    run_dir, config, run_row,
                    retries=retries, observer=observer, force=force,
                )
                summary['per_run'][run_row['run_id']] = status
                if status == STATUS_QUEUED:
                    summary['skipped_queued'].append(run_row['run_id'])
                else:
                    overlay_touched = True
            if overlay_touched:
                touched_overlays.append(overlay)
            if stop:
                break

        # ---- M4 Auto-repair: run BEFORE selection-aware rebuild ----
        # Repair any ERROR ballots in touched overlays so the subsequent rebuild
        # promotes the newly-repaired votes into consensus.  Late import to avoid
        # circular imports (repair → run_executor → repair).
        if touched_overlays:
            try:
                from . import repair as _repair
                _repair.maybe_auto_repair(run_dir, config, tuple(touched_overlays))
            except KeyboardInterrupt:
                raise
            except Exception as _e:  # noqa: BLE001
                print(f"  [auto-repair] hook failed: {_e} — continuing to rebuild.")

        # ---- Selection-aware rebuild for each touched overlay ----
        for overlay in touched_overlays:
            _apply_selection_then_rebuild(run_dir, config, overlay, summary)

    return summary


def _apply_selection_then_rebuild(run_dir, config, overlay, summary):
    """Auto-select (policy 'all') then rebuild one overlay's consensus."""
    if _selection_is_effectively_all(run_dir, overlay):
        runs = _rr.list_runs(run_dir, overlay=overlay)
        completed_ids = [r['run_id'] for r in runs
                         if r['status'] in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_ERRORS)]
        if completed_ids:
            _rr.set_selected(run_dir, overlay, completed_ids)
            # Persist the selection decision under the SAME manifest key
            # analysis.run_selection writes, so `qra runs` / the IRR report read a
            # consistent record (keys mirror run_selection._persist; written inline
            # to avoid a process->analysis import).
            from . import db as _db
            from . import classifications_io as _cio
            _cio.update_classification_manifest(
                run_dir, key=f'run_selection:{overlay}',
                entry={
                    'overlay': overlay,
                    'strategy': 'auto_all_completed',
                    'selected_run_ids': list(completed_ids),
                    'decided_at': _db._now_iso(),
                    'rationale': 'executor auto-select: all completed runs',
                    'fallback_used': False,
                },
            )
            print(f"  [{overlay}] selection policy 'all' → selected runs "
                  f"{completed_ids}.")
    else:
        print(f"  [{overlay}] selection is a curated subset — leaving it untouched. "
              f"Run `qra runs select --what "
              f"{'vaamr' if overlay == 'theme' else overlay}` to adjust which runs feed "
              f"consensus.")
    try:
        stats = _crebuild.rebuild_overlay(run_dir, overlay, config)
    except Exception as e:  # noqa: BLE001
        print(f"  [{overlay}] consensus rebuild failed: {e}")
        return
    if stats.get('skipped'):
        print(f"  [{overlay}] rebuild skipped: {stats.get('reason')}")
        return
    summary['overlays_rebuilt'].append(overlay)
    print(f"  [{overlay}] rebuilt {stats['n_units']} unit(s) from "
          f"{len(stats['run_ids'])} selected run(s): labeled={stats['n_labeled']} "
          f"abstain={stats['n_abstain']} unlabeled={stats['n_unlabeled']} "
          f"changed={stats['n_changed']}")


def _resolve_default_retries(config) -> int:
    """Read the default retry count tolerantly (M6 RunExecutionConfig)."""
    re_cfg = getattr(config, 'run_execution', None)
    val = getattr(re_cfg, 'retries', None) if re_cfg is not None else None
    try:
        return int(val) if val is not None else 2
    except (TypeError, ValueError):
        return 2
