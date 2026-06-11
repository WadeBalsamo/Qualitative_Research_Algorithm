"""
analysis/run_selection.py
-------------------------
Per-run Cohen κ vs the human consensus, and the IRR-gated *selection* policy
that decides which registry runs feed an overlay's consensus (schema v2; see
``process.run_registry`` / ``process.consensus_rebuild``).

Two responsibilities:

  1. :func:`per_run_kappa` — for every non-archived run of an overlay, score its
     durable ``label_ballots`` against the imported human consensus codes
     (``irr_human_codes`` / ``irr_testsets``), reusing the EXACT consensus-row
     filter + ABSTAIN/ERROR handling that ``analysis.irr_analysis`` applies to
     the operative consensus.  This is the κ table the IRR report renders and the
     ranking signal the selection policy ranks on.

  2. :func:`select_runs` — apply the overlay's selection policy (``RunSelectionConfig``
     arrives in M6; until then read off the config getattr-tolerantly with the
     documented defaults — theme: top-3 by human IRR, purer: all).  Eligible runs
     are completed/completed_with_errors with a matching segmentation_params_hash.
     The decision (selected/rejected ids, κ snapshot, human band, rationale,
     fallback flag) is recorded under the manifest key ``run_selection:<overlay>``
     and ``run_registry.set_selected`` is applied so the next
     ``consensus_rebuild.rebuild_overlay`` re-derives consensus from exactly the
     chosen runs.

User decisions baked in (plan §"User decisions"):
  * VAAMR selection = top-3 by HIGHEST Cohen κ vs human consensus.
  * No κ computable (no human codes / no overlap) → **select ALL eligible runs +
    a loud warning** (never block, never silently keep the prior selection).

This module reads ballots + human codes and writes only the selection manifest +
``selected`` flags; it never re-runs the LLM and never recomputes human↔human IRR
from scratch (it lifts the stored human band).
"""

import datetime
from typing import Dict, List, Optional, Tuple

from process import run_registry as _rr
from process import classifications_io as _cio
from process import irr_import
from . import irr_stats

# Mirror irr_analysis's bootstrap apparatus exactly so a run's κ-CI here matches
# the per_llm_rater CI in the IRR report (same seed, same reps, same estimator).
from .irr_analysis import (
    _bootstrap_kappa_ci as _kappa_ci,
    _consensus_rows,
)

ABSTAIN_CODE = irr_import.ABSTAIN_CODE

# theme overlay scores against vaamr-kind worksheets; purer would score against
# purer-kind human codes (none exist today → per_run_kappa returns {} for it).
_OVERLAY_TO_FRAMEWORK = {'theme': 'vaamr', 'purer': 'purer'}

# Non-archived statuses whose ballots we score (any status that can carry ballots,
# including the partial 'failed'/'running' — marked via each run's status field).
_SCORED_STATUSES = (
    'queued', 'running', 'completed', 'completed_with_errors', 'failed',
)

# Runs eligible to be SELECTED (must have actually finished a sweep).
_ELIGIBLE_STATUSES = ('completed', 'completed_with_errors')

# Selection-policy defaults until M6's RunSelectionConfig lands. Read getattr-
# tolerantly off config.run_selection so the dataclass slots straight in.
_DEFAULT_POLICY = {
    'theme': {'strategy': 'top_n_by_human_irr', 'n': 3, 'min_kappa': None},
    'purer': {'strategy': 'all', 'n': None, 'min_kappa': None},
}


# ---------------------------------------------------------------------------
# Human consensus truth (derived EXACTLY like irr_analysis's human_vs_llm)
# ---------------------------------------------------------------------------

def _human_truth_by_segment(output_dir: str) -> Dict[str, int]:
    """``{segment_id: human primary code}`` from the resolved consensus rows.

    Mirrors ``irr_analysis``: ``_consensus_rows`` (is_consensus AND source not
    'unresolved'), keyed by the resolved ``segment_id`` (rows without one are
    dropped — they cannot be scored against a machine ballot).  The primary code
    is a VAAMR theme_id or ``ABSTAIN_CODE`` (the "No code" 6th category).
    """
    codes = irr_import.read_human_codes(output_dir)
    if not codes:
        return {}
    out: Dict[str, int] = {}
    for c in _consensus_rows(codes):
        sid = c.get('segment_id')
        if not sid or c.get('primary') is None:
            continue
        out[sid] = c['primary']
    return out


def _ballot_to_stage(cell: Optional[dict]) -> Optional[int]:
    """One run's ballot → a reliability label, mirroring ``_rater_vote_stage``.

    CODED → its stage; ABSTAIN → ``ABSTAIN_CODE`` (the same 6th category the
    human side uses); ERROR / unparseable (``cell is None``) → ``None`` (the cell
    is skipped, never paired).  Accepts both the parsed-run shape
    (``primary_stage``) and the rater_votes-cache shape (``stage``).
    """
    if cell is None or not isinstance(cell, dict):
        return None
    vote = cell.get('vote')
    if vote == 'ERROR':
        return None
    stage = cell.get('primary_stage', cell.get('stage'))
    if vote == 'ABSTAIN':
        return ABSTAIN_CODE
    if vote == 'CODED':
        return stage
    # legacy / no explicit vote field — infer from the stage like irr_analysis.
    if vote is None and stage is not None:
        return stage
    if vote is None and stage is None:
        return ABSTAIN_CODE
    return None


# ---------------------------------------------------------------------------
# Per-run κ
# ---------------------------------------------------------------------------

def per_run_kappa(output_dir: str, overlay: str = 'theme') -> Dict[int, dict]:
    """Cohen κ of EACH non-archived run vs the human consensus, for ``overlay``.

    For every run (any ballot-bearing status except ``archived``) we pair its
    ballots with the human consensus on overlapping segments: CODED→stage,
    ABSTAIN→``ABSTAIN_CODE``, ERROR cells skipped — exactly the encoding
    ``irr_analysis`` uses for the operative consensus.  κ is computed via
    ``irr_stats.cohen_kappa`` with the same bootstrap CI (``_bootstrap_kappa_ci``)
    the IRR report uses for per-model rows.

    Returns ``{run_id: {...}}`` with ``rater_label, model, quantization,
    thinking, note, status, selected, n, cohen_kappa, kappa_ci,
    percent_agreement}``.  κ (and the CI) are ``None`` when ``n < 2`` (the
    cohen_kappa floor).  Returns ``{}`` gracefully when there is no run registry,
    no overlay framework (purer→no human codes), or no human truth.
    """
    if overlay not in _OVERLAY_TO_FRAMEWORK:
        return {}
    runs = [r for r in _rr.list_runs(output_dir, overlay=overlay)
            if r['status'] != 'archived']
    if not runs:
        return {}
    truth = _human_truth_by_segment(output_dir)
    if not truth:
        # No human codes (or none with resolved consensus) — every run scores
        # n=0/κ=None, but still surface the registry so the report/CLI can list
        # the runs (with κ n/a) and the fallback path can select among them.
        return {r['run_id']: _run_meta(r, n=0, kappa=None, ci=None, pa=None)
                for r in runs}

    ballots = _rr.ballots_for_runs(
        output_dir, overlay, [r['run_id'] for r in runs])
    # Invert to per-run cell maps so each run pairs independently on its overlap.
    per_run_cells: Dict[int, Dict[str, Optional[dict]]] = {r['run_id']: {} for r in runs}
    for seg_id, by_run in ballots.items():
        for rid, cell in by_run.items():
            if rid in per_run_cells:
                per_run_cells[rid][seg_id] = cell

    out: Dict[int, dict] = {}
    for r in runs:
        rid = r['run_id']
        h_list: List[int] = []
        m_list: List[int] = []
        for seg_id, human_code in truth.items():
            if seg_id not in per_run_cells[rid]:
                continue
            stage = _ballot_to_stage(per_run_cells[rid][seg_id])
            if stage is None:  # ERROR / unparseable cell — skip, never pair.
                continue
            h_list.append(human_code)
            m_list.append(stage)
        n = len(h_list)
        kappa = irr_stats.cohen_kappa(h_list, m_list) if n >= 2 else None
        ci = _kappa_ci(h_list, m_list) if n >= 2 else None
        pa = irr_stats.observed_agreement(h_list, m_list) if n else None
        out[rid] = _run_meta(r, n=n, kappa=kappa, ci=ci, pa=pa)
    return out


def _run_meta(run: dict, *, n: int, kappa, ci, pa) -> dict:
    """Assemble the per-run κ record (registry metadata + the scored stats)."""
    return {
        'run_id': run['run_id'],
        'rater_label': run['rater_label'],
        'model': run.get('model'),
        'quantization': run.get('quantization'),
        'thinking': run.get('thinking'),
        'note': run.get('note'),
        'status': run.get('status'),
        'selected': bool(run.get('selected')),
        'n': n,
        'cohen_kappa': kappa,
        'kappa_ci': ci,
        'percent_agreement': pa,
    }


# ---------------------------------------------------------------------------
# Human band (lifted from stored IRR results — never recomputed here)
# ---------------------------------------------------------------------------

def _human_band(output_dir: str) -> Optional[Tuple[float, float]]:
    """The stored human↔human Krippendorff α band (lo, hi), or None.

    Lifts ``load_irr_metrics``'s ``llm.human_human_band`` (read off the persisted
    ``irr_results.json``); we never recompute human IRR from scratch (plan §M5).
    """
    try:
        from .irr_analysis import load_irr_metrics
        band = (load_irr_metrics(output_dir).get('llm') or {}).get('human_human_band')
    except Exception:
        band = None
    if band and len(band) == 2 and band[0] is not None and band[1] is not None:
        return (float(band[0]), float(band[1]))
    return None


# ---------------------------------------------------------------------------
# Selection policy
# ---------------------------------------------------------------------------

def selection_manifest_key(overlay: str) -> str:
    """The classification-manifest key the selection record lives under."""
    return f'run_selection:{overlay}'


def load_selection_record(output_dir: str, overlay: str) -> Optional[dict]:
    """Return the persisted selection decision record for ``overlay``, or None."""
    man = _cio.read_classification_manifest(output_dir) or {}
    return man.get(selection_manifest_key(overlay))


def _policy_for(config, overlay: str) -> dict:
    """Resolve the overlay's selection policy from config (getattr-tolerant).

    Reads ``config.run_selection.<overlay-framework>`` when present (M6's
    ``RunSelectionConfig``), falling back to ``_DEFAULT_POLICY`` per field so a
    partial / absent config still yields the documented defaults.
    """
    base = dict(_DEFAULT_POLICY[overlay])
    rs = getattr(config, 'run_selection', None) if config is not None else None
    if rs is None:
        return base
    fw = _OVERLAY_TO_FRAMEWORK[overlay]
    sub = getattr(rs, fw, None)
    if sub is None and isinstance(rs, dict):
        sub = rs.get(fw)
    if sub is None:
        return base
    # 'ids' carries the explicit manual selection (strategy 'manual'); copied
    # through alongside the policy knobs.
    for field in ('strategy', 'n', 'min_kappa', 'ids'):
        val = getattr(sub, field, None)
        if val is None and isinstance(sub, dict):
            val = sub.get(field)
        if val is not None:
            base[field] = val
    return base


def _current_params_hash(output_dir: str) -> Optional[str]:
    """The project's current segmentation params_hash (the staleness anchor)."""
    return _cio._read_any_params_hash(output_dir)


def _eligible_runs(output_dir: str, overlay: str) -> Tuple[List[dict], List[str]]:
    """Runs eligible for selection + any warnings (staleness / NULL-hash).

    Eligible = status ∈ completed/completed_with_errors AND
    ``segmentation_params_hash`` matches the project's current hash.  A run with a
    NULL stored hash is treated as matching (warned), since pre-staleness-stamp
    runs predate the guard.
    """
    current = _current_params_hash(output_dir)
    runs = _rr.list_runs(output_dir, overlay=overlay,
                         statuses=list(_ELIGIBLE_STATUSES))
    eligible: List[dict] = []
    warnings: List[str] = []
    for r in runs:
        h = r.get('segmentation_params_hash')
        if h is None:
            warnings.append(
                f"run {r['run_id']} ({r['rater_label']!r}) has no "
                f"segmentation_params_hash — treating as current (legacy run).")
            eligible.append(r)
        elif current is not None and h != current:
            warnings.append(
                f"run {r['run_id']} ({r['rater_label']!r}) is STALE "
                f"(segmentation hash {h[:8]} != current {current[:8]}) — excluded "
                f"from selection; re-segmentation invalidated its ballots.")
        else:
            eligible.append(r)
    return eligible, warnings


def _rank_by_kappa(eligible: List[dict], kappa: Dict[int, dict]) -> List[dict]:
    """Eligible runs sorted by κ desc, ties by n desc then run_id asc.

    Runs with κ=None (too few overlap items) sort last (treated as worst), so a
    κ-computable run always outranks an un-scorable one.
    """
    def key(r):
        k = kappa.get(r['run_id'], {})
        kv = k.get('cohen_kappa')
        n = k.get('n') or 0
        # κ=None → sort to the bottom: primary key (no-κ flag, -κ); ties n desc, id asc.
        return (kv is None, -(kv if kv is not None else 0.0), -n, r['run_id'])
    return sorted(eligible, key=key)


def select_runs(output_dir: str, config, overlay: str) -> dict:
    """Apply ``overlay``'s selection policy; persist + return the decision record.

    Strategies (``_policy_for``):
      * ``top_n_by_human_irr`` — rank eligible runs by κ desc (ties n desc / id asc),
        apply the optional ``min_kappa`` floor, take the top ``n``.  Fewer than
        ``n`` qualifying → select the qualifiers + a loud warning.  **Zero κ
        computable (no human codes / no overlap on any eligible run) → select ALL
        eligible + a loud warning** (user decision; ``fallback_used=True``).
      * ``all`` — every eligible run (the PURER default).
      * ``manual`` — passthrough of ``policy['ids']`` (used by ``runs select --ids``);
        recorded verbatim.

    Applies ``run_registry.set_selected`` and writes the decision under
    ``run_selection:<overlay>``.  Returns the record plus ``'changed'`` (vs the
    prior selection) so the caller can decide whether to rebuild downstream.

    Never raises on an empty registry; returns a ``skipped`` record instead.
    """
    prior_selected = set(_rr.selected_runs(output_dir, overlay))
    policy = _policy_for(config, overlay)
    strategy = policy.get('strategy', _DEFAULT_POLICY[overlay]['strategy'])

    eligible, warnings = _eligible_runs(output_dir, overlay)
    kappa = per_run_kappa(output_dir, overlay)
    band = _human_band(output_dir)

    if not eligible:
        rationale = (f"no eligible runs for overlay {overlay!r} "
                     f"(need status completed/completed_with_errors with a current "
                     f"segmentation hash) — selection left unchanged.")
        for w in warnings:
            print(f"  *** run_selection WARNING: {w} ***")
        print(f"  *** run_selection: {rationale} ***")
        return _persist(output_dir, overlay, strategy=strategy, policy=policy,
                        selected=[], rejected=[], kappa=kappa, band=band,
                        rationale=rationale, fallback_used=False,
                        prior_selected=prior_selected, warnings=warnings,
                        skipped=True)

    eligible_ids = [r['run_id'] for r in eligible]
    fallback_used = False

    if strategy == 'manual':
        wanted = [int(i) for i in (policy.get('ids') or [])]
        selected = [i for i in wanted if i in eligible_ids]
        rejected = [i for i in eligible_ids if i not in selected]
        rationale = f"manual selection of run ids {selected}"

    elif strategy == 'all':
        selected = list(eligible_ids)
        rejected = []
        rationale = (f"strategy 'all': selected every eligible run "
                     f"({len(selected)}).")

    elif strategy == 'top_n_by_human_irr':
        n_kappa_computable = sum(
            1 for r in eligible if (kappa.get(r['run_id'], {}).get('cohen_kappa')) is not None)
        if n_kappa_computable == 0:
            # User decision: never block, never silently keep — select ALL eligible.
            fallback_used = True
            selected = list(eligible_ids)
            rejected = []
            rationale = (
                "NO human-IRR κ computable for any eligible run (no human "
                "consensus codes or no ballot overlap) → FALLBACK: selected ALL "
                f"{len(selected)} eligible runs.")
            print("  *** run_selection WARNING: " + rationale + " ***")
        else:
            ranked = _rank_by_kappa(eligible, kappa)
            min_kappa = policy.get('min_kappa')
            qualifying = ranked
            if min_kappa is not None:
                qualifying = [
                    r for r in ranked
                    if (kappa.get(r['run_id'], {}).get('cohen_kappa')) is not None
                    and kappa[r['run_id']]['cohen_kappa'] >= min_kappa]
            # κ must be computable to be picked by the top-n ranking.
            qualifying = [
                r for r in qualifying
                if kappa.get(r['run_id'], {}).get('cohen_kappa') is not None]
            n = int(policy.get('n') or 3)
            chosen = qualifying[:n]
            selected = [r['run_id'] for r in chosen]
            rejected = [i for i in eligible_ids if i not in selected]
            top_desc = ', '.join(
                f"{r['rater_label']} κ={kappa[r['run_id']]['cohen_kappa']:+.3f}"
                for r in chosen)
            rationale = (f"top-{n} by human-IRR κ: selected [{top_desc}].")
            if len(chosen) < n:
                floor_note = (f" (min_kappa={min_kappa})" if min_kappa is not None else "")
                warn = (f"only {len(chosen)} run(s) qualified for top-{n}"
                        f"{floor_note}; selected those.")
                rationale += " " + warn
                print("  *** run_selection WARNING: " + warn + " ***")
    else:
        raise ValueError(f"select_runs: unknown strategy {strategy!r}")

    for w in warnings:
        print(f"  *** run_selection WARNING: {w} ***")

    return _persist(output_dir, overlay, strategy=strategy, policy=policy,
                    selected=selected, rejected=rejected, kappa=kappa, band=band,
                    rationale=rationale, fallback_used=fallback_used,
                    prior_selected=prior_selected, warnings=warnings,
                    skipped=False)


def _persist(output_dir, overlay, *, strategy, policy, selected, rejected,
             kappa, band, rationale, fallback_used, prior_selected, warnings,
             skipped) -> dict:
    """Apply ``set_selected`` (unless skipped) + write the manifest record."""
    if not skipped:
        _rr.set_selected(output_dir, overlay, selected)

    snapshot = {
        rid: {
            'kappa': k.get('cohen_kappa'),
            'ci': k.get('kappa_ci'),
            'n': k.get('n'),
            'rater_label': k.get('rater_label'),
        }
        for rid, k in kappa.items()
    }
    record = {
        'overlay': overlay,
        'strategy': strategy,
        'n': policy.get('n'),
        'min_kappa': policy.get('min_kappa'),
        'selected_run_ids': list(selected),
        'rejected_run_ids': list(rejected),
        'kappa_snapshot': snapshot,
        'human_band': list(band) if band else None,
        'decided_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'rationale': rationale,
        'fallback_used': bool(fallback_used),
        'warnings': list(warnings),
        'skipped': bool(skipped),
    }
    _cio.update_classification_manifest(
        output_dir, key=selection_manifest_key(overlay), entry=record)
    record['changed'] = (set(selected) != prior_selected) and not skipped
    return record
