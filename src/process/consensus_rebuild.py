"""
process/consensus_rebuild.py
----------------------------
Rebuild a classification overlay's consensus from the durable per-rater ballots
(schema v2; see ``process.run_registry`` / ``process.db``).

The overlay's consensus columns (``theme_labels`` / ``purer_labels``) are a
*derived view* of the ``label_ballots`` rows of the **selected** runs.  This
module re-votes those selected ballots through the very same
``llm_classifier.build_merge_result`` the inline classification path uses, then
applies the result onto frozen segments via the same ``response_parser`` idioms
— so identical ballots produce a byte-identical overlay (the M2 gate test
asserts this record-for-record).

``rebuild_overlay`` is the single entry point.  It never re-runs the LLM; it
only re-aggregates ballots already in ``qra.db``.  Because consensus is now
derived from the *selected* subset, changing which runs are selected and
rebuilding is how VAAMR top-n / PURER all-runs selection takes effect, with zero
reader changes downstream (``rater_votes``/``rater_ids`` become the
selected-ballots cache).

Supported overlays: ``'theme'`` and ``'purer'``.  Codebook is multi-label and
out of scope for v1 (raises ``ValueError``).
"""

from typing import Dict, List, Optional

from classification_tools.data_structures import Segment
from classification_tools.response_parser import parse_all_results, parse_purer_results
from classification_tools.theme_llm.llm_classifier import build_merge_result

from . import db
from . import run_registry as _rr
from . import segments_io
from . import classifications_io as _cio


_SUPPORTED = ('theme', 'purer')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _overlay_vote_config(overlay: str, config):
    """Resolve (vote_mode, tie_break_order, secondary_weight, presence_threshold)
    from the overlay's classification sub-config — read off the OBJECT at call
    time (the concurrent vote_mode default work may change these).

    Mirrors ``classify_purer_cue_units`` after M0: PURER falls back to the
    documented move-precedence (``PURER_TIE_BREAK_ORDER``) when the config
    supplies no explicit tie-break order.
    """
    sub = getattr(config, 'theme_classification', None) if overlay == 'theme' \
        else getattr(config, 'purer_classification', None)

    vote_mode = getattr(sub, 'vote_mode', 'majority') if sub is not None else 'majority'
    tie_break_order = getattr(sub, 'tie_break_order', None) if sub is not None else None
    secondary_weight = getattr(sub, 'evidence_secondary_weight', 0.6) if sub is not None else 0.6
    presence_threshold = getattr(sub, 'evidence_presence_threshold', 0.5) if sub is not None else 0.5

    if overlay == 'purer' and not tie_break_order:
        from constructs.purer import PURER_TIE_BREAK_ORDER
        tie_break_order = PURER_TIE_BREAK_ORDER

    return vote_mode, tie_break_order, secondary_weight, presence_threshold


def _applies_to_map(run_dir: str, overlay: str, run_ids: List[int]) -> Dict[str, list]:
    """Map ``unit segment_id -> constituent ids`` from ``applies_to_json``.

    The propagation target is identical for a unit across runs, so any run's row
    serves; we read all selected runs and keep the first non-null mapping seen.
    Units without an ``applies_to_json`` (turn-mode PURER, where the unit IS the
    therapist segment) are absent from the result and treated as self-mapping by
    the caller.
    """
    out: Dict[str, list] = {}
    if not run_ids or not db.db_exists(run_dir):
        return out
    placeholders = ', '.join('?' for _ in run_ids)
    with db.open_db(run_dir) as conn:
        rows = conn.execute(
            f"SELECT segment_id, applies_to_json FROM label_ballots "
            f"WHERE overlay = ? AND run_id IN ({placeholders}) "
            f"AND applies_to_json IS NOT NULL",
            (overlay, *[int(r) for r in run_ids]),
        ).fetchall()
    for r in rows:
        if r['segment_id'] in out:
            continue
        ids = db.loads(r['applies_to_json'])
        if ids:
            out[r['segment_id']] = list(ids)
    return out


def _merged_by_unit(run_dir, overlay, selected, rater_ids,
                    *, vote_mode, tie_break_order,
                    secondary_weight, presence_threshold) -> Dict[str, dict]:
    """Re-vote every unit's selected ballots → ``{unit_segment_id: merge_dict}``.

    Slot ``k`` of each unit's ``parsed_runs`` is the ``selected[k]`` run's ballot
    (missing cell → None, i.e. an ERROR ballot), so the rater ordering matches
    ``rater_ids`` exactly — the same slot contract the inline path relies on.
    """
    ballots = _rr.ballots_for_runs(run_dir, overlay, selected)
    n = len(selected)
    merged: Dict[str, dict] = {}
    for seg_id, by_run in ballots.items():
        parsed_runs = [by_run.get(rid) for rid in selected]
        merged[seg_id] = build_merge_result(
            parsed_runs, rater_ids,
            n_runs=n,
            secondary_weight=secondary_weight,
            presence_threshold=presence_threshold,
            vote_mode=vote_mode,
            tie_break_order=tie_break_order,
        )
    return merged


def _primary_by_segment(records: List[dict], key: str) -> Dict[str, object]:
    """Map ``segment_id -> primary label`` from prior overlay records (for the
    n_changed diff)."""
    return {r['segment_id']: r.get(key) for r in records}


def _count_outcomes(merged_by_unit: Dict[str, dict]) -> Dict[str, int]:
    """Tally labeled / abstain / unlabeled units from their merge dicts."""
    n_labeled = n_abstain = n_unlabeled = 0
    for m in merged_by_unit.values():
        cons = m.get('consensus') or {}
        if cons.get('primary_stage') is not None:
            n_labeled += 1
        elif cons.get('consensus_vote') == 'ABSTAIN':
            n_abstain += 1
        else:
            n_unlabeled += 1
    return {'n_labeled': n_labeled, 'n_abstain': n_abstain, 'n_unlabeled': n_unlabeled}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def rebuild_overlay(
    run_dir: str,
    overlay: str,
    config,
    *,
    only_segment_ids: Optional[set] = None,
) -> dict:
    """Rebuild ``overlay``'s consensus from the selected runs' ballots.

    Parameters
    ----------
    run_dir : str
        Project output directory (holds ``qra.db``).
    overlay : str
        ``'theme'`` or ``'purer'`` (anything else raises ``ValueError``).
    config : PipelineConfig
        Source of the overlay's vote_mode / tie_break_order (read off the config
        OBJECT at call time) and framework names for the manifest entry.
    only_segment_ids : set, optional
        When given, only these *unit* segment ids are re-voted and the overlay is
        UPSERT-merged (untouched rows preserved).  When None, the whole overlay
        is rewritten from every selected ballot (the intentional full-rewrite
        asymmetry vs the legacy inline merge-over-stale).

    Returns
    -------
    dict
        ``{'n_units', 'n_labeled', 'n_abstain', 'n_unlabeled', 'n_changed',
        'models_used', 'run_ids'}`` on success, or ``{'skipped': True,
        'reason': ...}`` when no runs are selected (the overlay is left
        untouched).
    """
    if overlay not in _SUPPORTED:
        raise ValueError(
            f"rebuild_overlay: overlay must be one of {_SUPPORTED}, got {overlay!r} "
            "(codebook is multi-label and unsupported in v1)."
        )

    selected = _rr.selected_runs(run_dir, overlay)
    if not selected:
        reason = (f"no selected runs for overlay {overlay!r}; overlay left "
                  f"untouched (run `qra runs select` / classify first)")
        print(f"  *** consensus_rebuild WARNING: {reason} ***")
        return {'skipped': True, 'reason': reason}

    # rater_label, in run_id (= slot) order — the rater ids carried into ballots.
    runs = _rr.list_runs(run_dir, overlay=overlay)
    label_by_id = {r['run_id']: r['rater_label'] for r in runs}
    rater_ids = [label_by_id[rid] for rid in selected]

    (vote_mode, tie_break_order,
     secondary_weight, presence_threshold) = _overlay_vote_config(overlay, config)

    merged_by_unit = _merged_by_unit(
        run_dir, overlay, selected, rater_ids,
        vote_mode=vote_mode, tie_break_order=tie_break_order,
        secondary_weight=secondary_weight, presence_threshold=presence_threshold,
    )
    if only_segment_ids is not None:
        merged_by_unit = {sid: m for sid, m in merged_by_unit.items()
                          if sid in only_segment_ids}

    stats = _count_outcomes(merged_by_unit)
    if overlay == 'theme':
        n_changed = _apply_theme(run_dir, merged_by_unit, only_segment_ids)
    else:
        n_changed = _apply_purer(run_dir, selected, merged_by_unit, only_segment_ids)

    _update_manifest(run_dir, overlay, config, selected, rater_ids,
                     n_segments=len(merged_by_unit))

    return {
        'n_units': len(merged_by_unit),
        'n_labeled': stats['n_labeled'],
        'n_abstain': stats['n_abstain'],
        'n_unlabeled': stats['n_unlabeled'],
        'n_changed': n_changed,
        'models_used': list(rater_ids),
        'run_ids': list(selected),
    }


# ---------------------------------------------------------------------------
# Theme apply path
# ---------------------------------------------------------------------------

def _apply_theme(run_dir, merged_by_unit, only_segment_ids) -> int:
    """Map theme merge dicts onto frozen participant segments and (over)write the
    overlay.  Returns the number of rows whose primary_stage changed vs the prior
    overlay (the M0 label-churn signal)."""
    prior = _primary_by_segment(_cio.read_overlay(run_dir, 'theme'), 'primary_stage')

    # Frozen participant segments (no theme overlay applied — we are rebuilding
    # it).  Speaker filtering is the participant scope the classify stage uses.
    segments = segments_io.load_segments_for_stage(
        run_dir, apply=('purer', 'codebook', 'cv'),
    )
    by_id = {s.segment_id: s for s in segments}

    # Apply consensus onto the matching Segment objects (same code as inline).
    targets = [by_id[sid] for sid in merged_by_unit if sid in by_id]
    parse_all_results(merged_by_unit, targets, name_to_id={})

    if only_segment_ids is not None:
        changed_segs = [s for s in targets if s.segment_id in only_segment_ids]
        _cio.merge_theme_overlay(run_dir, changed_segs)
        new_primary = {s.segment_id: s.primary_stage for s in changed_segs}
    else:
        _cio.write_theme_overlay(run_dir, segments)
        new_primary = {s.segment_id: s.primary_stage for s in targets}

    return _count_changed(prior, new_primary)


# ---------------------------------------------------------------------------
# Purer apply path
# ---------------------------------------------------------------------------

def _apply_purer(run_dir, selected, merged_by_unit, only_segment_ids) -> int:
    """Propagate PURER unit consensus onto constituent therapist segments and
    (over)write the overlay.

    A unit's ballots may be cue-unit-level (``applies_to_json`` lists the
    constituent therapist ids) or plain turn-level (the unit IS the therapist
    segment → self-mapping).  Either way we expand each unit's merge dict onto
    its constituents and run them through ``parse_purer_results`` — the same
    response_parser code the inline path uses — so ``purer_run_consistency`` /
    agreement fields come out identical to ``_propagate``.

    Full-rewrite by default (clears stale rows the legacy inline merge would have
    left); ``only_segment_ids`` restricts to a constituent subset (upsert).
    """
    prior = _primary_by_segment(_cio.read_overlay(run_dir, 'purer'), 'purer_primary')

    applies = _applies_to_map(run_dir, 'purer', selected)

    # Expand: constituent_segment_id -> the unit's merge dict.
    by_constituent: Dict[str, dict] = {}
    for unit_id, merge in merged_by_unit.items():
        constituents = applies.get(unit_id, [unit_id])  # turn mode: self
        for cid in constituents:
            by_constituent[cid] = merge

    # Frozen segments (no purer overlay applied — we are rebuilding it).
    segments = segments_io.load_segments_for_stage(
        run_dir, apply=('theme', 'codebook', 'cv'),
    )
    by_id = {s.segment_id: s for s in segments}
    targets = [by_id[cid] for cid in by_constituent if cid in by_id]

    # parse_purer_results keys results_all by the *constituent* segment_id.
    results_for_parse = {s.segment_id: by_constituent[s.segment_id] for s in targets}
    parse_purer_results(results_for_parse, targets)

    if only_segment_ids is not None:
        changed_segs = [s for s in targets if s.segment_id in only_segment_ids]
        _cio.merge_purer_overlay(run_dir, changed_segs)
        new_primary = {s.segment_id: s.purer_primary for s in changed_segs}
    else:
        _cio.write_purer_overlay(run_dir, segments)
        new_primary = {s.segment_id: s.purer_primary for s in targets}

    return _count_changed(prior, new_primary)


def _count_changed(prior: Dict[str, object], new: Dict[str, object]) -> int:
    """Count segments whose primary label differs from the prior overlay."""
    changed = 0
    for sid, val in new.items():
        if prior.get(sid) != val:
            changed += 1
    return changed


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _update_manifest(run_dir, overlay, config, selected, rater_ids, *, n_segments):
    """Record the rebuild provenance under the overlay's manifest key.

    ``model`` is the FIRST selected run's real model string (NOT the joined
    rater_labels — pinning that joined string as an LLM model id later poisoned
    ``qra add-data`` incremental classification).  Per-run roster provenance is
    preserved in ``per_run_models`` (each selected run's model, in slot order)
    and ``rater_labels``; plus the rebuild-specific keys (``vote_mode``,
    ``rebuilt_from_ballots``, ``run_ids``).
    """
    vote_mode, _, _, _ = _overlay_vote_config(overlay, config)
    # Read each selected run's real model string (slot order = selected order).
    runs = _rr.list_runs(run_dir, overlay=overlay)
    model_by_id = {r['run_id']: r.get('model') for r in runs}
    per_run_models = [model_by_id.get(rid) for rid in selected]
    first_model = next((m for m in per_run_models if m), None)
    entry = {
        'model': first_model,
        'per_run_models': list(per_run_models),
        'rater_labels': list(rater_ids),
        'n_runs': len(selected),
        'n_segments': n_segments,
        'vote_mode': vote_mode,
        'rebuilt_from_ballots': True,
        'run_ids': list(selected),
    }
    # Framework name+version when cheaply available (theme → VAAMR).
    fw_name = None
    if overlay == 'theme':
        fw_name = getattr(config, 'participant_framework', 'vaamr') if config else 'vaamr'
    elif overlay == 'purer':
        fw_name = getattr(config, 'therapist_framework', 'purer') if config else 'purer'
    if fw_name:
        try:
            from constructs.registry import load as _load_fw
            fw = _load_fw(fw_name)
            entry['framework'] = {
                'name': fw.name, 'version': getattr(fw, 'version', '?'),
            }
        except Exception:
            pass

    _cio.update_classification_manifest(run_dir, key=overlay, entry=entry)
