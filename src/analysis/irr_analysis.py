"""
analysis/irr_analysis.py
------------------------
Inter-rater-reliability (IRR) orchestrator.

Loads the imported human codes from ``qra.db`` (see ``process.irr_import``), pulls
the project's CURRENT machine labels live (LLM ``theme_labels`` + GNN ``gnn_labels``
via reconstructed ``Segment`` objects), and computes the comparison families:

  1. Human ↔ Human            — per test-set, primary + secondary
  2. Human-consensus ↔ LLM    — vs the multi-run LLM consensus AND vs each
                                individual LLM rater/model (from ``rater_votes``)
  3. Human-consensus ↔ GNN    — vs the graph model's predictions

Because the human codes live in ``qra.db`` and machine labels are pulled live each
run, re-running ``qra irr run`` after re-classifying (new LLM) or retraining the GNN
re-measures IRR against the updated models without re-importing — the human
ground-truth is maintained for all validation moving forward.

All chance-corrected statistics come from proven libraries (see ``analysis.irr_stats``):
Cohen's κ (scikit-learn), Fleiss' κ (statsmodels), Krippendorff's α (krippendorff).

All machine-readable artifacts (results JSON, pairwise CSV, discrepancy CSV) and the
figures are written under ``04_validation/irr/``; the single human-facing report is
written by ``analysis.reports.irr_report``.

Label encoding: VAAMR theme_ids 0–4, plus ``-1`` = ABSTAIN ("No code").
``None`` = no ballot / no machine data (excluded).
"""

import csv
import datetime
import itertools
import json
import os
from typing import Dict, List, Optional, Tuple

from process import output_paths as _paths
from process import irr_import
from process import segments_io
from constructs.registry import load as load_framework
from . import irr_stats

ABSTAIN_CODE = irr_import.ABSTAIN_CODE
CONSENSUS_RATER = irr_import.CONSENSUS_RATER


def _label_name(id_to_short: Dict[int, str], code: Optional[int]) -> str:
    if code is None:
        return '(none)'
    if code == ABSTAIN_CODE:
        return 'No code'
    return id_to_short.get(code, str(code))


# ---------------------------------------------------------------------------
# Human ↔ Human
# ---------------------------------------------------------------------------

def _human_codes_by_worksheet(
    codes: List[dict],
) -> Dict[int, Dict[int, Dict[str, dict]]]:
    """Nest rater (non-consensus) codes as ws -> item -> rater -> {primary, secondary}."""
    out: Dict[int, Dict[int, Dict[str, dict]]] = {}
    for c in codes:
        if c['is_consensus']:
            continue
        out.setdefault(c['worksheet_n'], {}).setdefault(c['item_num'], {})[c['rater']] = c
    return out


def _ballot(code: Optional[int]):
    """Map a stored code to a reliability ballot: int theme_id, 'ABSTAIN', or None."""
    if code is None:
        return None
    if code == ABSTAIN_CODE:
        return 'ABSTAIN'
    return code


def _human_human_for_worksheet(
    roster: List[str],
    items: Dict[int, Dict[str, dict]],
    field: str,
) -> Optional[dict]:
    """Human↔Human IRR for one worksheet on ``field`` ('primary'|'secondary')."""
    matrix: List[list] = []
    for item_num in sorted(items):
        raters = items[item_num]
        row = [_ballot(raters[r][field]) if r in raters else None for r in roster]
        if sum(1 for b in row if b is not None) >= 1:
            matrix.append(row)
    multi = [r for r in matrix if sum(1 for b in r if b is not None) >= 2]
    if len(multi) < 2:
        return None

    agree = irr_stats.unanimous_agreement(matrix)
    fleiss = irr_stats.fleiss_kappa(matrix)
    result = {
        'field': field,
        'n_raters': len(roster),
        'n_items_scored': agree['n_items'],
        'krippendorff_alpha': irr_stats.krippendorff_alpha(matrix),
        'fleiss_kappa': fleiss['kappa'],
        'fleiss_n_complete': fleiss['n_complete'],
        'percent_agreement_unanimous': agree['unanimous'],
        'percent_agreement_pairwise': agree['pairwise'],
        'pairwise': [],
    }

    for ra, rb in itertools.combinations(roster, 2):
        a_vals, b_vals = [], []
        for item_num in sorted(items):
            raters = items[item_num]
            if ra in raters and rb in raters:
                va, vb = raters[ra][field], raters[rb][field]
                if va is not None and vb is not None:
                    a_vals.append(va)
                    b_vals.append(vb)
        if not a_vals:
            continue  # pair never overlapped (e.g. Adam↔Ryan on T2)
        result['pairwise'].append({
            'rater_a': ra,
            'rater_b': rb,
            'n': len(a_vals),
            'cohen_kappa': irr_stats.cohen_kappa(a_vals, b_vals),
            'percent_agreement': irr_stats.observed_agreement(a_vals, b_vals),
        })
    return result


# ---------------------------------------------------------------------------
# Human-consensus ↔ machine
# ---------------------------------------------------------------------------

def _consensus_rows(codes: List[dict]) -> List[dict]:
    """Consensus-of-record rows with a usable (non-unresolved) source."""
    return [c for c in codes
            if c['is_consensus'] and (c['source'] or '').lower() != 'unresolved']


def _machine_labels(run_dir: str):
    """Return ``(by_id, present)`` mapping segment_id -> Segment."""
    try:
        segs = segments_io.read_master_segments(run_dir)
    except FileNotFoundError:
        return {}, False
    return {s.segment_id: s for s in segs}, True


def _human_vs_llm(
    consensus: List[dict],
    by_id: Dict[str, object],
    id_to_short: Dict[int, str],
) -> dict:
    """Human consensus vs the multi-run LLM consensus (theme_labels.primary_stage)."""
    return _agreement_block(
        consensus, by_id, id_to_short,
        machine_of=lambda seg: (ABSTAIN_CODE if seg.primary_stage is None
                                else seg.primary_stage),
    )


def _gnn_pred(seg, gnn_heldout: Dict[str, dict]):
    """Return ``(pred, source)`` for the GNN axis, preferring the honest held-out
    prediction over the in-sample distillation overlay. ``pred`` is None when the
    GNN has no usable label for the segment (deferred)."""
    if gnn_heldout:
        rec = gnn_heldout.get(seg.segment_id)
        if rec and rec.get('vaamr_pred') is not None:
            return rec['vaamr_pred'], 'heldout'
        return None, 'heldout'
    if getattr(seg, 'gnn_vaamr_abstain', None) == 1 or seg.gnn_vaamr_pred is None:
        return None, 'distillation'
    return seg.gnn_vaamr_pred, 'distillation'


def _agreement_block(consensus, by_id, id_to_short, *, machine_of) -> dict:
    """Shared human-consensus-vs-machine aggregation. ``machine_of(seg)`` returns
    the machine label (int / ABSTAIN_CODE) or None to defer (exclude) the item."""
    overall_h, overall_m = [], []
    per_ws: Dict[int, Tuple[List[int], List[int]]] = {}
    n_excluded = 0
    n_deferred = 0
    for c in consensus:
        seg = by_id.get(c['segment_id']) if c['segment_id'] else None
        if seg is None:
            n_excluded += 1
            continue
        m = machine_of(seg)
        if m is None:
            n_deferred += 1
            continue
        h = c['primary']
        overall_h.append(h)
        overall_m.append(m)
        per_ws.setdefault(c['worksheet_n'], ([], []))
        per_ws[c['worksheet_n']][0].append(h)
        per_ws[c['worksheet_n']][1].append(m)

    labels = sorted(set(overall_h) | set(overall_m))
    names = [_label_name(id_to_short, c) for c in labels]
    return {
        'n': len(overall_h),
        'n_excluded_no_machine': n_excluded,
        'n_deferred': n_deferred,
        'cohen_kappa': irr_stats.cohen_kappa(overall_h, overall_m),
        'percent_agreement': irr_stats.observed_agreement(overall_h, overall_m),
        'per_worksheet': {
            str(ws): {
                'n': len(h),
                'cohen_kappa': irr_stats.cohen_kappa(h, m),
                'percent_agreement': irr_stats.observed_agreement(h, m),
            }
            for ws, (h, m) in sorted(per_ws.items())
        },
        'confusion': irr_stats.confusion(overall_h, overall_m, labels, names),
    }


def _machine_vs_machine(seg_ids, by_id, pred_a, pred_b) -> dict:
    """Agreement between two machine substrates over the given segments (e.g. GNN
    vs LLM consensus). ``pred_*(seg)`` return a label or None to skip."""
    a_vals, b_vals = [], []
    for sid in seg_ids:
        seg = by_id.get(sid)
        if seg is None:
            continue
        pa, pb = pred_a(seg), pred_b(seg)
        if pa is None or pb is None:
            continue
        a_vals.append(pa)
        b_vals.append(pb)
    return {
        'n': len(a_vals),
        'cohen_kappa': irr_stats.cohen_kappa(a_vals, b_vals),
        'percent_agreement': irr_stats.observed_agreement(a_vals, b_vals),
    }


def _llm_label(seg):
    return ABSTAIN_CODE if seg.primary_stage is None else seg.primary_stage


def _gnn_full(
    consensus: List[dict],
    all_seg_ids: List[str],
    by_id: Dict[str, object],
    gnn_heldout: Dict[str, dict],
    id_to_short: Dict[int, str],
) -> dict:
    """Both GNN axes, each vs human consensus AND vs the LLM consensus.

    held-out      = out-of-fold predictions (never trained on the segment's own LLM
                    label) -> the honest validity axis + reliability gate.
    distillation  = the in-sample consensus overlay (trained on every label) -> the
                    operational default; vs-LLM is distillation FIDELITY (circular).
    """
    def held(seg):
        rec = gnn_heldout.get(seg.segment_id) if gnn_heldout else None
        return rec.get('vaamr_pred') if rec else None

    def dist(seg):
        if getattr(seg, 'gnn_vaamr_abstain', None) == 1 or seg.gnn_vaamr_pred is None:
            return None
        return seg.gnn_vaamr_pred

    heldout_block = {
        'available': bool(gnn_heldout),
        'vs_human': _agreement_block(consensus, by_id, id_to_short, machine_of=held),
        'vs_llm': _machine_vs_machine(all_seg_ids, by_id, held, _llm_label),
    }
    distill_block = {
        'available': any(dist(by_id[s]) is not None for s in all_seg_ids if s in by_id),
        'vs_human': _agreement_block(consensus, by_id, id_to_short, machine_of=dist),
        'vs_llm': _machine_vs_machine(all_seg_ids, by_id, dist, _llm_label),
    }
    axis = 'heldout' if gnn_heldout else 'distillation'
    operative = heldout_block if gnn_heldout else distill_block

    result = {
        'operative_axis': axis,
        'heldout': heldout_block,
        'distillation': distill_block,
    }
    # Mirror the operative vs-human block at the top level for the headline/figures.
    result.update(operative['vs_human'])
    result['gnn_source'] = axis
    return result


def _rater_vote_stage(rv: dict) -> Optional[int]:
    """Extract one LLM rater's VAAMR ballot from a rater_votes entry."""
    vote = rv.get('vote')
    if vote == 'ABSTAIN':
        return ABSTAIN_CODE
    if vote == 'CODED':
        return rv.get('stage')
    # legacy / no explicit vote field
    if vote is None and rv.get('stage') is not None:
        return rv.get('stage')
    return None  # ERROR / unparseable


def _human_vs_llm_raters(
    consensus: List[dict],
    by_id: Dict[str, object],
) -> Dict[str, dict]:
    """Human consensus vs EACH individual LLM rater/model (from ``rater_votes``).

    Lets us report agreement per LLM rater, not just against the pooled consensus.
    """
    per_rater_h: Dict[str, List[int]] = {}
    per_rater_m: Dict[str, List[int]] = {}
    for c in consensus:
        seg = by_id.get(c['segment_id']) if c['segment_id'] else None
        if seg is None:
            continue
        votes = getattr(seg, 'rater_votes', None) or []
        h = c['primary']
        for rv in votes:
            rid = rv.get('rater')
            if not rid:
                continue
            m = _rater_vote_stage(rv)
            if m is None:
                continue
            per_rater_h.setdefault(rid, []).append(h)
            per_rater_m.setdefault(rid, []).append(m)
    out = {}
    for rid in sorted(per_rater_h):
        h, m = per_rater_h[rid], per_rater_m[rid]
        out[rid] = {
            'n': len(h),
            'cohen_kappa': irr_stats.cohen_kappa(h, m),
            'percent_agreement': irr_stats.observed_agreement(h, m),
        }
    return out


# ---------------------------------------------------------------------------
# Discrepancies
# ---------------------------------------------------------------------------

def _discrepancies(
    consensus: List[dict],
    rater_codes: Dict[int, Dict[int, Dict[str, dict]]],
    by_id: Dict[str, object],
    gnn_heldout: Dict[str, dict],
    id_to_short: Dict[int, str],
) -> List[dict]:
    """Every consensus item where human ≠ LLM and/or human ≠ GNN."""
    rows = []
    for c in consensus:
        seg = by_id.get(c['segment_id']) if c['segment_id'] else None
        if seg is None:
            continue
        h = c['primary']
        llm = ABSTAIN_CODE if seg.primary_stage is None else seg.primary_stage
        gnn, _src = _gnn_pred(seg, gnn_heldout)

        disagrees_llm = (h != llm)
        disagrees_gnn = (gnn is not None and h != gnn)
        if not (disagrees_llm or disagrees_gnn):
            continue

        ws, item = c['worksheet_n'], c['item_num']
        per_rater = rater_codes.get(ws, {}).get(item, {})
        rater_str = '; '.join(
            f"{r}={_label_name(id_to_short, per_rater[r]['primary'])}"
            for r in sorted(per_rater)
        )
        rows.append({
            'worksheet_n': ws,
            'item_num': item,
            'segment_id': c['segment_id'],
            'human_consensus': _label_name(id_to_short, h),
            'consensus_source': c['source'],
            'rater_codes': rater_str,
            'llm_label': _label_name(id_to_short, llm),
            'llm_confidence': getattr(seg, 'llm_confidence_primary', None),
            'gnn_label': 'deferred' if gnn is None else _label_name(id_to_short, gnn),
            'gnn_confidence': (gnn_heldout.get(seg.segment_id, {}).get('vaamr_conf')
                               if gnn_heldout else getattr(seg, 'gnn_vaamr_conf', None)),
            'disagrees_llm': disagrees_llm,
            'disagrees_gnn': disagrees_gnn,
            'text': (seg.text or '').replace('\n', ' ').strip(),
        })
    rows.sort(key=lambda r: (r['worksheet_n'], r['item_num']))
    return rows


# ---------------------------------------------------------------------------
# Per-item, per-testset detail (full content + reasonings)
# ---------------------------------------------------------------------------

def _llm_rater_rows(seg, id_to_short) -> List[dict]:
    """Per-LLM-rater ballots + justifications from a segment's rater_votes."""
    out = []
    for rv in (getattr(seg, 'rater_votes', None) or []):
        stage = _rater_vote_stage(rv)
        out.append({
            'rater': rv.get('rater'),
            'vote': rv.get('vote'),
            'label': _label_name(id_to_short, stage) if stage is not None else 'ERROR',
            'justification': (rv.get('justification') or '').strip(),
        })
    return out


def _item_details(
    codes: List[dict],
    rater_codes: Dict[int, Dict[int, Dict[str, dict]]],
    by_id: Dict[str, object],
    gnn_heldout: Dict[str, dict],
    id_to_short: Dict[int, str],
) -> List[dict]:
    """Build a full per-item record for every worksheet item (including unresolved):
    segment text, human codes + reasoning, LLM codes + justifications, GNN held-out
    prediction, and the LLM↔GNN consensus."""
    cons_by_item: Dict[int, Dict[int, dict]] = {}
    for c in codes:
        if c['is_consensus']:
            cons_by_item.setdefault(c['worksheet_n'], {})[c['item_num']] = c

    out: List[dict] = []
    for ws in sorted(cons_by_item):
        for item in sorted(cons_by_item[ws]):
            cons = cons_by_item[ws][item]
            seg = by_id.get(cons['segment_id']) if cons['segment_id'] else None
            raters = rater_codes.get(ws, {}).get(item, {})
            human_raters = {
                r: {
                    'primary': _label_name(id_to_short, raters[r]['primary']),
                    'secondary': (_label_name(id_to_short, raters[r]['secondary'])
                                  if raters[r]['secondary'] is not None else ''),
                }
                for r in sorted(raters)
            }

            llm_stage = seg.primary_stage if seg else None
            gnn_pred, gnn_src = (_gnn_pred(seg, gnn_heldout) if seg else (None, None))
            gnn_conf = None
            if seg and gnn_heldout:
                gnn_conf = gnn_heldout.get(seg.segment_id, {}).get('vaamr_conf')
            elif seg:
                gnn_conf = getattr(seg, 'gnn_vaamr_conf', None)

            # LLM↔GNN consensus (on the comparable VAAMR axis).
            llm_norm = ABSTAIN_CODE if llm_stage is None else llm_stage
            if gnn_pred is None:
                llm_gnn = {'agree': None, 'label': None, 'note': 'GNN deferred'}
            else:
                llm_gnn = {
                    'agree': (llm_norm == gnn_pred),
                    'label': _label_name(id_to_short, gnn_pred) if llm_norm == gnn_pred else None,
                }

            out.append({
                'worksheet_n': ws,
                'item_num': item,
                'segment_id': cons['segment_id'],
                'text': (seg.text or '').strip() if seg else '',
                'human': {
                    'consensus': _label_name(id_to_short, cons['primary'])
                                 if cons['primary'] is not None else 'unresolved',
                    'consensus_secondary': (_label_name(id_to_short, cons['secondary'])
                                            if cons['secondary'] is not None else ''),
                    'source': cons['source'],
                    'notes': cons.get('notes') or '',
                    'raters': human_raters,
                },
                'llm': {
                    'consensus': _label_name(id_to_short, llm_stage) if seg else '(no machine label)',
                    'justification': (getattr(seg, 'llm_justification', '') or '').strip() if seg else '',
                    'raters': _llm_rater_rows(seg, id_to_short) if seg else [],
                },
                'gnn': {
                    'prediction': _label_name(id_to_short, gnn_pred) if gnn_pred is not None
                                  else ('deferred' if seg else '(no machine label)'),
                    'confidence': gnn_conf,
                    'source': gnn_src,
                    'distillation': (_label_name(id_to_short, seg.gnn_vaamr_pred)
                                     if seg and seg.gnn_vaamr_pred is not None else ''),
                },
                'llm_gnn_consensus': llm_gnn,
            })
    return out


# ---------------------------------------------------------------------------
# Change detection + auto-regeneration
# ---------------------------------------------------------------------------

def machine_signature(output_dir: str) -> str:
    """A short hash of the current machine state (LLM + GNN labels + held-out
    predictions). Changes whenever re-classification or GNN retraining would alter
    the IRR — used to decide whether to regenerate during `qra analyze`."""
    import hashlib
    from process import classifications_io as _cio
    man = _cio.read_classification_manifest(output_dir) or {}
    parts = []
    for k in ('theme', 'gnn'):
        e = man.get(k) or {}
        parts.append(f"{k}:{e.get('model', '')}|{e.get('completed_at', '')}|{e.get('n_segments', '')}")
    hp = os.path.join(_paths.gnn_data_dir(output_dir), 'gnn_heldout_predictions.csv')
    parts.append(f"heldout:{os.path.getmtime(hp) if os.path.isfile(hp) else 0}")
    # Coded segment content: drift here means IRR would score text the humans never saw,
    # so a change must force regeneration (and surface the drift banner).
    parts.append(f"testset:{irr_import.testset_content_signature(output_dir)}")
    return hashlib.sha256('||'.join(parts).encode()).hexdigest()[:16]


def _prev_signature(output_dir: str) -> Optional[str]:
    path = os.path.join(_paths.irr_validation_dir(output_dir), 'irr_results.json')
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f).get('machine_signature')
    except Exception:
        return None


def maybe_run_irr(output_dir: str, config=None, *, verbose: bool = True,
                  force: bool = False):
    """Regenerate the full IRR suite IFF the project has imported human codes AND
    the machine state changed since the last run (or ``force``).

    Returns the results dict, the string ``'unchanged'`` (skipped), or ``None``
    (no human codes imported). Safe to call unconditionally from the analysis runner.
    """
    if not irr_import.read_human_codes(output_dir):
        return None
    report_path = _paths.reports_irr_path(output_dir)
    if (not force and os.path.isfile(report_path)
            and _prev_signature(output_dir) == machine_signature(output_dir)):
        return 'unchanged'
    results = run_irr_analysis(output_dir, config, verbose=verbose)
    from .reports import irr_report, irr_items
    irr_report.generate_irr_report(results, output_dir)
    irr_items.write_irr_item_details(results, output_dir)
    try:
        from . import irr_figures
        irr_figures.write_irr_figures(results, output_dir)
    except Exception:
        pass  # figures are non-essential; never block analysis on a plotting error
    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_irr_analysis(output_dir: str, config=None, *, verbose: bool = True,
                     strict_drift: bool = False) -> dict:
    """Compute the IRR families and write JSON + CSV artifacts. Returns the results dict.

    ``strict_drift`` (the ``qra irr run`` default): raise ``irr_import.TestsetDriftError``
    if any test-set item's current segment text no longer matches what the humans coded.
    When False (the auto ``qra analyze`` path), drift is recorded in
    ``results['testset_drift']`` and warned loudly instead of aborting.
    """
    fw = load_framework('vaamr')
    id_to_short = fw.build_id_to_short_map()

    codes = irr_import.read_human_codes(output_dir)
    if not codes:
        raise RuntimeError(
            "No imported IRR human codes found. Run `qra irr import` first."
        )
    # Content guard: the segments scored below must be byte-identical to what the humans
    # coded. Re-segmentation / re-ingest after import is the way this could silently break.
    drift = irr_import.check_testset_drift(output_dir)
    if drift:
        banner = irr_import.format_drift_banner(drift)
        if strict_drift:
            raise irr_import.TestsetDriftError(banner, drift)
        if verbose:
            print("‼ TESTSET DRIFT — IRR may not reflect the coded content:\n" + banner)
    testsets = irr_import.list_imported_testsets(output_dir)
    roster_by_ws = {t['worksheet_n']: t['raters'] for t in testsets}

    rater_codes = _human_codes_by_worksheet(codes)
    by_id, have_segments = _machine_labels(output_dir)
    # Honest GNN axis: out-of-fold predictions that never trained on the segment's
    # own LLM label (persisted by `qra gnn train`). Empty -> fall back to the
    # in-sample distillation overlay.
    from gnn_layer.classifier.validation import read_heldout_predictions
    gnn_heldout = read_heldout_predictions(output_dir) if have_segments else {}

    # --- Family 1: Human ↔ Human ---
    human_human: Dict[str, dict] = {}
    for ws in sorted(rater_codes):
        roster = roster_by_ws.get(ws) or sorted(
            {r for it in rater_codes[ws].values() for r in it}
        )
        items = rater_codes[ws]
        human_human[str(ws)] = {
            'raters': roster,
            'n_items': len(items),
            'primary': _human_human_for_worksheet(roster, items, 'primary'),
            'secondary': _human_human_for_worksheet(roster, items, 'secondary'),
        }

    # --- Families 2 & 3 ---
    consensus = _consensus_rows(codes)
    # Every resolved test-set segment (incl. unresolved-by-humans) — the LLM sample
    # the GNN axes are scored against.
    all_seg_ids = sorted({c['segment_id'] for c in codes
                          if c['is_consensus'] and c['segment_id']})
    if have_segments:
        human_vs_llm = _human_vs_llm(consensus, by_id, id_to_short)
        human_vs_llm['per_llm_rater'] = _human_vs_llm_raters(consensus, by_id)
        human_vs_gnn = _gnn_full(consensus, all_seg_ids, by_id, gnn_heldout, id_to_short)
        discrepancies = _discrepancies(consensus, rater_codes, by_id, gnn_heldout, id_to_short)
        item_details = _item_details(codes, rater_codes, by_id, gnn_heldout, id_to_short)
    else:
        note = 'no frozen segments / machine labels available'
        human_vs_llm = {'n': 0, 'note': note, 'per_llm_rater': {}}
        human_vs_gnn = {'n': 0, 'note': note}
        discrepancies = []
        item_details = _item_details(codes, rater_codes, {}, {}, id_to_short)

    results = {
        'framework': 'vaamr',
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'machine_signature': machine_signature(output_dir),
        'statistics': {
            'cohen_kappa': 'scikit-learn cohen_kappa_score',
            'fleiss_kappa': 'statsmodels fleiss_kappa (complete-case)',
            'krippendorff_alpha': 'krippendorff package (nominal)',
        },
        'have_machine_labels': have_segments,
        'testset_drift': drift,
        'gnn_axis': human_vs_gnn.get('gnn_source', 'distillation'),
        'n_consensus_usable': len(consensus),
        'human_human': human_human,
        'human_vs_llm': human_vs_llm,
        'human_vs_gnn': human_vs_gnn,
        'n_discrepancies': len(discrepancies),
    }

    _write_outputs(output_dir, results, discrepancies, item_details)
    if verbose:
        gnn_axis = results['gnn_axis']
        print(f"IRR analysis complete: {len(consensus)} consensus items, "
              f"{len(discrepancies)} discrepancies (GNN axis: {gnn_axis}).")
    results['_discrepancies'] = discrepancies
    results['_item_details'] = item_details
    return results


def load_irr_metrics(output_dir: str) -> dict:
    """Aggregate the already-computed reliability numbers for the add-data mode picker.

    Reads (best-effort, no recomputation) the persisted gate verdicts + IRR results and
    returns a compact comparison the TUI renders so the user can choose a classification
    mode for new data:

      probe (probe_gate.json)  → probe↔human / probe↔LLM κ, per-stage recall, ready
      gnn   (gnn_gate.json + irr_results.json) → gnn↔LLM / gnn↔human κ, ready
      llm   (irr_results.json) → LLM↔human κ (the human-level reference) + human↔human band

    Missing sources yield empty sub-dicts; the caller degrades gracefully.
    """
    out: dict = {'probe': {}, 'gnn': {}, 'llm': {}}
    try:
        from classification_tools.probe.probe_classifier import read_probe_gate
        pg = read_probe_gate(output_dir) or {}
        if pg:
            out['probe'] = {
                'human_kappa': pg.get('probe_human_kappa'),
                'llm_kappa': pg.get('probe_llm_kappa'),
                'ready': bool(pg.get('ready_for_scaling')),
                'per_class_recall': pg.get('per_class_recall') or {},
                'mode': pg.get('mode'),
            }
    except Exception:
        pass
    try:
        from gnn_layer.classifier.validation import read_gate_verdict
        gg = read_gate_verdict(output_dir) or {}
        if gg:
            out['gnn']['llm_kappa'] = gg.get('vaamr_kappa')
            out['gnn']['ready'] = bool(gg.get('ready_for_scaling'))
    except Exception:
        pass
    try:
        path = os.path.join(_paths.irr_validation_dir(output_dir), 'irr_results.json')
        if os.path.isfile(path):
            with open(path, encoding='utf-8') as f:
                irr = json.load(f)
            out['llm']['human_kappa'] = (irr.get('human_vs_llm') or {}).get('cohen_kappa')
            gh = (irr.get('human_vs_gnn') or {}).get('cohen_kappa')
            if gh is not None:
                out['gnn']['human_kappa'] = gh
            alphas = []
            for ws in (irr.get('human_human') or {}).values():
                a = ((ws.get('primary') or {}).get('krippendorff_alpha'))
                if isinstance(a, (int, float)):
                    alphas.append(a)
            if alphas:
                out['llm']['human_human_band'] = [min(alphas), max(alphas)]
    except Exception:
        pass
    return out


def _write_outputs(output_dir: str, results: dict, discrepancies: List[dict],
                   item_details: List[dict]) -> None:
    out_dir = _paths.irr_validation_dir(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'irr_results.json'), 'w', encoding='utf-8') as f:
        json.dump({k: v for k, v in results.items() if not k.startswith('_')},
                  f, indent=2)

    pairwise_path = os.path.join(out_dir, 'irr_pairwise.csv')
    with open(pairwise_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['comparison', 'worksheet_n', 'field', 'rater_a', 'rater_b',
                    'n', 'cohen_kappa', 'percent_agreement'])
        for ws, hh in results['human_human'].items():
            for field in ('primary', 'secondary'):
                block = hh.get(field)
                if not block:
                    continue
                for pr in block['pairwise']:
                    w.writerow(['human_human', ws, field, pr['rater_a'], pr['rater_b'],
                                pr['n'], _fmt(pr['cohen_kappa']), _fmt(pr['percent_agreement'])])
        for comp, key in (('human_vs_llm', 'human_vs_llm'), ('human_vs_gnn', 'human_vs_gnn')):
            block = results.get(key, {})
            if block.get('n'):
                w.writerow([comp, 'ALL', 'primary', 'human_consensus', 'machine',
                            block['n'], _fmt(block.get('cohen_kappa')),
                            _fmt(block.get('percent_agreement'))])
            for ws, sub in block.get('per_worksheet', {}).items():
                w.writerow([comp, ws, 'primary', 'human_consensus', 'machine',
                            sub['n'], _fmt(sub.get('cohen_kappa')),
                            _fmt(sub.get('percent_agreement'))])
        for rid, sub in results.get('human_vs_llm', {}).get('per_llm_rater', {}).items():
            w.writerow(['human_vs_llm_rater', 'ALL', 'primary', 'human_consensus', rid,
                        sub['n'], _fmt(sub.get('cohen_kappa')),
                        _fmt(sub.get('percent_agreement'))])
        # Both GNN axes, each vs human and vs LLM.
        gnn = results.get('human_vs_gnn', {})
        for axis in ('heldout', 'distillation'):
            blk = gnn.get(axis) or {}
            vh, vl = blk.get('vs_human') or {}, blk.get('vs_llm') or {}
            if vh.get('n'):
                w.writerow([f'gnn_{axis}_vs_human', 'ALL', 'primary', 'human_consensus',
                            'gnn', vh['n'], _fmt(vh.get('cohen_kappa')),
                            _fmt(vh.get('percent_agreement'))])
            if vl.get('n'):
                w.writerow([f'gnn_{axis}_vs_llm', 'ALL', 'primary', 'llm_consensus',
                            'gnn', vl['n'], _fmt(vl.get('cohen_kappa')),
                            _fmt(vl.get('percent_agreement'))])

    disc_path = os.path.join(out_dir, 'irr_discrepancies.csv')
    fieldnames = ['worksheet_n', 'item_num', 'segment_id', 'human_consensus',
                  'consensus_source', 'rater_codes', 'llm_label', 'llm_confidence',
                  'gnn_label', 'gnn_confidence', 'disagrees_llm', 'disagrees_gnn', 'text']
    with open(disc_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in discrepancies:
            w.writerow(row)

    # irr_item_detail.csv — every item, all substrates flattened (one row/item).
    item_path = os.path.join(out_dir, 'irr_item_detail.csv')
    item_fields = ['worksheet_n', 'item_num', 'segment_id', 'human_consensus',
                   'human_source', 'human_notes', 'human_raters',
                   'llm_consensus', 'llm_justification', 'llm_raters',
                   'gnn_prediction', 'gnn_source', 'gnn_confidence',
                   'llm_gnn_agree', 'text']
    with open(item_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=item_fields)
        w.writeheader()
        for it in item_details:
            human_raters = '; '.join(
                f"{r}={v['primary']}" + (f"/{v['secondary']}" if v['secondary'] else '')
                for r, v in it['human']['raters'].items()
            )
            llm_raters = '; '.join(
                f"{rv['rater']}={rv['label']}" for rv in it['llm']['raters']
            )
            w.writerow({
                'worksheet_n': it['worksheet_n'],
                'item_num': it['item_num'],
                'segment_id': it['segment_id'],
                'human_consensus': it['human']['consensus'],
                'human_source': it['human']['source'],
                'human_notes': it['human']['notes'],
                'human_raters': human_raters,
                'llm_consensus': it['llm']['consensus'],
                'llm_justification': it['llm']['justification'],
                'llm_raters': llm_raters,
                'gnn_prediction': it['gnn']['prediction'],
                'gnn_source': it['gnn']['source'],
                'gnn_confidence': ('' if it['gnn']['confidence'] is None
                                   else f"{it['gnn']['confidence']:.4f}"),
                'llm_gnn_agree': it['llm_gnn_consensus']['agree'],
                'text': it['text'].replace('\n', ' '),
            })


def _fmt(v) -> str:
    return '' if v is None else f"{v:.4f}"
