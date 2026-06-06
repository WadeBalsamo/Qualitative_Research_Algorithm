"""
gnn_layer/influence.py
----------------------
Track B3/B4 — model-counterfactual therapist→participant influence + triangulation.

The observed Δprogression of cue blocks (analysis/mechanism.py) is the LEAD evidence for
"what therapist language progresses participants" — a more direct, more controlled measure
than any learned edge weight. This module supplies the GNN's genuinely-additive,
*secondary* contribution: a **model-counterfactual sensitivity** lens that captures the
context-dependent / nonlinear influence the additive Δprogression tables cannot.

The counterfactual
------------------
For each cue block (FROM participant → therapist cue → TO participant) the trained model
already predicts the TO participant's VAAMR mixture from the graph. We ask: had the
therapist deployed a *different* PURER move there, how would the model's prediction for the
TO participant shift? We answer it by **swapping the therapist cue node feature** with each
PURER move's centroid embedding (and a neutral/null baseline), re-running the forward pass,
and reading the change in the TO participant's predicted progression coordinate (E[stage]).

    influence(move m) = E_blocks[ prog(TO | cue := centroid_m) − prog(TO | cue := null) ]

aggregated per move (and per from_stage × move) with participant-clustered bootstrap CIs.
This is sensitivity analysis of the model, **NOT causation** (n≈32 observational + the
elicitation confound — methodology §9.2/§9.4). It is GATED: the runner only invokes it from
a gate-passing model, and B4 triangulates it against mechanism.py — the GNN is positioned
primary only where it passes the gate AND converges with the observed signal.

GPU (D11): the per-block re-forwards are the GPU-relevant cost — they run on the model's
device (via inference._graph_tensors_on_model_device), and the block count is capped with a
logged note (never silently truncated).
"""

import os
from typing import Dict, List, Optional

from process import output_paths as _paths

# PURER move display names (mirror analysis.mechanism / validation).
PURER_NAMES = {0: 'Phenomenology', 1: 'Utilization', 2: 'Reframing',
               3: 'Education', 4: 'Reinforcement'}
N_PURER = 5


def purer_centroids(graph, df_all):
    """Per-PURER-move centroid of therapist segment INPUT features + a neutral null baseline.

    Returns (centroids, null_vec, counts):
      centroids : {move:int -> np.ndarray[D]}  mean input feature of therapist nodes with that move
      null_vec  : np.ndarray[D]                mean input feature across ALL therapist nodes
      counts    : {move:int -> int}            therapist nodes contributing to each centroid
    Moves with no therapist support are absent from ``centroids``.
    """
    import numpy as np

    X = graph.x.detach().cpu().numpy()
    purer_of: Dict[str, int] = {}
    for _, r in df_all.iterrows():
        if str(r.get('speaker', '')) != 'therapist':
            continue
        try:
            purer_of[str(r.get('segment_id'))] = int(r.get('purer_primary'))
        except (ValueError, TypeError):
            continue

    by_move: Dict[int, List[int]] = {}
    ther_rows: List[int] = []
    for sid, gi in graph.index_of.items():
        if gi >= X.shape[0]:
            continue
        if graph.node_types[gi] != 'therapist_segment':
            continue
        ther_rows.append(gi)
        m = purer_of.get(str(sid))
        if m is not None and 0 <= m < N_PURER:
            by_move.setdefault(m, []).append(gi)

    centroids = {m: X[idxs].mean(axis=0) for m, idxs in by_move.items() if idxs}
    counts = {m: len(idxs) for m, idxs in by_move.items()}
    null_vec = X[ther_rows].mean(axis=0) if ther_rows else X.mean(axis=0)
    return centroids, null_vec.astype('float32'), counts


def _forward_progression(model, graph, config, x_override=None):
    """Forward pass → per-node progression coordinate E[stage]; optional input override.

    ``x_override`` (when given) replaces graph.x. Returns np.ndarray[N] of progression
    coordinates (Σ k·p_k over the calibrated soft-VAAMR mixture). Runs on the model device.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .inference import _graph_tensors_on_model_device
    from .calibration import apply_temperature

    _x, _ei, _ew, _eti = _graph_tensors_on_model_device(model, graph)
    if x_override is not None:
        _x = torch.as_tensor(x_override, dtype=_x.dtype, device=_x.device)
    model.eval()
    with torch.no_grad():
        out = model(_x, _ei, _ew, _eti)
    if 'soft_vaamr' not in out:
        return None
    _T = getattr(config, 'calibration_temperature', None) if config is not None else None
    mix = F.softmax(apply_temperature(out['soft_vaamr'], _T), dim=1).cpu().numpy()
    return (mix * np.arange(mix.shape[1])).sum(axis=1)


def counterfactual_influence(model, graph, df_all, config,
                             blocks: Optional[List[dict]] = None) -> dict:
    """Model-counterfactual influence of each PURER move on the TO participant's progression.

    For every mediated cue block, swap its therapist node features with each PURER move
    centroid (and the null baseline), re-forward, and record the shift in the TO
    participant's predicted progression coordinate relative to the null. Aggregated per
    move and per (from_stage, move) with participant-clustered bootstrap CIs.

    Returns a dict with: ``per_move``, ``per_stage_move``, ``n_blocks``, ``n_capped``,
    ``move_support``, plus block-level rows under ``rows`` for downstream subgroup analysis.
    Returns ``{'status': ...}`` when prerequisites are missing.
    """
    import numpy as np
    from analysis import stats as S
    from .inference import build_cue_blocks_with_segments

    centroids, null_vec, counts = purer_centroids(graph, df_all)
    if not centroids:
        return {'status': 'skipped: no PURER-labelled therapist segments for centroids'}

    if blocks is None:
        blocks = build_cue_blocks_with_segments(df_all)
    blocks = [b for b in blocks if b.get('therapist_seg_ids') and b.get('to_seg_id') in graph.index_of]
    if not blocks:
        return {'status': 'skipped: no mediated cue blocks in graph'}

    # Cap blocks (logged, never silent) so the per-block re-forwards stay bounded.
    cap = getattr(config, 'counterfactual_max_blocks', None)
    n_capped = 0
    if cap is not None and len(blocks) > int(cap):
        n_capped = len(blocks) - int(cap)
        blocks = blocks[:int(cap)]

    pid_of = {}
    for _, r in df_all.iterrows():
        pid_of[str(r.get('segment_id'))] = r.get('participant_id')

    base_X = graph.x.detach().cpu().numpy()
    moves = sorted(centroids.keys())

    # Null-baseline forward once: replace EVERY block's therapist nodes with the neutral
    # cue. (Each block is evaluated against the same null reference so influences are
    # comparable across blocks; we recompute per block to isolate that block's TO node.)
    rows = []  # one row per (block, move): influence on TO progression coordinate
    for b in blocks:
        to_idx = graph.index_of.get(b['to_seg_id'])
        if to_idx is None:
            continue
        th_idx = [graph.index_of[s] for s in b['therapist_seg_ids'] if s in graph.index_of]
        if not th_idx:
            continue
        # Null reference: this block's therapist nodes set to the neutral cue.
        x_null = base_X.copy()
        x_null[th_idx] = null_vec
        prog_null = _forward_progression(model, graph, config, x_override=x_null)
        if prog_null is None:
            return {'status': 'skipped: model has no soft_vaamr head'}
        ref = float(prog_null[to_idx])
        for m in moves:
            x_cf = base_X.copy()
            x_cf[th_idx] = centroids[m]
            prog_cf = _forward_progression(model, graph, config, x_override=x_cf)
            infl = float(prog_cf[to_idx]) - ref
            rows.append({
                'from_seg_id': b['from_seg_id'], 'to_seg_id': b['to_seg_id'],
                'from_stage': int(b.get('from_stage', 0)),
                'participant_id': pid_of.get(b['from_seg_id']),
                'session_id': b.get('session_id'),
                'move': int(m), 'move_name': PURER_NAMES.get(int(m), str(m)),
                'influence': round(infl, 5),
            })

    # ---- aggregate per move (participant-clustered bootstrap CI) ----
    per_move = []
    for m in moves:
        mrows = [r for r in rows if r['move'] == m]
        if not mrows:
            continue
        vals = [r['influence'] for r in mrows]
        clusters = [r['participant_id'] for r in mrows]
        ci = S.cluster_bootstrap_ci(vals, clusters, statistic=np.mean,
                                    n_boot=int(getattr(config, 'influence_bootstrap_n', 1000)))
        per_move.append({
            'move': int(m), 'move_name': PURER_NAMES.get(int(m), str(m)),
            'n_blocks': len(mrows), 'n_participants': ci.get('n_clusters', 0),
            'centroid_support': counts.get(m, 0),
            'mean_influence': round(float(np.mean(vals)), 5),
            'ci_lo': round(ci['lo'], 5) if ci['lo'] == ci['lo'] else None,
            'ci_hi': round(ci['hi'], 5) if ci['hi'] == ci['hi'] else None,
        })
    per_move.sort(key=lambda r: -r['mean_influence'])

    # ---- aggregate per (from_stage, move) ----
    per_stage_move = []
    by_cell: Dict[tuple, List[float]] = {}
    for r in rows:
        by_cell.setdefault((r['from_stage'], r['move']), []).append(r['influence'])
    for (stage, m), vals in sorted(by_cell.items()):
        if len(vals) < 2:
            continue
        per_stage_move.append({
            'from_stage': int(stage), 'move': int(m),
            'move_name': PURER_NAMES.get(int(m), str(m)),
            'n_blocks': len(vals), 'mean_influence': round(float(np.mean(vals)), 5),
        })
    per_stage_move.sort(key=lambda r: (r['from_stage'], -r['mean_influence']))

    return {
        'per_move': per_move,
        'per_stage_move': per_stage_move,
        'rows': rows,
        'n_blocks': len({(r['from_seg_id'], r['to_seg_id']) for r in rows}),
        'n_capped': n_capped,
        'move_support': counts,
    }


# ---------------------------------------------------------------------------
# B4 — triangulation with the observed Δprogression signal (mechanism.py LEAD)
# ---------------------------------------------------------------------------

def _observed_per_move(output_dir: str) -> Dict[int, float]:
    """Observed mean Δprogression per PURER move from mechanism_delta_progression.csv.

    The behaviour column there reads e.g. 'Reframing(2)'; we parse the trailing id.
    Returns {move:int -> mean observed Δprogression} (averaged across from_stage cells).
    """
    import re
    import pandas as pd

    path = os.path.join(_paths.mechanism_dir(output_dir), 'mechanism_delta_progression.csv')
    if not os.path.isfile(path):
        return {}
    try:
        mdf = pd.read_csv(path)
    except Exception:
        return {}
    if 'grouping' not in mdf.columns or 'behavior' not in mdf.columns:
        return {}
    mdf = mdf[mdf['grouping'] == 'purer']
    out: Dict[int, List[float]] = {}
    for _, r in mdf.iterrows():
        mobj = re.search(r'\((\d+)\)\s*$', str(r['behavior']))
        if not mobj:
            continue
        m = int(mobj.group(1))
        try:
            out.setdefault(m, []).append(float(r['mean_delta_prog']))
        except (ValueError, TypeError):
            continue
    return {m: sum(v) / len(v) for m, v in out.items() if v}


def triangulate(influence_result: dict, output_dir: str) -> Optional[dict]:
    """Align the counterfactual influence ranking with mechanism.py's observed Δprogression.

    Returns {n_moves, spearman, sign_agreement, per_move:[...]} or None if the observed
    table is unavailable. ``per_move`` carries both signals + a convergence flag so
    divergences (model says X, observed says not-X) are surfaced for human review.

    Backward-compatible. Additively carries ``cue_transition`` — the PRE-REGISTERED
    §1A success metric (Spearman ρ on per-(from_stage × move) cells with a participant-
    clustered bootstrap CI + FDR-restricted sign agreement; see ``triangulation_metric``).
    The legacy per-move keys above are kept untouched for existing callers.
    """
    observed = _observed_per_move(output_dir)
    if not observed:
        return None
    per_move = {r['move']: r['mean_influence'] for r in influence_result.get('per_move', [])}
    common = sorted(set(observed) & set(per_move))
    if len(common) < 2:
        return _attach_cue(
            {'n_moves': len(common), 'spearman': None, 'sign_agreement': None,
             'per_move': [{'move': m, 'move_name': PURER_NAMES.get(m, str(m)),
                           'observed_delta': round(observed.get(m, float('nan')), 5),
                           'counterfactual_influence': round(per_move.get(m, float('nan')), 5),
                           'converges': None} for m in common]},
            influence_result, output_dir)
    import numpy as np
    try:
        from scipy import stats as _sps
        rho = float(_sps.spearmanr([observed[m] for m in common],
                                   [per_move[m] for m in common]).correlation)
    except Exception:
        rho = None
    sign_agree = sum(1 for m in common
                     if (observed[m] >= 0) == (per_move[m] >= 0)) / len(common)
    rows = []
    for m in common:
        conv = (observed[m] >= 0) == (per_move[m] >= 0)
        rows.append({
            'move': m, 'move_name': PURER_NAMES.get(m, str(m)),
            'observed_delta': round(observed[m], 5),
            'counterfactual_influence': round(per_move[m], 5),
            'converges': bool(conv),
        })
    rows.sort(key=lambda r: -r['counterfactual_influence'])
    return _attach_cue(
        {'n_moves': len(common),
         'spearman': round(rho, 4) if rho is not None and rho == rho else None,
         'sign_agreement': round(sign_agree, 4),
         'per_move': rows},
        influence_result, output_dir)


# ---------------------------------------------------------------------------
# B4 (refined) — PRE-REGISTERED cue→transition triangulation (design_decisions §1A/§9)
#
# Unit of analysis = the cue→transition = (from_stage × PURER move). Two INDEPENDENT
# estimates of "does this therapist move progress the participant?":
#   • GNN counterfactual influence  — per (from_stage, move), from counterfactual_influence
#   • Observed Δprogression (LEAD)  — per (from_stage, move), from analysis/mechanism.py's
#     mechanism_delta_progression.csv (mean_delta_prog + the fdr_significant flag)
# The KEY join is (from_stage:int, move:int): mechanism.py's `behavior` is `Name(id)`
# (move parsed from the trailing id) over the SAME PURER id-space and the SAME block
# `from_stage` as influence.py — so the cells align. They measure conceptually different
# things (observed-when-the-move-was-used vs counterfactual-if-the-cue-were-swapped),
# which is precisely why their convergence is evidence.
#
# SUCCESS (§1A, pre-registered): Spearman ρ>0 with a participant-clustered bootstrap 95%
# CI excluding 0, AND ≥70% sign agreement on the FDR-significant cells — reported beside
# the GNN reliability-gate κ (the trust context). Fail ⇒ mechanism.py LEADS, GNN exploratory.
# ---------------------------------------------------------------------------

def _as_bool(v) -> bool:
    """Coerce a CSV/DataFrame cell to bool, tolerant of 'True'/'False' strings + NaN."""
    import pandas as pd
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(v, str):
        return v.strip().lower() in ('true', '1', 'yes', 't')
    try:
        return bool(int(v))
    except (ValueError, TypeError):
        return bool(v)


def _parse_observed_stage_move(mdf) -> Dict[tuple, dict]:
    """Parse mechanism.py's Δprogression table → {(from_stage, move): {...}} for grouping=purer.

    Reads the EXACT mechanism.py schema: rows where ``grouping == 'purer'`` carry a
    ``behavior`` of the form 'Reframing(2)' (move id = trailing parenthesized int), an int
    ``from_stage``, the signed effect ``mean_delta_prog``, and the ``fdr_significant`` flag
    (with ``fdr_q`` when present). Motif rows and unparseable behaviours are skipped.
    """
    import re
    import pandas as pd

    out: Dict[tuple, dict] = {}
    if mdf is None or 'grouping' not in mdf.columns or 'behavior' not in mdf.columns:
        return out
    sub = mdf[mdf['grouping'] == 'purer']
    has_fdr = 'fdr_significant' in sub.columns
    has_q = 'fdr_q' in sub.columns
    for _, r in sub.iterrows():
        mobj = re.search(r'\((\d+)\)\s*$', str(r['behavior']))
        if not mobj:
            continue
        move = int(mobj.group(1))
        try:
            stage = int(r['from_stage'])
            delta = float(r['mean_delta_prog'])
        except (ValueError, TypeError, KeyError):
            continue
        if not (delta == delta):                       # NaN effect → unusable cell
            continue
        out[(stage, move)] = {
            'mean_delta': delta,
            'fdr_significant': _as_bool(r['fdr_significant']) if has_fdr else False,
            'fdr_q': (float(r['fdr_q']) if has_q and pd.notna(r['fdr_q']) else None),
        }
    return out


def _observed_per_stage_move(output_dir: str) -> Dict[tuple, dict]:
    """Per-(from_stage, move) observed Δprogression from the default mechanism CSV path."""
    import pandas as pd
    path = os.path.join(_paths.mechanism_dir(output_dir), 'mechanism_delta_progression.csv')
    if not os.path.isfile(path):
        return {}
    try:
        return _parse_observed_stage_move(pd.read_csv(path))
    except Exception:
        return {}


def _read_observed_csv(path: str) -> Dict[tuple, dict]:
    """Per-(from_stage, move) observed Δprogression from an EXPLICIT mechanism CSV path."""
    import pandas as pd
    if not path or not os.path.isfile(path):
        return {}
    try:
        return _parse_observed_stage_move(pd.read_csv(path))
    except Exception:
        return {}


def _spearman(a, b) -> float:
    """Spearman ρ(a, b); NaN if undefined (constant input / <2 points / no scipy)."""
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    try:
        from scipy import stats as _sps
        res = _sps.spearmanr(a, b)
        rho = getattr(res, 'correlation', None)
        if rho is None:
            rho = getattr(res, 'statistic', float('nan'))
        return float(rho)
    except Exception:
        return float('nan')


def _cells_detail(common: List[tuple], observed: Dict[tuple, dict],
                  gnn_cell: Dict[tuple, float]) -> List[dict]:
    """Per-cell side-by-side of observed Δprog vs GNN counterfactual + sign match / FDR flag."""
    rows = []
    for (stage, m) in common:
        od = float(observed[(stage, m)]['mean_delta'])
        cf = float(gnn_cell[(stage, m)])
        rows.append({
            'from_stage': int(stage), 'move': int(m),
            'move_name': PURER_NAMES.get(int(m), str(m)),
            'observed_delta': round(od, 5),
            'counterfactual_influence': round(cf, 5),
            'fdr_significant': bool(observed[(stage, m)].get('fdr_significant')),
            'sign_match': bool((od >= 0) == (cf >= 0)),
        })
    rows.sort(key=lambda r: (r['from_stage'], -r['counterfactual_influence']))
    return rows


def _bootstrap_rho_ci(rows: List[dict], common: List[tuple], obs_vec: Dict[tuple, float],
                      point_rho: float, *, n_boot: int, seed: int) -> dict:
    """Participant-clustered bootstrap 95% CI on the Spearman ρ between the two cell vectors.

    REUSES ``analysis.stats.cluster_bootstrap_ci`` as the resampling engine: participants
    are the clusters; the per-resample statistic recomputes the GNN per-(from_stage, move)
    mean influence over the resampled participants' cue blocks and Spearman-correlates it
    against the FIXED observed Δprogression vector (the LEAD reference). The observed vector
    is held fixed because mechanism.py exposes only aggregated per-cell effects — so the
    bootstrap tests how stable the GNN↔observed alignment is to the GNN's participant sample.

    Degenerate resamples (<2 surviving common cells / zero variance ⇒ undefined ρ) fall back
    to ``point_rho`` so a single bad draw cannot poison the whole percentile CI; such draws
    are astronomically rare with realistic support. Returns {lo, hi, n_clusters} (NaN CI when
    there are <2 participants or no block-level rows to resample).
    """
    import numpy as np
    from analysis import stats as S

    common_set = set(common)
    blk = []          # (cell_key, influence) for block rows that fall in a common cell
    clusters = []     # participant per retained block row (the bootstrap cluster)
    for r in rows or []:
        try:
            k = (int(r['from_stage']), int(r['move']))
        except (ValueError, TypeError, KeyError):
            continue
        if k in common_set:
            blk.append((k, float(r.get('influence', 0.0))))
            clusters.append(r.get('participant_id'))
    if not blk:
        return {'lo': float('nan'), 'hi': float('nan'), 'n_clusters': 0}

    idx_values = np.arange(len(blk), dtype=float)

    def _rho(sample_idx):
        cell_vals: Dict[tuple, List[float]] = {}
        for j in sample_idx.astype(int):
            k, v = blk[j]
            cell_vals.setdefault(k, []).append(v)
        cells = sorted(cell_vals)
        if len(cells) < 2:
            return point_rho                       # degenerate resample → point fallback
        g = [float(np.mean(cell_vals[k])) for k in cells]
        o = [obs_vec[k] for k in cells]
        rho = _spearman(o, g)
        return rho if rho == rho else point_rho

    ci = S.cluster_bootstrap_ci(idx_values, clusters, statistic=_rho,
                                n_boot=int(n_boot), seed=int(seed))
    return {'lo': ci['lo'], 'hi': ci['hi'], 'n_clusters': int(ci.get('n_clusters', 0))}


def triangulation_metric(influence_result: dict, observed: Dict[tuple, dict], *,
                         sign_threshold: float = 0.70, n_boot: int = 1000,
                         seed: int = 42) -> dict:
    """PRE-REGISTERED (§1A) cue→transition triangulation metric — the peer-review success test.

    ``observed`` is the per-(from_stage, move) observed Δprogression mapping
    ``{(from_stage:int, move:int): {'mean_delta': float, 'fdr_significant': bool, ...}}``
    (from ``_observed_per_stage_move`` / ``_read_observed_csv``). The GNN side is read from
    ``influence_result['per_stage_move']`` (cell means) + ``['rows']`` (block-level, for the
    bootstrap). Computes, over the common cells:
      • spearman_rho + a participant-clustered bootstrap 95% CI [ci_lo, ci_hi] → ci_excludes_zero
      • sign_agreement RESTRICTED to the FDR-significant cells (+ n_fdr_significant used)
      • converges = (rho>0 AND ci_excludes_zero AND sign_agreement >= sign_threshold)
    Returns all components + a per-cell detail table + the non-causal caveat.
    """
    gnn_cell = {(int(r['from_stage']), int(r['move'])): float(r['mean_influence'])
                for r in influence_result.get('per_stage_move', [])}
    obs_vec = {k: float(observed[k]['mean_delta']) for k in observed}
    common = sorted(set(observed) & set(gnn_cell))

    out = {
        'unit': 'cue_transition (from_stage × PURER move)',
        'n_cells': len(common),
        'spearman_rho': None,
        'ci_lo': None, 'ci_hi': None, 'ci_excludes_zero': False,
        'n_boot': 0, 'n_participants': 0,
        'sign_agreement': None, 'n_fdr_significant': 0,
        'sign_agreement_threshold': float(sign_threshold),
        'converges': False,
        'per_cell': _cells_detail(common, observed, gnn_cell),
        'caveat': ('Model sensitivity, NOT causation: n≈32 observational + the elicitation '
                   'confound (methodology §9.2/§9.4). Convergence with the observed Δprogression '
                   'is corroboration, not proof — report beside the GNN reliability-gate κ.'),
    }
    if len(common) < 2:
        return out                                  # ρ undefined with <2 paired cells

    rho = _spearman([obs_vec[k] for k in common], [gnn_cell[k] for k in common])
    rho = rho if rho == rho else None

    boot = _bootstrap_rho_ci(influence_result.get('rows') or [], common, obs_vec,
                             point_rho=(rho if rho is not None else 0.0),
                             n_boot=n_boot, seed=seed)
    lo, hi = boot['lo'], boot['hi']
    ci_excludes_zero = bool(lo == lo and hi == hi and (lo > 0 or hi < 0))

    # Sign agreement is RESTRICTED to the cells mechanism.py flags FDR-significant.
    fdr_cells = [k for k in common if bool(observed[k].get('fdr_significant'))]
    n_fdr = len(fdr_cells)
    sign_agreement = (sum(1 for k in fdr_cells if (obs_vec[k] >= 0) == (gnn_cell[k] >= 0)) / n_fdr
                      if n_fdr else None)

    converges = bool(rho is not None and rho > 0 and ci_excludes_zero
                     and sign_agreement is not None and sign_agreement >= sign_threshold)

    out.update({
        'spearman_rho': round(rho, 4) if rho is not None else None,
        'ci_lo': round(lo, 4) if lo == lo else None,
        'ci_hi': round(hi, 4) if hi == hi else None,
        'ci_excludes_zero': ci_excludes_zero,
        'n_boot': int(n_boot),
        'n_participants': int(boot.get('n_clusters', 0)),
        'sign_agreement': round(sign_agreement, 4) if sign_agreement is not None else None,
        'n_fdr_significant': n_fdr,
        'converges': converges,
    })
    return out


def triangulate_v2(influence_result: dict, output_dir: str, *,
                   sign_threshold: float = 0.70, n_boot: int = 1000,
                   seed: int = 42) -> Optional[dict]:
    """``triangulation_metric`` over the default-path mechanism CSV; None if it is absent."""
    observed = _observed_per_stage_move(output_dir)
    if not observed:
        return None
    return triangulation_metric(influence_result, observed,
                                sign_threshold=sign_threshold, n_boot=n_boot, seed=seed)


def _attach_cue(result: Optional[dict], influence_result: dict, output_dir: str):
    """Additively attach the pre-registered ``cue_transition`` metric to a legacy triangulate dict."""
    if not isinstance(result, dict):
        return result
    try:
        result['cue_transition'] = triangulate_v2(influence_result, output_dir)
    except Exception:
        result['cue_transition'] = None
    return result


def run_counterfactual_experiment(model, graph, df_all, config, gate_kappa=None,
                                  mechanism_csv: Optional[str] = None) -> dict:
    """Run the counterfactual influence + §1A triangulation on a GIVEN model+graph — UN-GATED.

    The runner hard-gates the counterfactual on the legacy κ≥0.70 reliability gate, which is
    unreachable in principle (design_decisions §3) and would permanently veto the PRIMARY
    mechanism deliverable. This experiment entry point lets the architect run the readout on
    the best Qwen model REGARDLESS of that gate; the gate κ is echoed back as the reported
    TRUST CONTEXT (low κ ⇒ treat as exploratory and weight the triangulation accordingly).
    Reuses ``counterfactual_influence`` / ``purer_centroids`` verbatim — no swap logic is
    re-implemented here.

    Returns {influence, triangulation, gate_kappa, [status]}:
      influence     — the full counterfactual_influence dict (per_move/per_stage_move/rows)
      triangulation — the ``triangulation_metric`` dict, or None when no observed Δprogression
                      table is supplied (pass ``mechanism_csv`` to enable it)
      gate_kappa    — echoed unchanged as the trust context for the readout
    """
    infl = counterfactual_influence(model, graph, df_all, config)
    result = {'influence': infl, 'triangulation': None, 'gate_kappa': gate_kappa}
    if infl.get('status'):
        result['status'] = infl['status']
        return result
    observed = _read_observed_csv(mechanism_csv) if mechanism_csv else {}
    if observed:
        result['triangulation'] = triangulation_metric(
            infl, observed,
            sign_threshold=float(getattr(config, 'triangulation_sign_threshold', 0.70)),
            n_boot=int(getattr(config, 'influence_bootstrap_n', 1000)),
            seed=int(getattr(config, 'seed', 42)))
    return result


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_influence_csv(result: dict, output_dir: str) -> Optional[str]:
    """Per-move counterfactual influence table → 03_analysis_data/gnn/gnn_counterfactual_influence.csv."""
    import pandas as pd
    rows = result.get('per_move')
    if not rows:
        return None
    gnn_dir = _paths.gnn_data_dir(output_dir)
    os.makedirs(gnn_dir, exist_ok=True)
    path = os.path.join(gnn_dir, 'gnn_counterfactual_influence.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_influence_report(result: dict, tri: Optional[dict], output_dir: str) -> str:
    """Human-readable B3/B4 influence + triangulation report → 06_reports/06_gnn/influence.txt."""
    W = 78
    L = ["=" * W, "GNN MODEL-COUNTERFACTUAL INFLUENCE (B3) + TRIANGULATION (B4)", "=" * W, ""]
    if result.get('status'):
        L.append(f"  {result['status']}")
        L.append("")
    else:
        L.append("MODEL SENSITIVITY, NOT CAUSATION. For each cue block we swap the therapist")
        L.append("cue's node feature with each PURER move's centroid (vs a neutral baseline) and")
        L.append("measure the shift in the model's predicted progression coordinate (E[stage]) for")
        L.append("the FOLLOWING participant turn. This captures context-dependent influence the")
        L.append("additive Δprogression tables miss — but n≈32 observational + the elicitation")
        L.append("confound mean it is hypothesis-generating, never a causal effect (methodology §9.2).")
        L.append("The OBSERVED Δprogression (mechanism.py) remains the label of record; this is the")
        L.append("secondary, corroborating lens.")
        L.append("")
        L.append(f"  cue blocks scored: {result.get('n_blocks')}"
                 + (f"   (capped: {result['n_capped']} blocks dropped — raise "
                    f"counterfactual_max_blocks to include them)" if result.get('n_capped') else ""))
        L.append("")
        L.append("-" * W)
        L.append("1. COUNTERFACTUAL INFLUENCE BY PURER MOVE (Δ predicted progression vs neutral cue)")
        L.append("-" * W)
        L.append(f"  {'Move':<18}{'influence':>11}{'95% CI':>20}{'n blk/part':>14}")
        for r in result.get('per_move', []):
            ci = (f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]"
                  if r.get('ci_lo') is not None else "[ n/a ]")
            L.append(f"  {r['move_name']:<18}{r['mean_influence']:>+11.3f}{ci:>20}"
                     f"{str(r['n_blocks'])+'/'+str(r['n_participants']):>14}")
        L.append("")
        psm = result.get('per_stage_move') or []
        if psm:
            L.append("-" * W)
            L.append("2. INFLUENCE BY FROM-STAGE × MOVE (where each move has the most leverage)")
            L.append("-" * W)
            for r in psm[:20]:
                L.append(f"   from_stage {r['from_stage']}  {r['move_name']:<16} "
                         f"Δ={r['mean_influence']:+.3f} (n={r['n_blocks']})")
            L.append("")

    # B4 triangulation
    L.append("-" * W)
    L.append("3. TRIANGULATION vs OBSERVED Δprogression (mechanism.py — the LEAD signal)")
    L.append("-" * W)
    if not tri:
        L.append("  Observed Δprogression table not available (run analysis/mechanism.py first).")
        L.append("  Counterfactual influence is reported above WITHOUT corroboration — treat as")
        L.append("  exploratory until it can be aligned with the observed signal.")
    else:
        L.append(f"  Moves compared: {tri.get('n_moves')}   "
                 f"Spearman ρ(observed, counterfactual): "
                 f"{tri.get('spearman') if tri.get('spearman') is not None else 'n/a'}   "
                 f"sign agreement: "
                 f"{tri.get('sign_agreement') if tri.get('sign_agreement') is not None else 'n/a'}")
        L.append("")
        L.append(f"  {'Move':<18}{'observed Δ':>12}{'counterfactual':>16}{'converges':>11}")
        for r in tri.get('per_move', []):
            conv = '—' if r.get('converges') is None else ('YES' if r['converges'] else 'no')
            L.append(f"  {r['move_name']:<18}{r['observed_delta']:>+12.3f}"
                     f"{r['counterfactual_influence']:>+16.3f}{conv:>11}")
        L.append("")
        L.append("  Convergence across the two independent methods is stronger evidence than")
        L.append("  either alone. DIVERGENCES (converges=no) are flagged here for human review,")
        L.append("  never hidden — the GNN is positioned primary only where it converges.")
    L.append("")

    # B4 (refined): the PRE-REGISTERED §1A cue→transition success metric.
    ct = tri.get('cue_transition') if tri else None
    L.append("-" * W)
    L.append("4. PRE-REGISTERED CUE→TRANSITION TRIANGULATION (design_decisions §1A) — SUCCESS TEST")
    L.append("-" * W)
    if not ct:
        L.append("  Per-(from_stage × move) observed Δprogression table not available — the")
        L.append("  pre-registered metric needs analysis/mechanism.py to have run first.")
    else:
        rho = ct.get('spearman_rho')
        lo, hi = ct.get('ci_lo'), ct.get('ci_hi')
        sa = ct.get('sign_agreement')
        ci = (f"[{lo:+.3f}, {hi:+.3f}]" if lo is not None and hi is not None else "[ n/a ]")
        L.append(f"  Unit = {ct.get('unit')}   cells: {ct.get('n_cells')}"
                 f"   participants: {ct.get('n_participants')}")
        L.append(f"  Spearman ρ(observed Δprog, GNN counterfactual): "
                 f"{rho if rho is not None else 'n/a'}   95% CI {ci}"
                 f"   excludes 0: {'YES' if ct.get('ci_excludes_zero') else 'no'}")
        L.append(f"  Sign agreement on FDR-significant cells: "
                 f"{sa if sa is not None else 'n/a'} "
                 f"(n_FDR={ct.get('n_fdr_significant')}; threshold={ct.get('sign_agreement_threshold')})")
        L.append("")
        L.append(f"  VERDICT: {'CONVERGES' if ct.get('converges') else 'does NOT converge'} "
                 "— ρ>0 AND CI excludes 0 AND FDR-sign agreement ≥ threshold.")
        L.append("  Read this BESIDE the GNN reliability-gate κ (06_gnn/validation*.txt): low κ ⇒")
        L.append("  treat as exploratory and weight the triangulation. If it does NOT converge,")
        L.append("  mechanism.py LEADS and the GNN is reported as exploratory only (§1A).")
        if ct.get('per_cell'):
            L.append("")
            L.append(f"  {'from_stage':<11}{'move':<14}{'obs Δprog':>11}{'GNN cf':>10}{'FDR':>5}{'sign':>8}")
            for r in ct['per_cell']:
                L.append(f"  {r['from_stage']:<11}{str(r['move_name'])[:13]:<14}"
                         f"{r['observed_delta']:>+11.3f}{r['counterfactual_influence']:>+10.3f}"
                         f"{('*' if r['fdr_significant'] else ''):>5}"
                         f"{('match' if r['sign_match'] else 'DIFF'):>8}")
    L.append("")

    rep_dir = _paths.reports_gnn_dir(output_dir)
    os.makedirs(rep_dir, exist_ok=True)
    path = os.path.join(rep_dir, 'influence.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path


# ---------------------------------------------------------------------------
# B5 — context / subgroup sensitivity (only when N supports it)
# ---------------------------------------------------------------------------

def subgroup_influence(result: dict, df_all, output_dir: str,
                       min_blocks_per_cell: int = 8) -> Optional[str]:
    """Split counterfactual influence by session-number tertile (early/mid/late).

    Underpowered at n≈32: every cell carries an explicit flag and thin cells (<
    ``min_blocks_per_cell``) are dropped with a logged note. Writes a CSV sidecar; returns
    its path or None when there is nothing powered enough to report.
    """
    import numpy as np
    import pandas as pd

    rows = result.get('rows') or []
    if not rows:
        return None
    # session_number per session_id (from df_all).
    snum = {}
    if 'session_number' in df_all.columns:
        for _, r in df_all.iterrows():
            try:
                snum[str(r.get('session_id'))] = int(r.get('session_number'))
            except (ValueError, TypeError):
                continue
    if not snum:
        return None
    vals = sorted(set(snum.values()))
    if len(vals) < 3:
        return None
    lo_cut, hi_cut = np.percentile(vals, [33, 66])

    def _phase(sid):
        n = snum.get(str(sid))
        if n is None:
            return None
        return 'early' if n <= lo_cut else ('late' if n > hi_cut else 'mid')

    cells: Dict[tuple, List[float]] = {}
    for r in rows:
        ph = _phase(r.get('session_id'))
        if ph is None:
            continue
        cells.setdefault((ph, r['move']), []).append(r['influence'])

    out_rows, dropped = [], 0
    for (ph, m), v in sorted(cells.items()):
        if len(v) < min_blocks_per_cell:
            dropped += 1
            continue
        out_rows.append({'phase': ph, 'move': int(m),
                         'move_name': PURER_NAMES.get(int(m), str(m)),
                         'n_blocks': len(v), 'mean_influence': round(float(np.mean(v)), 5),
                         'underpowered': len(v) < 15})
    if dropped:
        print(f"  [gnn_layer] subgroup influence: dropped {dropped} thin cell(s) "
              f"(< {min_blocks_per_cell} blocks)")
    if not out_rows:
        return None
    gnn_dir = _paths.gnn_data_dir(output_dir)
    os.makedirs(gnn_dir, exist_ok=True)
    path = os.path.join(gnn_dir, 'gnn_counterfactual_influence_by_phase.csv')
    pd.DataFrame(out_rows).to_csv(path, index=False)
    return path
