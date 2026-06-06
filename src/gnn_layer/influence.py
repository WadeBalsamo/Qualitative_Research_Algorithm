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
    """
    observed = _observed_per_move(output_dir)
    if not observed:
        return None
    per_move = {r['move']: r['mean_influence'] for r in influence_result.get('per_move', [])}
    common = sorted(set(observed) & set(per_move))
    if len(common) < 2:
        return {'n_moves': len(common), 'spearman': None, 'sign_agreement': None,
                'per_move': [{'move': m, 'move_name': PURER_NAMES.get(m, str(m)),
                              'observed_delta': round(observed.get(m, float('nan')), 5),
                              'counterfactual_influence': round(per_move.get(m, float('nan')), 5),
                              'converges': None} for m in common]}
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
    return {'n_moves': len(common),
            'spearman': round(rho, 4) if rho is not None and rho == rho else None,
            'sign_agreement': round(sign_agree, 4),
            'per_move': rows}


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
