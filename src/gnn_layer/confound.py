"""
gnn_layer/confound.py
---------------------
WS3 — elicitation/responsiveness confound localization (a caveat instrument, NOT a claim).

The observed Δprogression of a cue (analysis/mechanism.py) conflates the move's influence with
*when therapists choose to deploy it* (the elicitation/responsiveness confound, §9.4). The
transition model's learned counterfactual (gnn_layer/transition.py) estimates a move's influence
*independent of when it is used* (swap the cue, read the predicted shift). Where the two DIVERGE
— especially where they invert in sign — is where responsiveness most distorts the observational
signal. This module maps that signed divergence per (from_stage × PURER move) cell with
participant-clustered bootstrap CIs and ranks the most-confounded cells.

This is a methodological instrument for *caveating* the mechanism tables, not evidence of an
effect. n≈32 observational; nothing here is causal; every cell is hypothesis-generating. It reuses
the WS-T transition counterfactual (handed in from the runner, or recomputed) and the observed
Δprogression in the SAME E[stage] basis (so the divergence is apples-to-apples).
"""

import os
from typing import Dict, List, Optional

from process import output_paths as _paths

PURER_NAMES = {0: 'Phenomenology', 1: 'Utilization', 2: 'Reframing',
               3: 'Education', 4: 'Reinforcement'}
STAGE_NAMES = {0: 'Vigilance', 1: 'Avoidance', 2: 'AttentionReg',
               3: 'Metacognition', 4: 'Reappraisal'}


def _cell_values(ds, cf):
    """Per (from_stage, move): observed per-block Δ (E[TO]−E[FROM] for blocks where the move was
    used) and learned per-block influence (the move applied to from_stage blocks), each with
    participant ids — the raw material for the divergence + its clustered CI."""
    import numpy as np
    from . import transition as _T
    from collections import defaultdict

    e_obs = _T._estage(ds['Y']) - _T._estage(ds['F'])
    fs, mv, parts = ds['from_stage'], ds['move'], ds['parts']
    obs: Dict[tuple, tuple] = defaultdict(lambda: ([], []))
    for i in range(len(parts)):
        if mv[i] < 0:
            continue
        v, p = obs[(int(fs[i]), int(mv[i]))]
        v.append(float(e_obs[i])); p.append(parts[i])
    lrn: Dict[tuple, tuple] = defaultdict(lambda: ([], []))
    for r in cf.get('rows', []):
        v, p = lrn[(int(r['from_stage']), int(r['move']))]
        v.append(float(r['influence'])); p.append(r['participant_id'])
    return obs, lrn


def _divergence_ci(obs_vals, obs_parts, lrn_vals, lrn_parts, n_boot=1000, seed=42) -> dict:
    """Participant-clustered bootstrap CI on (mean observed Δ − mean learned influence)."""
    import numpy as np
    from collections import defaultdict
    og, lg = defaultdict(list), defaultdict(list)
    for v, p in zip(obs_vals, obs_parts):
        og[p].append(v)
    for v, p in zip(lrn_vals, lrn_parts):
        lg[p].append(v)
    point = float(np.mean(obs_vals)) - float(np.mean(lrn_vals))
    parts = sorted(set(list(og) + list(lg)), key=str)
    out = {'point': round(point, 4), 'lo': None, 'hi': None}
    if len(parts) < 2:
        return out
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(len(parts), size=len(parts), replace=True)
        o, l = [], []
        for ci in chosen:
            p = parts[ci]
            o += og.get(p, [])
            l += lg.get(p, [])
        if o and l:
            boots.append(np.mean(o) - np.mean(l))
    if boots:
        out['lo'] = round(float(np.percentile(boots, 2.5)), 4)
        out['hi'] = round(float(np.percentile(boots, 97.5)), 4)
    return out


def write_confound_csv(cells: List[dict], output_dir: str) -> Optional[str]:
    import pandas as pd
    if not cells:
        return None
    gnn = _paths.gnn_data_dir(output_dir)
    os.makedirs(gnn, exist_ok=True)
    path = os.path.join(gnn, 'confound_localization.csv')
    pd.DataFrame(cells).to_csv(path, index=False)
    return path


def _f(v) -> str:
    if v is None:
        return 'n/a'
    if isinstance(v, float):
        return 'n/a' if v != v else f'{v:+.3f}'
    return str(v)


def write_confound_report(cells: List[dict], output_dir: str) -> str:
    W = 78
    L = ["=" * W, "CONFOUND LOCALIZATION — where responsiveness distorts the observed signal", "=" * W, ""]
    L.append("CAVEAT INSTRUMENT, NOT A CLAIM. n≈32 observational; nothing here is causal (§9.4).")
    L.append("Observed Δprogression conflates a move's influence with WHEN therapists deploy it")
    L.append("(responsiveness). The transition model's learned counterfactual estimates influence")
    L.append("INDEPENDENT of timing. Their signed divergence (observed − counterfactual) localizes")
    L.append("where the observational mechanism table is most confounded — read it to caveat those")
    L.append("cells, not as an effect. Divergence CI = participant-clustered bootstrap.")
    L.append("")
    disagree = [c for c in cells if c.get('sign_disagree')]
    L.append(f"  {len(cells)} (from_stage × move) cells; {len(disagree)} INVERT in sign")
    L.append("  (observed and counterfactual point opposite ways — the strongest confound signature).")
    L.append("")
    L.append("-" * W)
    L.append("MOST-CONFOUNDED CELLS (by |divergence|; ‡ = sign inversion, * = observed FDR-sig)")
    L.append("-" * W)
    L.append(f"  {'from_stage':<14}{'move':<14}{'observed':<11}{'counterf':<11}{'divergence [95% CI]'}")
    for c in cells[:15]:
        tag = ('‡' if c.get('sign_disagree') else ' ') + ('*' if c.get('fdr_significant') else ' ')
        ci = f"[{_f(c.get('div_lo'))}, {_f(c.get('div_hi'))}]"
        L.append(f"  {c['from_stage_name']:<14}{c['move_name']:<14}"
                 f"{_f(c['observed_delta']):<11}{_f(c['counterfactual']):<11}"
                 f"{_f(c['divergence'])} {ci} {tag}  (n_obs={c['n_observed']})")
    L.append("")
    L.append("  A large positive divergence = observed looks more favourable than the cue itself")
    L.append("  warrants (the move was deployed at already-improving moments); a large negative")
    L.append("  divergence = observed looks worse (the move was deployed at moments of difficulty —")
    L.append("  the classic responsiveness pattern that depresses a genuinely-helpful move).")
    L.append("")
    L.append("-" * W)
    L.append("Cohorts 3–4 re-run trigger: with more data the per-cell CIs tighten and FDR-significant")
    L.append("observed cells appear; the divergence map then says which of THOSE to trust. Re-run via")
    L.append("`qra analyze`. Until then this is a map of where NOT to over-read the observed table.")
    L.append("=" * W)

    rep = _paths.reports_gnn_dir(output_dir)
    os.makedirs(rep, exist_ok=True)
    path = os.path.join(rep, 'confound_localization.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path


def run_confound_localization(df_all, output_dir: str, config=None, *,
                              transition_result: Optional[dict] = None,
                              seg_emb=None, verbose: bool = False) -> dict:
    """Map the signed divergence between the learned counterfactual and observed Δprogression per
    (from_stage × move), with participant-clustered CIs. Reuses the WS-T transition result when
    handed in (no retraining); otherwise recomputes it. Returns {status, files_written, cells}."""
    import numpy as np
    from . import transition as _T

    if (transition_result and transition_result.get('counterfactual', {}).get('status') == 'ok'
            and transition_result.get('dataset') is not None):
        ds = transition_result['dataset']
        cf = transition_result['counterfactual']
    else:
        emb = _T._qwen_embeddings(df_all, output_dir, seg_emb)
        if not emb:
            return {'status': 'skipped: Qwen embeddings unavailable', 'files_written': []}
        ds = _T.build_block_dataset(df_all, emb)
        if ds is None:
            return {'status': 'skipped: too few cue-block triples', 'files_written': []}
        seed = int(getattr(config, 'seed', 42)) if config is not None else 42
        cdim = int(getattr(config, 'transition_cue_dim', 32)) if config is not None else 32
        cue_proj, _p, _c = _T._project_cue(ds['C'], cdim, seed=seed)
        cf = _T.counterfactual_response(ds, cue_proj, config, seed=seed)
        if cf.get('status') != 'ok':
            return {'status': cf.get('status'), 'files_written': []}

    obs, lrn = _cell_values(ds, cf)
    fdr = _T.observed_per_stage_move(output_dir)
    boot = int(getattr(config, 'transition_bootstrap_n', 1000)) if config is not None else 1000
    seed = int(getattr(config, 'seed', 42)) if config is not None else 42

    cells = []
    for key in sorted(set(obs) & set(lrn)):
        ov, op = obs[key]
        lv, lp = lrn[key]
        if len(ov) < 2:
            continue
        ci = _divergence_ci(ov, op, lv, lp, n_boot=boot, seed=seed)
        s, m = key
        obs_mean, lrn_mean = round(float(np.mean(ov)), 4), round(float(np.mean(lv)), 4)
        cells.append({
            'from_stage': s, 'from_stage_name': STAGE_NAMES.get(s, str(s)),
            'move': m, 'move_name': PURER_NAMES.get(m, str(m)),
            'observed_delta': obs_mean, 'counterfactual': lrn_mean, 'divergence': ci['point'],
            'div_lo': ci['lo'], 'div_hi': ci['hi'],
            'sign_disagree': bool((obs_mean >= 0) != (lrn_mean >= 0)),
            'n_observed': len(ov),
            'fdr_significant': bool(fdr.get(key, {}).get('fdr_significant')),
        })
    cells.sort(key=lambda c: -abs(c['divergence']) if c['divergence'] is not None else 0)

    files = []
    p = write_confound_csv(cells, output_dir)
    if p:
        files.append(p)
    files.append(write_confound_report(cells, output_dir))
    if verbose:
        print(f"  [confound] {len(cells)} cells, "
              f"{sum(1 for c in cells if c['sign_disagree'])} sign-inverting")
    return {'status': 'ok', 'files_written': files, 'cells': cells, 'n_cells': len(cells)}
