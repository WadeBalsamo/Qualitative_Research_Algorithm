"""
gnn_layer/transition.py
-----------------------
The mechanism instrument — a dyadic FROM→CUE→TO transition model (the rebuild).

The as-built GraphSAGE is a per-segment *classifier* repurposed post-hoc, mis-specified for
mechanism three ways (methodology §4.7 / graph_experiments.md): (1) kNN-similarity edges are
content noise on a *process* question — H6 shows similarity neighbourhoods are stage-mixed;
(2) it never trains on *transitions*, so its counterfactual reads only a classifier's incidental
sensitivity; (3) the cue reaches a participant node through a single edge among many, so swapping
it moves the prediction by ≈0.03 (a likely contributor to the ρ=−0.13 triangulation failure).

This module replaces that path with a small **learned response function** over cue-block triples:

    TO_mixture  ≈  f( FROM_mixture , FROM_stage , pooled_cue_embedding )

trained on the observed FROM→CUE→TO triples with a KL objective. It is dyadic (the cue is a
first-class input, not a diluted edge), conditions on FROM stage (which is also the partial
control for the elicitation confound — comparing moves *within* a starting state), and uses NO
kNN — only the directed FROM→CUE→TO structure the cue blocks already encode. The cue is the raw
Qwen pooled-therapist embedding (discovery-grade features), projected to a small space so the
regressor stays small at n≈160 blocks.

What it buys:
  • an *earns-its-place* check — does adding the cue improve held-out TO prediction over a
    FROM-only baseline, under participant-grouped CV? (If not, that is the honest n≈32
    under-identification result, consistent with H2 — reported, not hidden.)
  • a counterfactual that reads a genuine learned response: swap the block's cue to each PURER
    move's centroid (vs a neutral baseline) and read the shift in predicted TO E[stage], per move
    and per (from_stage × move), with participant-clustered bootstrap CIs.
  • triangulation against the observed Δprogression (analysis/mechanism.py) — the LEAD signal.

Sensitivity analysis of a model, NOT causation (n≈32 observational + the elicitation confound,
§9.4). Every output is hypothesis-generating. torch + sklearn + numpy; degrades gracefully.
"""

import os
import sys
from collections import Counter
from typing import Dict, List, Optional

from process import output_paths as _paths

PURER_NAMES = {0: 'Phenomenology', 1: 'Utilization', 2: 'Reframing',
               3: 'Education', 4: 'Reinforcement'}
N_STAGES = 5  # the developmental arc; No-code is absence, not a transition target


# ---------------------------------------------------------------------------
# Embeddings (raw Qwen, reused from the discovery substrate)
# ---------------------------------------------------------------------------

def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _qwen_embeddings(df_all, output_dir, seg_emb):
    """Reuse a passed-in 4096-d dict, else load the cached Qwen3-8B vectors via the harness."""
    if seg_emb:
        if len(next(iter(seg_emb.values()))) >= 4096:
            return seg_emb
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from experiments.gnn_reliability import harness as _h
        return _h.get_embeddings(df_all, 'qwen', output_dir)
    except Exception as e:
        print(f"  [transition] Qwen embeddings unavailable ({e})")
        return None


def _isnan(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and v != v)
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# Cue-block triple dataset
# ---------------------------------------------------------------------------

def build_block_dataset(df_all, seg_emb, n_stages: int = N_STAGES) -> Optional[dict]:
    """Assemble FROM→CUE→TO triples with FROM/TO soft mixtures + raw pooled cue embeddings.

    Keeps only mediated blocks (non-empty therapist cue) whose FROM and TO are real-stage
    participant turns. Returns aligned numpy arrays + per-block metadata, or None if too few.
    """
    import numpy as np
    from .cue_features import build_cue_blocks_with_segments, cue_block_embeddings
    from . import soft_labels as _sl

    soft = _sl.build_soft_targets(df_all, 'weak', n_stages=n_stages)
    fl, pid = {}, {}
    purer: Dict[str, int] = {}
    for _, r in df_all.iterrows():
        sid = str(r.get('segment_id'))
        fl[sid] = r.get('final_label')
        pid[sid] = r.get('participant_id')
        if str(r.get('speaker', '')) == 'therapist':
            try:
                purer[sid] = int(r.get('purer_primary'))
            except (TypeError, ValueError):
                pass

    blocks = build_cue_blocks_with_segments(df_all)
    rows, cue_X = cue_block_embeddings(blocks, seg_emb)
    if cue_X is None or len(rows) == 0:
        return None

    F, St, Y, C, parts, fstages, moves, sess, fseg, tseg = ([] for _ in range(10))
    for i, b in enumerate(rows):
        fs, ts = b['from_seg_id'], b['to_seg_id']
        if fs not in soft or ts not in soft or _isnan(fl.get(fs)) or _isnan(fl.get(ts)):
            continue
        fstage = int(float(fl[fs]))
        if not (0 <= fstage < n_stages):
            continue
        F.append(np.asarray(soft[fs], dtype=np.float64)[:n_stages])
        Y.append(np.asarray(soft[ts], dtype=np.float64)[:n_stages])
        oh = np.zeros(n_stages); oh[fstage] = 1.0
        St.append(oh)
        C.append(np.asarray(cue_X[i], dtype=np.float64))
        parts.append(str(pid.get(fs)))
        fstages.append(fstage)
        ms = [purer[s] for s in b['therapist_seg_ids'] if s in purer]
        moves.append(Counter(ms).most_common(1)[0][0] if ms else -1)
        sess.append(b.get('session_id'))
        fseg.append(fs); tseg.append(ts)

    if len(F) < 12:
        return None
    # renormalize mixtures defensively (build_soft_targets already returns normalized)
    F = np.vstack(F); Y = np.vstack(Y)
    F = F / np.clip(F.sum(1, keepdims=True), 1e-9, None)
    Y = Y / np.clip(Y.sum(1, keepdims=True), 1e-9, None)
    return {'F': F, 'St': np.vstack(St), 'Y': Y, 'C': np.vstack(C),
            'parts': parts, 'from_stage': np.asarray(fstages, dtype=int),
            'move': np.asarray(moves, dtype=int), 'session': sess,
            'from_seg': fseg, 'to_seg': tseg, 'n_stages': n_stages}


def _project_cue(C, cue_dim: int, seed: int = 42):
    """PCA-project the raw pooled cue embeddings to a small space; returns (proj, pca)."""
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    Cn = normalize(C, norm='l2')
    k = int(min(cue_dim, Cn.shape[0] - 1, Cn.shape[1]))
    pca = PCA(n_components=k, random_state=seed).fit(Cn)
    return pca.transform(Cn), pca, Cn


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _make_net(in_dim: int, n_stages: int, hidden: int, dropout: float):
    import torch.nn as nn
    class _TransitionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, n_stages))

        def forward(self, x):
            return self.net(x)
    return _TransitionNet()


def _assemble_X(ds, cue_proj, idx, use_cue: bool):
    import numpy as np
    F, St = ds['F'][idx], ds['St'][idx]
    if use_cue:
        C = cue_proj[idx]
    else:
        C = np.zeros((len(idx), cue_proj.shape[1]), dtype=cue_proj.dtype)
    return np.hstack([F, St, C]).astype(np.float32)


def _train_net(X, Y, n_stages, cfg, seed: int = 42):
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    hidden = int(getattr(cfg, 'transition_hidden', 16))
    dropout = float(getattr(cfg, 'transition_dropout', 0.3))
    epochs = int(getattr(cfg, 'transition_epochs', 300))
    lr = float(getattr(cfg, 'transition_lr', 1e-2))
    wd = float(getattr(cfg, 'transition_weight_decay', 1e-3))
    net = _make_net(X.shape[1], n_stages, hidden, dropout)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    Yt = torch.as_tensor(Y, dtype=torch.float32)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.kl_div(F.log_softmax(net(Xt), dim=1), Yt, reduction='batchmean')
        loss.backward()
        opt.step()
    net.eval()
    return net


def _predict(net, X):
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        return F.softmax(net(torch.as_tensor(X, dtype=torch.float32)), dim=1).numpy()


def _estage(mix):
    import numpy as np
    return (np.asarray(mix) * np.arange(mix.shape[1])).sum(axis=1)


# ---------------------------------------------------------------------------
# Participant-grouped CV — does the cue earn its place?
# ---------------------------------------------------------------------------

def _participant_folds(parts: List[str], n_folds: int, seed: int = 42):
    import numpy as np
    uniq = sorted(set(parts))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {p: i % n_folds for i, p in enumerate(uniq)}
    return np.asarray([fold_of[p] for p in parts], dtype=int)


def crossval(ds, cue_proj, cfg, seed: int = 42) -> dict:
    """Held-out TO-prediction quality WITH the cue vs a FROM-only baseline (participant-grouped).

    Returns per-arm {kl, estage_mae, argmax_acc} plus the deltas (cue − from_only). The cue
    earns its place iff it lowers held-out KL / E[stage] MAE.
    """
    import numpy as np
    n_folds = int(getattr(cfg, 'transition_folds', 5))
    folds = _participant_folds(ds['parts'], n_folds, seed)
    n_stages = ds['n_stages']
    res = {True: {'kl': [], 'mae': [], 'acc': []}, False: {'kl': [], 'mae': [], 'acc': []}}
    import torch.nn.functional as F
    import torch
    for f in sorted(set(folds.tolist())):
        te = np.where(folds == f)[0]
        tr = np.where(folds != f)[0]
        if len(te) == 0 or len(tr) < 6:
            continue
        for use_cue in (True, False):
            net = _train_net(_assemble_X(ds, cue_proj, tr, use_cue), ds['Y'][tr],
                             n_stages, cfg, seed=seed + f)
            P = _predict(net, _assemble_X(ds, cue_proj, te, use_cue))
            Yte = ds['Y'][te]
            kl = float(F.kl_div(torch.as_tensor(np.log(np.clip(P, 1e-9, 1))),
                                torch.as_tensor(Yte), reduction='batchmean'))
            mae = float(np.mean(np.abs(_estage(P) - _estage(Yte))))
            acc = float(np.mean(P.argmax(1) == Yte.argmax(1)))
            res[use_cue]['kl'].append(kl)
            res[use_cue]['mae'].append(mae)
            res[use_cue]['acc'].append(acc)

    def _agg(d):
        import numpy as np
        return {k: (round(float(np.mean(v)), 4) if v else None) for k, v in d.items()}
    cue, base = _agg(res[True]), _agg(res[False])
    delta = {}
    for k in ('kl', 'mae', 'acc'):
        if cue[k] is not None and base[k] is not None:
            delta[k] = round(cue[k] - base[k], 4)
    return {'with_cue': cue, 'from_only': base, 'delta_cue_minus_from': delta,
            'n_blocks': len(ds['parts']), 'n_participants': len(set(ds['parts'])),
            'n_folds': n_folds}


# ---------------------------------------------------------------------------
# Learned counterfactual response
# ---------------------------------------------------------------------------

def _move_centroids(ds, cue_proj):
    """Per-PURER-move centroid in cue-projection space + neutral (all-block) baseline."""
    import numpy as np
    centroids, counts = {}, {}
    for m in range(len(PURER_NAMES)):
        idx = np.where(ds['move'] == m)[0]
        if len(idx) >= 2:
            centroids[m] = cue_proj[idx].mean(axis=0)
            counts[m] = int(len(idx))
    null = cue_proj.mean(axis=0)
    return centroids, null, counts


def counterfactual_response(ds, cue_proj, cfg, seed: int = 42) -> dict:
    """Train on all blocks; swap each block's cue → each PURER centroid (vs null), read the shift
    in predicted TO E[stage]. Aggregate per move and per (from_stage × move) with
    participant-clustered bootstrap CIs. Returns per_move, per_stage_move, rows."""
    import numpy as np
    from analysis import stats as _stats
    n_stages = ds['n_stages']
    net = _train_net(_assemble_X(ds, cue_proj, np.arange(len(ds['parts'])), True),
                     ds['Y'], n_stages, cfg, seed=seed)
    centroids, null, counts = _move_centroids(ds, cue_proj)
    if not centroids:
        return {'status': 'no PURER-move support among cue blocks'}

    n = len(ds['parts'])
    F, St = ds['F'], ds['St']
    boot = int(getattr(cfg, 'transition_bootstrap_n', 1000))

    def _pred_E(cue_rows):
        X = np.hstack([F, St, cue_rows]).astype(np.float32)
        return _estage(_predict(net, X))

    E_null = _pred_E(np.tile(null, (n, 1)))
    rows = []
    per_move, per_stage_move = [], []
    for m, cm in sorted(centroids.items()):
        E_m = _pred_E(np.tile(cm, (n, 1)))
        infl = E_m - E_null                                  # per-block predicted ΔE[stage]
        for i in range(n):
            rows.append({'from_seg_id': ds['from_seg'][i], 'to_seg_id': ds['to_seg'][i],
                         'from_stage': int(ds['from_stage'][i]), 'participant_id': ds['parts'][i],
                         'session_id': ds['session'][i], 'move': m,
                         'move_name': PURER_NAMES[m], 'influence': round(float(infl[i]), 5)})
        ci = _stats.cluster_bootstrap_ci(infl, ds['parts'], statistic=np.mean,
                                         n_boot=boot, seed=seed)
        per_move.append({'move': m, 'move_name': PURER_NAMES[m],
                         'mean_influence': round(float(np.mean(infl)), 5),
                         'ci_lo': round(ci['lo'], 5) if ci['lo'] == ci['lo'] else None,
                         'ci_hi': round(ci['hi'], 5) if ci['hi'] == ci['hi'] else None,
                         'n_blocks': n, 'n_participants': ci['n_clusters'],
                         'centroid_support': counts.get(m)})
        for s in range(n_stages):
            si = np.where(ds['from_stage'] == s)[0]
            if len(si) >= 2:
                per_stage_move.append({'from_stage': s, 'move': m, 'move_name': PURER_NAMES[m],
                                       'mean_influence': round(float(np.mean(infl[si])), 5),
                                       'n_blocks': int(len(si))})
    per_move.sort(key=lambda r: -r['mean_influence'])
    return {'status': 'ok', 'per_move': per_move, 'per_stage_move': per_stage_move,
            'rows': rows, 'n_blocks': n, 'move_support': counts}


# ---------------------------------------------------------------------------
# Triangulation vs the observed Δprogression (LEAD signal)
# ---------------------------------------------------------------------------

def observed_per_stage_move(output_dir: str) -> Dict[tuple, dict]:
    """{(from_stage, move): {mean_delta, fdr_significant}} from mechanism_delta_progression.csv."""
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
    out: Dict[tuple, dict] = {}
    for _, r in mdf[mdf['grouping'] == 'purer'].iterrows():
        mobj = re.search(r'\((\d+)\)\s*$', str(r['behavior']))
        if not mobj:
            continue
        try:
            key = (int(r['from_stage']), int(mobj.group(1)))
            out[key] = {'mean_delta': float(r['mean_delta_prog']),
                        'fdr_significant': bool(r.get('fdr_significant'))}
        except (ValueError, TypeError):
            continue
    return out


def triangulate(cf_result: dict, output_dir: str) -> Optional[dict]:
    """Spearman ρ + sign agreement between the learned per-(stage×move) counterfactual and the
    observed Δprogression cells (overall + FDR-restricted). Hypothesis-generating, not a claim."""
    observed = observed_per_stage_move(output_dir)
    if not observed or cf_result.get('status') != 'ok':
        return None
    learned = {(r['from_stage'], r['move']): r['mean_influence']
               for r in cf_result.get('per_stage_move', [])}
    common = sorted(set(observed) & set(learned))
    if len(common) < 3:
        return {'n_cells': len(common), 'spearman': None, 'sign_agreement': None,
                'sign_agreement_fdr': None, 'n_fdr_cells': 0}
    import numpy as np
    obs = [observed[c]['mean_delta'] for c in common]
    lrn = [learned[c] for c in common]
    try:
        from scipy import stats as _sps
        rho = float(_sps.spearmanr(obs, lrn).correlation)
    except Exception:
        rho = None
    sign = float(np.mean([(observed[c]['mean_delta'] >= 0) == (learned[c] >= 0) for c in common]))
    fdr_cells = [c for c in common if observed[c]['fdr_significant']]
    sign_fdr = (float(np.mean([(observed[c]['mean_delta'] >= 0) == (learned[c] >= 0)
                               for c in fdr_cells])) if fdr_cells else None)
    return {'n_cells': len(common),
            'spearman': round(rho, 4) if rho is not None and rho == rho else None,
            'sign_agreement': round(sign, 4),
            'sign_agreement_fdr': round(sign_fdr, 4) if sign_fdr is not None else None,
            'n_fdr_cells': len(fdr_cells)}


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _f(v) -> str:
    if v is None:
        return 'n/a'
    if isinstance(v, float):
        return 'n/a' if v != v else f'{v:+.4f}'
    return str(v)


def write_transition_csv(cf_result: dict, output_dir: str) -> List[str]:
    """Write the per-(from_stage×move) cells (for WS3 + the heatmap) and the per-move table
    (with CIs, for the figure). Returns the paths written."""
    import pandas as pd
    if cf_result.get('status') != 'ok':
        return []
    gnn = _paths.gnn_data_dir(output_dir)
    os.makedirs(gnn, exist_ok=True)
    paths = []
    p1 = os.path.join(gnn, 'transition_counterfactual.csv')
    pd.DataFrame(cf_result['per_stage_move']).to_csv(p1, index=False)
    paths.append(p1)
    p2 = os.path.join(gnn, 'transition_per_move.csv')
    pd.DataFrame(cf_result['per_move']).to_csv(p2, index=False)
    paths.append(p2)
    return paths


def write_transition_report(cv: dict, cf_result: dict, tri: Optional[dict],
                            output_dir: str) -> str:
    W = 78
    L = ["=" * W, "DYADIC FROM→CUE→TO TRANSITION MODEL (mechanism instrument)", "=" * W, ""]
    L.append("HYPOTHESIS-GENERATING. n≈32 participants, observational; sensitivity analysis of a")
    L.append("learned model, NOT causation (elicitation confound §9.4). A small regressor predicts")
    L.append("the TO participant's VAAMR mixture from (FROM_mixture, FROM_stage, pooled cue")
    L.append("embedding) over cue-block triples — directed FROM→CUE→TO structure, NO kNN (H6:")
    L.append("similarity neighbourhoods are stage-mixed), FROM-stage conditioning baked in (the")
    L.append("partial control for responsiveness — moves are compared WITHIN a starting state).")
    L.append("")
    L.append("-" * W)
    L.append("(1) DOES THE CUE EARN ITS PLACE? — held-out TO prediction, participant-grouped CV")
    L.append("-" * W)
    if cv:
        wc, fo, d = cv['with_cue'], cv['from_only'], cv.get('delta_cue_minus_from', {})
        L.append(f"  {cv['n_blocks']} cue blocks, {cv['n_participants']} participants, "
                 f"{cv['n_folds']}-fold grouped CV. Lower KL / E[stage] MAE = better.")
        L.append(f"    {'metric':<14}{'FROM only':<14}{'FROM+cue':<14}{'Δ(cue−from)':<14}")
        L.append(f"    {'KL':<14}{_f(fo.get('kl')):<14}{_f(wc.get('kl')):<14}{_f(d.get('kl')):<14}")
        L.append(f"    {'E[stage] MAE':<14}{_f(fo.get('mae')):<14}{_f(wc.get('mae')):<14}{_f(d.get('mae')):<14}")
        L.append(f"    {'argmax acc':<14}{_f(fo.get('acc')):<14}{_f(wc.get('acc')):<14}{_f(d.get('acc')):<14}")
        improved = (d.get('kl') is not None and d['kl'] < 0) or (d.get('mae') is not None and d['mae'] < 0)
        if improved:
            L.append("  → the cue lowers held-out error: the therapist language carries predictive")
            L.append("    signal for the next participant state beyond FROM alone (a lead, not proof).")
        else:
            L.append("  → the cue does NOT improve held-out prediction at this scale: the transition")
            L.append("    is under-identified at n≈32 (consistent with H2). The counterfactual below")
            L.append("    is therefore exploratory only — observed Δprogression (mechanism.py) LEADS.")
    L.append("")
    L.append("-" * W)
    L.append("(2) LEARNED COUNTERFACTUAL — predicted shift in TO E[stage] if the cue were each move")
    L.append("    (vs a neutral baseline). Participant-clustered bootstrap 95% CIs.")
    L.append("-" * W)
    if cf_result.get('status') == 'ok':
        for r in cf_result['per_move']:
            flag = '  [thin support — extrapolated]' if (r.get('centroid_support') or 0) < 10 else ''
            L.append(f"    {r['move_name']:<14} ΔE[stage] = {_f(r['mean_influence'])} "
                     f"[{_f(r['ci_lo'])}, {_f(r['ci_hi'])}]  "
                     f"(support {r['centroid_support']} blocks){flag}")
        L.append("  CI = across-block participant-clustered bootstrap; it does NOT capture")
        L.append("  model-training uncertainty, so it is tight by construction — read DIRECTIONS,")
        L.append("  not magnitudes. Moves with thin centroid support are extrapolations of the")
        L.append("  response function beyond where those cues actually occur (flagged above).")
    else:
        L.append(f"  {cf_result.get('status')}")
    L.append("")
    if tri:
        L.append("-" * W)
        L.append("(3) TRIANGULATION vs OBSERVED Δprogression (mechanism.py LEAD)")
        L.append("-" * W)
        L.append(f"  {tri['n_cells']} shared (from_stage × move) cells; "
                 f"Spearman ρ = {_f(tri.get('spearman'))}; "
                 f"sign agreement = {_f(tri.get('sign_agreement'))}")
        _rho = tri.get('spearman')
        if _rho is not None and _rho > 0:
            L.append("  The learned response triangulates POSITIVELY with the observed Δprogression")
            L.append("  (ρ>0) — unlike a per-segment classifier's counterfactual, which inverted it")
            L.append("  (ρ<0). Consistent with the transition-model rationale, but under-powered:")
        if tri.get('n_fdr_cells'):
            L.append(f"  sign agreement on the {tri['n_fdr_cells']} FDR-significant observed cells = "
                     f"{_f(tri.get('sign_agreement_fdr'))}")
        else:
            L.append("  (no FDR-significant observed cells at n≈32 — convergence is under-powered;")
            L.append("   a divergence is the elicitation-confound signature, not a refutation.)")
    L.append("")
    L.append("-" * W)
    L.append("Cohorts 3–4 re-run trigger: the earns-its-place check and triangulation become")
    L.append("adjudicable at higher N; re-run via `qra analyze` (reads master_segments.csv).")
    L.append("=" * W)

    rep = _paths.reports_gnn_dir(output_dir)
    os.makedirs(rep, exist_ok=True)
    path = os.path.join(rep, 'transition_model.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_transition_model(df_all, output_dir: str, config=None, *,
                         seg_emb=None, verbose: bool = False) -> dict:
    """Build the dataset → CV earns-its-place → counterfactual → triangulate → report/CSV.

    Returns {status, files_written, cv, counterfactual, triangulation}. Degrades gracefully.
    """
    def _log(m):
        if verbose:
            print(m)

    seg_emb = _qwen_embeddings(df_all, output_dir, seg_emb)
    if not seg_emb:
        return {'status': 'skipped: Qwen embeddings unavailable', 'files_written': []}

    seed = int(getattr(config, 'seed', 42)) if config is not None else 42
    ds = build_block_dataset(df_all, seg_emb, n_stages=N_STAGES)
    if ds is None:
        return {'status': 'skipped: too few cue-block triples', 'files_written': []}

    cue_dim = int(getattr(config, 'transition_cue_dim', 32)) if config is not None else 32
    cue_proj, _pca, _Cn = _project_cue(ds['C'], cue_dim, seed=seed)
    _log(f"  [transition] {ds['F'].shape[0]} triples, {len(set(ds['parts']))} participants, "
         f"cue→{cue_proj.shape[1]}d")

    cv = crossval(ds, cue_proj, config, seed=seed)
    _log(f"  [transition] earns-its-place Δ(cue−from): {cv.get('delta_cue_minus_from')}")
    cf = counterfactual_response(ds, cue_proj, config, seed=seed)
    tri = triangulate(cf, output_dir)

    files = []
    files.extend(write_transition_csv(cf, output_dir))
    files.append(write_transition_report(cv, cf, tri, output_dir))
    # ds + cue_proj are returned so WS3 (confound localization) can reuse the trained
    # counterfactual without retraining the model.
    return {'status': 'ok', 'files_written': files, 'cv': cv,
            'counterfactual': cf, 'triangulation': tri,
            'dataset': ds, 'cue_proj': cue_proj}
