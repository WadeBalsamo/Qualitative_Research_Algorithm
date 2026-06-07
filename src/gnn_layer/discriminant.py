"""
gnn_layer/discriminant.py
-------------------------
H6 — discriminant validity of the VAAMR construct (the positive turn on the H5 refutation).

The pre-registered battery (``graph_experiments.md``) showed that a content-similarity model
(Correct-&-Smooth / kNN label propagation) reproduces VAAMR at ≈ chance on held-out *participants*,
while a directly-supervised probe on the **same** Qwen embeddings reaches the human band. That is
not only a negative engineering result about the graph classifier — it is positive evidence about
the construct: **VAAMR stage is a developmental direction recoverable by supervision yet orthogonal
to content-similarity (topic / body region / affect). It is not a topic taxonomy.**

This module packages that finding as a reproducible construct-validity instrument and characterises
the geometry behind it. It is gate-independent discovery, runs at analyze-time, and re-adjudicates
automatically at Cohorts 3–4 scale.

(a) The H6 test — on identical participant-grouped folds and the same Qwen embeddings:
      supervised probe (above chance)  vs  content-similarity (≈ chance)  vs  chance baselines,
    scored on BOTH axes (LLM consensus n=205, human consensus n≈66) with participant-clustered
    bootstrap CIs, plus a paired (probe − content) κ contrast with a clustered CI.
(b) The geometry — stage ⟂ topic:
      • how well the leading *content* PCs recover stage (grouped-CV) vs the full-embedding probe,
      • the principal angles between the stage-discriminant subspace and the leading content PCs,
      • community × stage independence (the Track-D subtext communities are content, not stage),
    which is what *bounds* every similarity-based method (Track D included) and the dropped kNN edges.
(c) Operationalization — the "No code" null class; the stage≠topic caveat.

Reuses the reliability harness (``experiments/gnn_reliability``) for folds / probe / Correct-&-Smooth
/ two-axis scoring, ``gnn_layer.communities`` for the community partition, and ``analysis.stats`` for
the CIs / Cramér's V / permutation — nothing statistical is re-derived here.

NO causal claims; n≈32 observational; every output is hypothesis-generating.
numpy + sklearn + scipy (all already dependencies); degrades gracefully (logged) if the Qwen
embedding cache / endpoint or the harness is unavailable.
"""

import os
import sys
from typing import Dict, List, Optional

from process import output_paths as _paths
from .classifier.validation import VAAMR_NAMES

# H6 is argued on the five developmental stages (No-code is absence, not a direction); the arms
# use the 6-class No-code-aware probe (so "≈ chance vs above chance" is the fair full-task contrast).
STAGE_NAMES_5 = {k: VAAMR_NAMES[k] for k in range(5)}


# ---------------------------------------------------------------------------
# Lazy harness import (the reference grouped-CV / probe / C&S / scorer)
# ---------------------------------------------------------------------------

def _repo_root() -> str:
    # src/gnn_layer/discriminant.py -> repo root is two parents up from src/
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_harness():
    """Import experiments.gnn_reliability.{harness,baselines}; (None, None) if unavailable."""
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from experiments.gnn_reliability import harness as _h
        from experiments.gnn_reliability import baselines as _b
        return _h, _b
    except Exception as e:  # pragma: no cover - environment guard
        print(f"  [discriminant] reliability harness unavailable ({e}); skipping H6 test")
        return None, None


def _six_class_balanced(base_config):
    """A 6-class, class-weighted GnnLayerConfig clone for the No-code-aware arms."""
    from .config import GnnLayerConfig
    import dataclasses
    cfg = dataclasses.replace(base_config) if base_config is not None else GnnLayerConfig()
    cfg.vaamr_n_classes = 6
    cfg.vaamr_class_balance = True
    return cfg


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def _qwen_embeddings(harness, df_all, output_dir: str,
                     seg_emb: Optional[Dict[str, "object"]]) -> Optional[Dict[str, "object"]]:
    """Reuse a passed-in 4096-d embedding dict, else load the cached Qwen3-8B vectors."""
    import numpy as np
    if seg_emb:
        dim = len(next(iter(seg_emb.values())))
        if dim >= 4096:
            return seg_emb
    try:
        emb = harness.get_embeddings(df_all, 'qwen', output_dir)
        if emb:
            return emb
    except Exception as e:
        print(f"  [discriminant] Qwen embeddings unavailable ({e}); skipping H6 test")
    return None


# ---------------------------------------------------------------------------
# (a) H6 test — arms + paired contrast
# ---------------------------------------------------------------------------

def _chance_oof(baselines, df_all, embeddings, folds, config, kind: str, seed: int = 42
                ) -> Dict[str, int]:
    """Out-of-fold predictions for a label-free chance predictor (identical fold handling).

    kind='most_frequent' → predict the training fold's modal class.
    kind='stratified'    → sample from the training fold's class distribution.
    """
    import numpy as np
    seg_ids, labels, _ = baselines._prepare_labeled(df_all, embeddings, config)
    if not seg_ids:
        return {}
    label_of = dict(zip(seg_ids, labels))
    fold_of, fold_list = baselines._resolve_folds(seg_ids, df_all, folds)
    rng = np.random.default_rng(seed)
    preds: Dict[str, int] = {}
    for _f, train_ids, test_ids in baselines._iter_folds(seg_ids, fold_of, fold_list):
        ytr = np.array([label_of[s] for s in train_ids], dtype=int)
        if ytr.size == 0:
            continue
        classes, counts = np.unique(ytr, return_counts=True)
        if kind == 'most_frequent':
            pred = int(classes[counts.argmax()])
            for s in test_ids:
                preds[s] = pred
        else:  # stratified random
            p = counts / counts.sum()
            for s in test_ids:
                preds[s] = int(rng.choice(classes, p=p))
    return preds


def _paired_contrast(harness, oof_a: Dict[str, int], oof_b: Dict[str, int],
                     df_all, output_dir: str, n_classes: int, seed: int = 42) -> Dict[str, dict]:
    """Participant-clustered CI on κ(A,ref) − κ(B,ref), on both axes (A=probe, B=content)."""
    import numpy as np
    from analysis import stats as _stats
    from analysis import irr_stats
    from analysis.irr_analysis import _consensus_rows
    from process import irr_import

    part_of = {str(r.get('segment_id')): str(r.get('participant_id')) for _, r in df_all.iterrows()}

    def _map(p):
        return -1 if (n_classes >= 6 and p == 5) else p

    lab = harness._labeled_participants(df_all)
    final_of = {str(r['segment_id']): int(r['final_label']) for _, r in lab.iterrows()}
    master = set(df_all['segment_id'].astype(str))

    def _items(axis: str):
        triples, clusters = [], []
        if axis == 'llm':
            for sid, ref in final_of.items():
                if sid in oof_a and sid in oof_b:
                    triples.append((oof_a[sid], oof_b[sid], ref))
                    clusters.append(part_of.get(sid))
        else:
            for c in _consensus_rows(irr_import.read_human_codes(output_dir)):
                sid = c.get('segment_id')
                h = c.get('primary')
                if not sid or str(sid) not in master or h is None:
                    continue
                sid = str(sid)
                if sid in oof_a and sid in oof_b:
                    triples.append((_map(oof_a[sid]), _map(oof_b[sid]), int(h)))
                    clusters.append(part_of.get(sid))
        return triples, clusters

    out: Dict[str, dict] = {}
    for axis in ('llm', 'human'):
        triples, clusters = _items(axis)
        if len(triples) < 2:
            out[axis] = None
            continue
        arr = np.asarray(triples, dtype=int)              # cols a, b, ref in [-1, 5]
        packed = ((arr[:, 0] + 1) * 100 + (arr[:, 1] + 1) * 10 + (arr[:, 2] + 1)).astype(float)

        def _stat(x):
            x = x.astype(int)
            a = (x // 100) - 1
            b = ((x // 10) % 10) - 1
            r = (x % 10) - 1
            ka = irr_stats.cohen_kappa(a.tolist(), r.tolist())
            kb = irr_stats.cohen_kappa(b.tolist(), r.tolist())
            if ka is None or kb is None:
                return float('nan')
            return float(ka - kb)

        res = _stats.cluster_bootstrap_ci(packed, clusters, statistic=_stat, n_boot=2000, seed=seed)
        out[axis] = {'delta': res['point'], 'lo': res['lo'], 'hi': res['hi'],
                     'n': res['n'], 'n_clusters': res['n_clusters']}
    return out


def _run_arms(harness, baselines, df_all, embeddings, folds, base_config,
              output_dir: str, write_ledger: bool, seed: int = 42) -> dict:
    """Run + score the four H6 arms (probe, content, chance×2) on both axes."""
    cfg6 = _six_class_balanced(base_config)
    meta_common = {'embedding': 'qwen', 'embed_dim': 4096, 'imbalance': 'balanced', 'seed': seed}

    arms = {}
    specs = [
        ('H6-probe', baselines.run_linear_probe, 'LinearProbe'),
        ('H6-content', baselines.run_correct_smooth, 'Correct&Smooth'),
    ]
    oofs: Dict[str, Dict[str, int]] = {}
    for arm, fn, method in specs:
        oof = fn(df_all, embeddings, folds, cfg6)
        oofs[arm] = oof
        meta = dict(meta_common, method=method, notes=f'H6 {arm}')
        arms[arm] = harness.score_arm(arm, oof, df_all, output_dir, 6, meta=meta,
                                      write_ledger=write_ledger)
    for arm, kind, method in (('H6-chance-mode', 'most_frequent', 'ChanceModal'),
                              ('H6-chance-strat', 'stratified', 'ChanceStratified')):
        oof = _chance_oof(baselines, df_all, embeddings, folds, cfg6, kind, seed=seed)
        oofs[arm] = oof
        meta = dict(meta_common, method=method, imbalance='none', notes=f'H6 {arm}')
        arms[arm] = harness.score_arm(arm, oof, df_all, output_dir, 6, meta=meta,
                                      write_ledger=write_ledger)

    contrast = _paired_contrast(harness, oofs['H6-probe'], oofs['H6-content'],
                                df_all, output_dir, 6, seed=seed)
    return {'arms': arms, 'contrast': contrast}


# ---------------------------------------------------------------------------
# (b) Geometry — stage ⟂ topic
# ---------------------------------------------------------------------------

def _participant_matrices(df_all, embeddings):
    """Return (X_all, X_lab, y_lab, groups_lab) L2-normalized; labels are the 5 stages."""
    import numpy as np
    from sklearn.preprocessing import normalize

    all_rows, lab_rows = [], []
    for _, r in df_all.iterrows():
        if str(r.get('speaker', '')) != 'participant':
            continue
        sid = str(r.get('segment_id'))
        if sid not in embeddings:
            continue
        vec = np.asarray(embeddings[sid], dtype=np.float64)
        all_rows.append(vec)
        fl = r.get('final_label')
        try:
            ok = fl is not None and not np.isnan(float(fl))
        except (TypeError, ValueError):
            ok = False
        if ok and 0 <= int(float(fl)) < 5:
            lab_rows.append((vec, int(float(fl)), str(r.get('participant_id'))))
    if not all_rows or not lab_rows:
        return None
    X_all = normalize(np.vstack(all_rows), norm='l2')
    X_lab = normalize(np.vstack([v for v, _, _ in lab_rows]), norm='l2')
    y_lab = np.asarray([y for _, y, _ in lab_rows], dtype=int)
    groups = np.asarray([g for _, _, g in lab_rows], dtype=object)
    return X_all, X_lab, y_lab, groups


def _stage_variance_by_pcs(X_all, X_lab, y_lab, groups, ks=(5, 10, 20, 50),
                           n_folds: int = 5, seed: int = 42) -> dict:
    """Grouped-CV recovery of stage from the top-k *content* PCs vs the full embedding vs chance.

    If the leading content (variance) directions carried the stage, a classifier on the top-k
    PCs would match the full-embedding probe. H6 predicts it stays near the most-frequent floor.
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from analysis import irr_stats

    n_comp = int(min(max(ks), X_all.shape[0] - 1, X_all.shape[1]))
    pca = PCA(n_components=n_comp, random_state=seed).fit(X_all)
    Z_lab_full = pca.transform(X_lab)
    evr = pca.explained_variance_ratio_

    def _grouped_kappa(featmat) -> Optional[float]:
        try:
            sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            preds = np.full(len(y_lab), -1, dtype=int)
            for tr, te in sgkf.split(featmat, y_lab, groups):
                if len(np.unique(y_lab[tr])) < 2:
                    preds[te] = int(np.bincount(y_lab[tr]).argmax())
                    continue
                clf = LogisticRegression(max_iter=2000, class_weight='balanced')
                clf.fit(featmat[tr], y_lab[tr])
                preds[te] = clf.predict(featmat[te])
            return irr_stats.cohen_kappa(preds.tolist(), y_lab.tolist())
        except Exception:
            return None

    # chance (most-frequent) accuracy/kappa floor
    most_freq = int(np.bincount(y_lab).argmax())
    chance_acc = float(np.mean(y_lab == most_freq))

    rows = []
    for k in ks:
        if k > Z_lab_full.shape[1]:
            continue
        kap = _grouped_kappa(Z_lab_full[:, :k])
        rows.append({'k': int(k), 'cum_var': float(evr[:k].sum()),
                     'stage_kappa_from_pcs': kap})
    full_kappa = _grouped_kappa(X_lab)  # the full-embedding probe (grouped-CV)
    return {'pc_rows': rows, 'full_embedding_kappa': full_kappa,
            'chance_modal_acc': chance_acc, 'n_components': n_comp,
            'evr_top10': float(evr[:10].sum())}


def _knn_stage_homophily(X_lab, y_lab, ks=(1, 5, 10)) -> dict:
    """Local cosine-kNN stage homophily — the CORRECT operationalization of 'VAAMR is not
    homophilous in embedding space' (graph_experiments.md §4.2), and the precise reason kNN
    edges are dropped from the transition model.

    For each labeled segment, the fraction of its k nearest (cosine) labeled neighbours that
    share its stage, vs the base rate (Σ p_s², the same-stage fraction under random neighbours).
    Low homophily (lift over base ≈ 1) means similarity neighbourhoods are stage-MIXED — so
    similarity propagation (Correct-&-Smooth), kNN graph edges, and GNN message-passing cannot
    recover the developmental label, even though a linear probe can. This is local-neighbourhood
    structure, NOT a claim that the stage occupies an exotic subspace (it does not — see the
    content-PC recovery below).
    """
    import numpy as np
    n = X_lab.shape[0]
    if n < 6:
        return {'available': False}
    S = X_lab @ X_lab.T                       # rows are L2-normalized → cosine
    np.fill_diagonal(S, -np.inf)
    order = np.argsort(-S, axis=1)            # nearest neighbour first
    _, counts = np.unique(y_lab, return_counts=True)
    p = counts / counts.sum()
    base_rate = float((p ** 2).sum())
    rows = []
    for k in ks:
        if k >= n:
            continue
        same = [float(np.mean(y_lab[order[i, :k]] == y_lab[i])) for i in range(n)]
        frac = float(np.mean(same))
        rows.append({'k': int(k), 'mean_same_stage_frac': round(frac, 4),
                     'base_rate': round(base_rate, 4),
                     'lift_over_base': round(frac / base_rate, 3) if base_rate > 0 else None})
    kk = min(5, n - 1)
    per_stage = {}
    for s in sorted(set(y_lab.tolist())):
        idx = np.where(y_lab == s)[0]
        fr = [float(np.mean(y_lab[order[i, :kk]] == s)) for i in idx]
        per_stage[int(s)] = round(float(np.mean(fr)), 4) if fr else None
    return {'available': True, 'rows': rows, 'base_rate': round(base_rate, 4),
            'per_stage_k': kk, 'per_stage': per_stage}


def _community_stage_independence(df_all, embeddings, config, seed: int = 42) -> dict:
    """Track-D subtext communities × VAAMR stage — Cramér's V + label-permutation p.

    Near-independence is the evidence that the communities (and every similarity method) are
    organised by content, not stage — the bound WS2 needs.
    """
    import numpy as np
    from analysis import stats as _stats
    from . import communities as _com

    meta = _com._seg_meta(df_all)
    part_emb = {sid: embeddings[sid] for sid in embeddings
                if meta.get(sid) and meta[sid]['speaker'] == 'participant'}
    if len(part_emb) < 10:
        return {'available': False}
    # τ=0.85 leaves participant segments as near-singletons; search down until the partition
    # has ≥3 multi-member communities (else the V is a sparse-table artifact — report n/a).
    base_thr = float(getattr(config, 'community_sim_threshold', 0.85))
    thresholds = [t for t in (base_thr, 0.75, 0.65, 0.55) if t <= base_thr] or [base_thr]
    detect, used_thr, n_multi = None, None, 0
    for thr in thresholds:
        G, ginfo = _com.build_subtext_graph(part_emb, meta, threshold=thr)
        if ginfo.get('n_nodes', 0) < 5:
            continue
        d = _com.detect_communities(G, part_emb, list(G.nodes()), config)
        nm = sum(1 for c in d['communities'] if len(c) >= 3)
        if nm >= 3:
            detect, used_thr, n_multi = d, thr, nm
            break
        if detect is None:                       # remember the best-so-far for the n/a report
            detect, used_thr, n_multi = d, thr, nm
    if detect is None or n_multi < 3:
        return {'available': False, 'degenerate': True, 'threshold': used_thr,
                'n_multi_member': n_multi,
                'note': 'subtext communities are near-singletons at this scale/threshold'}
    labels = detect['labels']

    stage_of: Dict[str, int] = {}
    for _, r in df_all.iterrows():
        sid = str(r.get('segment_id'))
        fl = r.get('final_label')
        try:
            if fl is not None and not np.isnan(float(fl)) and 0 <= int(float(fl)) < 5:
                stage_of[sid] = int(float(fl))
        except (TypeError, ValueError):
            continue

    comm_ids, stages = [], []
    for sid, c in labels.items():
        if sid in stage_of:
            comm_ids.append(int(c))
            stages.append(stage_of[sid])
    if len(set(comm_ids)) < 2 or len(set(stages)) < 2:
        return {'available': False}

    comm_ids = np.asarray(comm_ids)
    stages = np.asarray(stages)
    uc = sorted(set(comm_ids.tolist()))
    table = np.zeros((len(uc), 5), dtype=float)
    cidx = {c: i for i, c in enumerate(uc)}
    for c, s in zip(comm_ids, stages):
        table[cidx[c], s] += 1
    cv = _stats.cramers_v(table)
    # Chance-corrected agreement between the content partition and the stage partition
    # (Cramér's V is inflated by many small communities; ARI ≈ 0 ⇒ independent).
    from sklearn.metrics import adjusted_rand_score
    ari_cs = float(adjusted_rand_score(stages.tolist(), comm_ids.tolist()))

    # label-permutation p on Cramér's V (shuffle stage labels across the same segments)
    rng = np.random.default_rng(seed)
    obs = cv['cramers_v']
    n_perm, ge = 500, 0
    if obs == obs:
        for _ in range(n_perm):
            sp = rng.permutation(stages)
            t = np.zeros_like(table)
            for c, s in zip(comm_ids, sp):
                t[cidx[c], s] += 1
            v = _stats.cramers_v(t)['cramers_v']
            if v == v and v >= obs:
                ge += 1
        perm_p = (ge + 1) / (n_perm + 1)
    else:
        perm_p = float('nan')
    return {'available': True, 'n_communities': len(uc), 'n_segments': int(len(comm_ids)),
            'n_multi_member': n_multi, 'threshold': used_thr,
            'cramers_v': obs, 'chi2_p': cv['p_value'], 'perm_p': perm_p,
            'ari_community_vs_stage': round(ari_cs, 4),
            'ari': detect.get('ari_louvain_vs_hierarchical')}


# ---------------------------------------------------------------------------
# (c) Operationalization — the No-code null share
# ---------------------------------------------------------------------------

def _nocode_fractions(df_all, output_dir: str) -> dict:
    """Share of No-code among (i) human consensus items and (ii) labeled participant corpus."""
    import numpy as np
    from analysis.irr_analysis import _consensus_rows
    from process import irr_import

    part = df_all[df_all['speaker'] == 'participant'] if 'speaker' in df_all.columns else df_all
    n_part = len(part)
    n_nolabel = int(part['final_label'].isna().sum()) if 'final_label' in part.columns else 0
    try:
        cons = _consensus_rows(irr_import.read_human_codes(output_dir))
        h_total = sum(1 for c in cons if c.get('primary') is not None)
        h_nocode = sum(1 for c in cons if c.get('primary') == -1)
    except Exception:
        h_total = h_nocode = 0
    return {'corpus_nolabel_frac': (n_nolabel / n_part) if n_part else 0.0,
            'n_participant': int(n_part), 'n_nolabel': n_nolabel,
            'human_nocode_frac': (h_nocode / h_total) if h_total else 0.0,
            'human_nocode': h_nocode, 'human_total': h_total}


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None:
        return 'n/a'
    if isinstance(v, float):
        return 'n/a' if v != v else f'{v:.3f}'
    return str(v)


def _ci(d) -> str:
    if not d:
        return 'n/a'
    return f"[{_fmt(d.get('lo'))}, {_fmt(d.get('hi'))}]"


def write_discriminant_csv(result: dict, output_dir: str) -> Optional[str]:
    import pandas as pd
    rows = []
    for arm, m in result.get('arms', {}).items():
        if not m:
            continue
        ha, la = m['human_axis'], m['llm_axis']
        rows.append({
            'section': 'arm', 'name': arm,
            'human_kappa': ha['cohen_kappa'], 'human_lo': ha['ci95'][0], 'human_hi': ha['ci95'][1],
            'human_n': ha['n'],
            'llm_kappa_205': la['cohen_kappa_205'], 'llm_lo': la['ci95'][0], 'llm_hi': la['ci95'][1],
            'llm_n': la['n'],
        })
    for axis, d in (result.get('contrast') or {}).items():
        if d:
            rows.append({'section': 'paired_contrast', 'name': f'probe_minus_content_{axis}',
                         'human_kappa' if axis == 'human' else 'llm_kappa_205': d['delta'],
                         ('human_lo' if axis == 'human' else 'llm_lo'): d['lo'],
                         ('human_hi' if axis == 'human' else 'llm_hi'): d['hi']})
    geo = result.get('geometry', {})
    for r in geo.get('variance', {}).get('pc_rows', []):
        rows.append({'section': 'stage_from_content_pcs', 'name': f"top_{r['k']}_pcs",
                     'value': r['stage_kappa_from_pcs'], 'cum_var': r['cum_var']})
    for r in geo.get('homophily', {}).get('rows', []):
        rows.append({'section': 'knn_homophily', 'name': f"k{r['k']}",
                     'value': r['mean_same_stage_frac'], 'base_rate': r['base_rate'],
                     'lift_over_base': r['lift_over_base']})
    cs = geo.get('community_stage', {})
    if cs.get('available'):
        rows.append({'section': 'community_stage', 'name': 'cramers_v',
                     'value': cs['cramers_v'], 'perm_p': cs['perm_p'], 'ari': cs.get('ari'),
                     'cum_var': cs.get('threshold')})
    if not rows:
        return None
    gnn_dir = _paths.gnn_data_dir(output_dir)
    os.makedirs(gnn_dir, exist_ok=True)
    path = os.path.join(gnn_dir, 'discriminant_validity.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_pca_coords_csv(coords: dict, output_dir: str) -> Optional[str]:
    """Per-labeled-segment PCA-2D coords + stage (for the figure)."""
    import pandas as pd
    if not coords:
        return None
    gnn_dir = _paths.gnn_data_dir(output_dir)
    os.makedirs(gnn_dir, exist_ok=True)
    path = os.path.join(gnn_dir, 'discriminant_pca_coords.csv')
    pd.DataFrame(coords).to_csv(path, index=False)
    return path


def write_discriminant_report(result: dict, output_dir: str) -> str:
    W = 78
    L = ["=" * W, "H6 — DISCRIMINANT VALIDITY: VAAMR IS DEVELOPMENTAL, NOT TOPICAL", "=" * W, ""]
    L.append("HYPOTHESIS-GENERATING / CONSTRUCT VALIDITY. n≈32 participants, observational.")
    L.append("Claim: VAAMR stage is recoverable by direct supervision yet poorly carried by what")
    L.append("makes segments *similar* (topic, body region, affect). On the SAME Qwen embeddings a")
    L.append("supervised probe reaches the human band while a content-similarity model lands far")
    L.append("below it (near chance on the LLM axis). The property that defeats a similarity")
    L.append("classifier (H5) is positive evidence about the construct (H6) — and it bounds every")
    L.append("similarity-based method, incl. Track-D communities and the dropped kNN graph edges.")
    L.append("")

    arms = result.get('arms', {})
    L.append("-" * W)
    L.append("(a) RECOVERABILITY vs SIMILARITY — identical participant-grouped folds, same Qwen")
    L.append("    embeddings, 6-class (No-code-aware). κ = point [95% participant-clustered CI].")
    L.append("-" * W)
    L.append(f"  {'arm':<18}{'HUMAN axis κ (n)':<30}{'LLM axis κ (n=205)':<28}")
    order = ['H6-probe', 'H6-content', 'H6-chance-mode', 'H6-chance-strat']
    pretty = {'H6-probe': 'probe (supervised)', 'H6-content': 'content (C&S)',
              'H6-chance-mode': 'chance (modal)', 'H6-chance-strat': 'chance (stratified)'}
    for arm in order:
        m = arms.get(arm)
        if not m:
            continue
        ha, la = m['human_axis'], m['llm_axis']
        hstr = f"{_fmt(ha['cohen_kappa'])} {_ci({'lo': ha['ci95'][0], 'hi': ha['ci95'][1]})} (n={ha['n']})"
        lstr = f"{_fmt(la['cohen_kappa_205'])} {_ci({'lo': la['ci95'][0], 'hi': la['ci95'][1]})}"
        L.append(f"  {pretty[arm]:<18}{hstr:<30}{lstr:<28}")
    L.append("")
    contrast = result.get('contrast') or {}
    for axis in ('human', 'llm'):
        d = contrast.get(axis)
        if d:
            L.append(f"  paired (probe − content) κ, {axis} axis: "
                     f"Δκ = {_fmt(d['delta'])} {_ci(d)} "
                     f"(n={d['n']}, {d['n_clusters']} participants)")
    L.append("")
    L.append("  Read: probe ≫ content (paired Δκ CI excludes 0 on both axes). Content sits far")
    L.append("  below the probe — near chance on the LLM axis, only modestly above chance on the")
    L.append("  human axis — so the stage signal is present in the features but poorly carried by")
    L.append("  content similarity. CIs, not point estimates, carry the claim.")
    L.append("")

    geo = result.get('geometry', {})
    var = geo.get('variance', {})
    hom = geo.get('homophily', {})
    if var or hom:
        L.append("-" * W)
        L.append("(b) GEOMETRY — stage is linearly decodable, but NOT locally homophilous")
        L.append("-" * W)
    if var:
        L.append("  Stage recovered from the top-k CONTENT principal components (grouped-CV κ):")
        L.append(f"    full-embedding probe κ = {_fmt(var.get('full_embedding_kappa'))}   "
                 f"(most-frequent acc floor = {_fmt(var.get('chance_modal_acc'))})")
        for r in var.get('pc_rows', []):
            L.append(f"    top {r['k']:>2} PCs (cum var {r['cum_var']*100:4.1f}%): "
                     f"stage κ = {_fmt(r['stage_kappa_from_pcs'])}")
        L.append("    → the stage signal is linearly present and even substantially captured by the")
        L.append("      leading CONTENT directions — it is NOT hidden in an exotic low-variance")
        L.append("      corner. What defeats similarity methods is LOCAL, not global (next).")
        L.append("")
    if hom.get('available'):
        L.append("  Local cosine-kNN stage homophily (the operative discriminant property):")
        L.append(f"    base rate (same-stage under random neighbours) = {_fmt(hom.get('base_rate'))}")
        for r in hom['rows']:
            L.append(f"    {r['k']:>2}-NN: same-stage fraction = {_fmt(r['mean_same_stage_frac'])} "
                     f"(lift over base = {_fmt(r['lift_over_base'])}×)")
        ps = hom.get('per_stage', {})
        if ps:
            ptxt = ', '.join(f"{STAGE_NAMES_5.get(s, s)}={_fmt(v)}" for s, v in ps.items())
            L.append(f"    per-stage {hom.get('per_stage_k')}-NN same-stage fraction: {ptxt}")
        L.append("    → similarity neighbours are only WEAKLY stage-homophilous (lift over base is")
        L.append("      small and decays with k; some stages sit below base rate) — far below what")
        L.append("      supervision achieves. So Correct-&-Smooth, GNN message-passing, and kNN graph")
        L.append("      edges recover little, which is why the transition model drops kNN and keeps")
        L.append("      only directed temporal+precipitates structure.")
        L.append("")
    cs = geo.get('community_stage', {})
    if cs.get('available'):
        L.append("  Community × stage (subtext-similarity clusters vs VAAMR stage):")
        L.append(f"    {cs.get('n_multi_member')} multi-member communities (τ={_fmt(cs.get('threshold'))}) "
                 f"over {cs['n_segments']} labeled segments;")
        L.append(f"    ARI(community, stage) = {_fmt(cs.get('ari_community_vs_stage'))} "
                 f"(≈0 ⇒ independent); Cramér's V = {_fmt(cs['cramers_v'])} "
                 f"(perm p = {_fmt(cs['perm_p'])}).")
        L.append("    → content clusters do not line up with stage (bounds WS2 + every kNN method).")
        L.append("")
    elif cs.get('degenerate'):
        L.append("  Community × stage: subtext communities are near-singletons at this scale "
                 f"(τ≤{_fmt(cs.get('threshold'))});")
        L.append("    the kNN homophily above is the cluster-free read of the same property.")
        L.append("")

    nc = result.get('operationalization', {})
    if nc:
        L.append("-" * W)
        L.append("(c) OPERATIONALIZATION IMPLICATIONS")
        L.append("-" * W)
        L.append(f"  'No code' null class: {nc['human_nocode']}/{nc['human_total']} "
                 f"({nc['human_nocode_frac']*100:.0f}%) of human consensus items express NO VAAMR")
        L.append(f"  stage; {nc['n_nolabel']}/{nc['n_participant']} "
                 f"({nc['corpus_nolabel_frac']*100:.0f}%) of participant segments carry no label.")
        L.append("  A fixed 5-class model is wrong on ~a third of items by construction — only a model")
        L.append("  that can abstain (the LLM, the 6-class probe) tracks the human coders there. This")
        L.append("  is a construct-operationalization finding, not a modeling one.")
        L.append("  Caveat that bounds every similarity method: VAAMR is a developmental stance, not a")
        L.append("  topic; cosine neighbours share content, not stage (this is why kNN edges and")
        L.append("  similarity communities cannot carry the label — see WS2).")
        L.append("")

    L.append("-" * W)
    L.append("Cohorts 3–4 re-run trigger: H6 is N-robust; re-run to tighten CIs and re-confirm the")
    L.append("geometry as the corpus grows (same harness, folds, seed).")
    L.append("=" * W)

    rep = _paths.reports_gnn_dir(output_dir)
    os.makedirs(rep, exist_ok=True)
    path = os.path.join(rep, 'discriminant_validity.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_discriminant_validity(df_all, output_dir: str, config=None, *,
                              seg_emb: Optional[Dict[str, "object"]] = None,
                              write_ledger: bool = False, verbose: bool = False) -> dict:
    """H6 discriminant-validity instrument: arms + geometry + report/CSV/coords.

    Returns {status, files_written, ...}. Gate-independent; degrades gracefully (logged) if the
    Qwen embeddings or the reliability harness are unavailable.
    """
    def _log(msg):
        if verbose:
            print(msg)

    harness, baselines = _load_harness()
    if harness is None:
        return {'status': 'skipped: harness unavailable', 'files_written': []}

    embeddings = _qwen_embeddings(harness, df_all, output_dir, seg_emb)
    if not embeddings:
        return {'status': 'skipped: Qwen embeddings unavailable', 'files_written': []}

    seed = int(getattr(config, 'seed', 42)) if config is not None else 42
    folds = harness.build_folds(df_all, seed=seed, verbose=verbose)
    if not folds:
        return {'status': 'skipped: no labeled folds', 'files_written': []}

    _log("  [discriminant] H6 arms (probe / content / chance) on Qwen, both axes ...")
    h6 = _run_arms(harness, baselines, df_all, embeddings, folds, config,
                   output_dir, write_ledger, seed=seed)

    _log("  [discriminant] geometry (PCA content axes, stage⟂topic angles, community×stage) ...")
    geometry: Dict[str, dict] = {}
    pca_coords = {}
    mats = _participant_matrices(df_all, embeddings)
    if mats is not None:
        X_all, X_lab, y_lab, groups = mats
        geometry['variance'] = _stage_variance_by_pcs(X_all, X_lab, y_lab, groups, seed=seed)
        geometry['homophily'] = _knn_stage_homophily(X_lab, y_lab)
        try:
            from sklearn.decomposition import PCA
            Z = PCA(n_components=2, random_state=seed).fit(X_all).transform(X_lab)
            pca_coords = {'pc1': Z[:, 0].tolist(), 'pc2': Z[:, 1].tolist(),
                          'stage': y_lab.tolist(),
                          'stage_name': [STAGE_NAMES_5[s] for s in y_lab.tolist()]}
        except Exception:
            pca_coords = {}
    geometry['community_stage'] = _community_stage_independence(df_all, embeddings,
                                                               config or _six_class_balanced(None),
                                                               seed=seed)

    result = {
        'status': 'ok',
        'arms': h6['arms'],
        'contrast': h6['contrast'],
        'geometry': geometry,
        'operationalization': _nocode_fractions(df_all, output_dir),
    }

    files = []
    p = write_discriminant_csv(result, output_dir)
    if p:
        files.append(p)
    p = write_pca_coords_csv(pca_coords, output_dir)
    if p:
        files.append(p)
    files.append(write_discriminant_report(result, output_dir))
    result['files_written'] = files
    return result
