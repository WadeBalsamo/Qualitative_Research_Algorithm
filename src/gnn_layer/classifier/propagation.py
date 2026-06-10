"""
gnn_layer/propagation.py
------------------------
Track A4 — semi-supervised label propagation (optional + measured).

Unlabeled nodes receive messages during training but do not explicitly inherit their
neighbours' soft labels. This module diffuses the trained model's per-node soft
predictions over the temporal/kNN edges:

    F^{0} = P_model
    F^{t+1} = alpha * neighbour_weighted_mean(F^{t}) + (1 - alpha) * P_model

which sharpens coverage near labeled regions (classic label-spreading, Zhou et al. 2004,
seeded by the model's own outputs rather than one-hot labels so it is a *smoothing*
complement, not a replacement).

It is kept ONLY if it earns its place: ``propagation_contribution`` runs the gate's held-out
k-fold twice — raw model vs diffused — and reports Δκ vs the LLM consensus; retain only when
Δκ >= +0.02 (per the master plan's "measured" rule). Pure numpy on top of the graph tensors.
"""

from typing import Optional


def neighbour_weighted_mean(F, edge_index, edge_weight, n_nodes):
    """(sum_j w_ij F_j) / (sum_j w_ij) bucketed by destination — numpy mirror of the SAGE agg."""
    import numpy as np
    src = edge_index[0]
    dst = edge_index[1]
    if len(src) == 0:
        return np.zeros_like(F)
    w = edge_weight if edge_weight is not None else np.ones(len(src), dtype=F.dtype)
    num = np.zeros_like(F)
    np.add.at(num, dst, F[src] * w[:, None])
    den = np.zeros(n_nodes, dtype=F.dtype)
    np.add.at(den, dst, w)
    den = np.clip(den, 1e-12, None)
    return num / den[:, None]


def propagate(P, graph, alpha: float = 0.5, n_iter: int = 20):
    """Diffuse soft predictions ``P`` ([N,C]) over the graph; returns row-normalized [N,C]."""
    import numpy as np

    P = np.asarray(P, dtype=np.float64)
    ei = graph.edge_index.detach().cpu().numpy() if hasattr(graph.edge_index, 'detach') \
        else np.asarray(graph.edge_index)
    ew = (graph.edge_weight.detach().cpu().numpy()
          if getattr(graph, 'edge_weight', None) is not None and hasattr(graph.edge_weight, 'detach')
          else (np.asarray(graph.edge_weight) if getattr(graph, 'edge_weight', None) is not None else None))
    n = P.shape[0]
    a = float(alpha)
    F = P.copy()
    for _ in range(int(n_iter)):
        F = a * neighbour_weighted_mean(F, ei, ew, n) + (1.0 - a) * P
    s = F.sum(axis=1, keepdims=True)
    s = np.clip(s, 1e-12, None)
    return F / s


def propagation_contribution(graph, targets, config, df_all, n_vce: int = 0) -> dict:
    """Measured test: does diffusing the model's outputs raise held-out VAAMR κ vs the LLM?

    Mirrors the reliability gate's k-fold: for each fold the held-out VAAMR rows are masked,
    a fresh model is trained, and the held-out nodes are predicted both ways — raw model
    argmax vs argmax of the diffused soft predictions. Both are scored with Cohen's κ against
    ``final_label``. Retain propagation only when Δκ >= +0.02.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .train import train_model, _subset_targets, _replace_seed
    from .triangulation import _kappa

    node_ids = list(graph.node_ids)
    v_idx = targets.get('vaamr_idx')
    n_v = int(v_idx.numel()) if v_idx is not None else 0
    if n_v == 0:
        return {'status': 'skipped: no labeled VAAMR nodes'}

    ref = {str(r.get('segment_id')): r.get('final_label') for _, r in df_all.iterrows()}

    def _ref_int(sid):
        try:
            return int(ref.get(str(sid)))
        except (ValueError, TypeError):
            return None

    folds = int(getattr(config, 'validation_folds', 5) or 1)
    rng = np.random.default_rng(int(config.seed))
    perm = rng.permutation(n_v)
    if folds <= 1:
        h = max(1, int(round(n_v * float(getattr(config, 'validation_holdout', 0.2)))))
        fold_list = [perm[:h]]
    else:
        fold_list = [f for f in np.array_split(perm, min(folds, n_v)) if len(f)]

    alpha = float(getattr(config, 'propagation_alpha', 0.5))
    n_it = int(getattr(config, 'propagation_iters', 20))

    model_pairs, prop_pairs = [], []
    for fi, hold in enumerate(fold_list):
        keep_v = sorted(set(range(n_v)) - set(int(i) for i in hold))
        fold_cfg = _replace_seed(config, int(config.seed) + 1 + fi)
        sub = _subset_targets(targets, keep_v, list(range(
            int(targets['purer_idx'].numel()) if targets.get('purer_idx') is not None else 0)))
        model, _ = train_model(graph, sub, fold_cfg, n_vce=n_vce)
        from .inference import _graph_tensors_on_model_device
        x, ei, ew, eti = _graph_tensors_on_model_device(model, graph)
        model.eval()
        with torch.no_grad():
            out = model(x, ei, ew, eti)
        if 'soft_vaamr' not in out:
            continue
        P = F.softmax(out['soft_vaamr'], dim=1).cpu().numpy()
        P_diff = propagate(P, graph, alpha=alpha, n_iter=n_it)
        for pos in hold:
            gi = int(v_idx[int(pos)])
            r = _ref_int(node_ids[gi])
            if r is None:
                continue
            model_pairs.append((int(P[gi].argmax()), r))
            prop_pairs.append((int(P_diff[gi].argmax()), r))

    if not model_pairs:
        return {'status': 'skipped: no held-out predictions'}

    k_model = _kappa([p for p, _ in model_pairs], [r for _, r in model_pairs])
    k_prop = _kappa([p for p, _ in prop_pairs], [r for _, r in prop_pairs])
    delta = (k_prop - k_model) if (k_model is not None and k_prop is not None) else None
    if delta is None:
        verdict, recommend = 'inconclusive', False
    elif delta >= 0.02:
        verdict, recommend = 'propagation_helps', True
    elif delta <= -0.02:
        verdict, recommend = 'propagation_harms', False
    else:
        verdict, recommend = 'propagation_neutral', False
    return {
        'n_heldout': len(model_pairs),
        'alpha': alpha, 'iters': n_it,
        'kappa_model': k_model,
        'kappa_propagated': k_prop,
        'delta_kappa': delta,
        'verdict': verdict,
        'recommend_propagation': recommend,
    }


def write_propagation_report(result: dict, output_dir: str) -> str:
    """Human-readable A4 label-propagation report → 06_reports/07_gnn/."""
    import os
    from process import output_paths as _paths

    def _k(x):
        return 'n/a' if not isinstance(x, (int, float)) else f"{x:+.3f}"

    W = 78
    L = ["=" * W, "LABEL-PROPAGATION CONTRIBUTION (A4)", "=" * W, ""]
    if result.get('status'):
        L.append(f"  {result['status']}")
        L.append("")
    else:
        L.append("Does diffusing the trained model's soft predictions over the temporal/kNN")
        L.append("edges sharpen held-out agreement with the LLM consensus? Measured out-of-sample")
        L.append("(same k-fold as the gate). Propagation is kept only if Δκ >= +0.02.")
        L.append("")
        L.append(f"  alpha={result.get('alpha')}  iters={result.get('iters')}  "
                 f"held-out n={result.get('n_heldout')}")
        L.append("")
        L.append(f"    κ raw model      : {_k(result.get('kappa_model'))}")
        L.append(f"    κ + propagation  : {_k(result.get('kappa_propagated'))}")
        L.append(f"    Δκ               : {_k(result.get('delta_kappa'))}")
        L.append("")
        L.append("=" * W)
        L.append(f"  VERDICT: {result.get('verdict')}    "
                 f"RECOMMEND propagation ON: {'YES' if result.get('recommend_propagation') else 'NO'}")
        L.append("=" * W)
    rep_dir = _paths.reports_gnn_dir(output_dir)
    os.makedirs(rep_dir, exist_ok=True)
    path = os.path.join(rep_dir, 'label_propagation.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path
