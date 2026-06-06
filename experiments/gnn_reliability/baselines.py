"""
experiments/gnn_reliability/baselines.py
----------------------------------------
Non-GNN baseline arms for the VAAMR reliability battery (design_decisions.md
§1B / §5 / §6). These answer the pivotal question: *does a graph neural net even
beat a simple baseline at n≈205?*

Two arm runners, both honoring the harness "arm runner" contract
``run(df_all, embeddings, folds, config) -> dict[str, int]``:

  * ``run_linear_probe``  — arms A1 / A1w / A1n: a calibrated logistic-regression
    probe on the (L2-normalized) Qwen embeddings. No graph at all.
  * ``run_correct_smooth`` — arm A2: Correct-&-Smooth (Huang et al., ICLR 2021)
    layered on the probe's soft predictions, propagated over the QRA kNN+temporal
    graph. The simplest "does graph propagation help a feature-only base?" test.

Contract (matches the harness; do NOT import harness.py — it may not exist yet):
  - ``df_all``   : pandas frame, cols incl. ``segment_id`` (str), ``participant_id``,
                   ``speaker``, ``final_label`` (0..4, or NaN = "No code").
  - ``embeddings``: ``{segment_id(str): np.ndarray[D]}`` for ALL segments
                    (participant + therapist). Assumed Qwen, already ~unit-norm;
                    we L2-normalize defensively anyway.
  - ``folds``    : ``{segment_id(str): fold_idx(int)}`` over the LABELED participant
                   segments (participant-grouped CV, built once by the harness).
  - ``config``   : a ``GnnLayerConfig``. We read ``vaamr_n_classes`` (5 | 6) and
                   ``vaamr_class_balance`` (bool), plus ``knn_k`` (via build_graph).
  - RETURNS      : ``{segment_id: predicted_class_int}`` — out-of-fold predictions
                   for EVERY labeled participant segment, in ``0..n_classes-1``
                   (class 5 = "No code" when ``vaamr_n_classes == 6``).

Design notes / documented choices
  * 5-class: labels = ``final_label`` over participants with a non-null label.
    6-class: ALSO include participants whose ``final_label`` is null as class 5
    ("No code"), so the probe can predict abstention.
  * ``class_weight = 'balanced' if config.vaamr_class_balance else None``.
  * Folds for the 6-class "No code" rows: the harness folds only the labeled-205,
    so the No-code rows are usually ABSENT from ``folds``. We re-derive their fold
    from the *same participant grouping* — every fold here is participant-pure, so
    each participant maps to exactly one fold; a No-code row inherits its
    participant's fold. A participant that is ENTIRELY No-code (no labeled row, so
    no fold in ``folds``) is assigned a fold by deterministic round-robin over the
    sorted unassigned-participant list. If a No-code row already appears in
    ``folds`` we honor that directly. This keeps every participant in exactly one
    fold (no train/test leakage).
  * Correct-&-Smooth deviations (kept simple + documented): row-normalized
    adjacency D^{-1}A (reusing ``gnn_layer.propagation.neighbour_weighted_mean``)
    rather than the paper's symmetric D^{-1/2}AD^{-1/2}; the spec's additive
    correction ``Z + Ê`` (no Huang autoscale / FDiff scaling); α = 0.8 and 50
    iterations for BOTH the Correct and the Smooth diffusion. The Smooth step
    clamps training nodes to their true one-hot (the canonical C&S "best guess"),
    so held-out nodes inherit their neighbours' corrected/true labels.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


# StandardScaler hook. Default OFF: the Qwen features are already unit-norm, so
# standardizing high-D embeddings tends to amplify near-constant dims. When True,
# a StandardScaler is fit on each fold's TRAIN rows only (leakage-free) and applied
# consistently to every matrix the fold scores (probe rows AND all C&S graph nodes).
_USE_STANDARD_SCALER = False

# Correct-&-Smooth hyperparameters (documented defaults; see module docstring).
_CS_ALPHA = 0.8
_CS_ITERS = 50


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------
def _np(t):
    """Torch tensor / array-like -> numpy (detached, cpu)."""
    import numpy as np
    if t is None:
        return None
    if hasattr(t, 'detach'):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _is_labeled(lab) -> bool:
    """True iff ``lab`` is a real (non-null) VAAMR label."""
    import pandas as pd
    if lab is None:
        return False
    try:
        return not pd.isna(lab)
    except (TypeError, ValueError):
        return False


def _as_class(lab) -> int:
    return int(round(float(lab)))


def _stack_l2(embeddings: Dict[str, "object"], ids: List[str]):
    """Stack the given segment embeddings into an [n, D] float64 matrix, L2-normalized."""
    import numpy as np
    from sklearn.preprocessing import normalize
    mat = np.stack([np.asarray(embeddings[s], dtype=np.float64) for s in ids], axis=0)
    return normalize(mat, norm='l2', axis=1)


# ---------------------------------------------------------------------------
# target selection + fold resolution (shared by both arms)
# ---------------------------------------------------------------------------
def _prepare_labeled(df_all, embeddings, config) -> Tuple[List[str], List[int], int]:
    """The labeled-participant target set this arm predicts.

    5-class -> participant rows with a non-null ``final_label`` (labels 0..4).
    6-class -> ALSO participant rows whose ``final_label`` is null, as class
               ``n_classes-1`` ( == 5, "No code").
    Only rows that actually have an embedding are kept (can't featurize otherwise).
    Returns (seg_ids, labels, n_classes), aligned.
    """
    n_classes = int(getattr(config, 'vaamr_n_classes', 5) or 5)
    no_code_class = n_classes - 1  # 5 when n_classes==6 (real stages are 0..4)
    seg_ids: List[str] = []
    labels: List[int] = []
    for _, r in df_all.iterrows():
        if str(r.get('speaker', '') or '') != 'participant':
            continue
        sid = str(r.get('segment_id'))
        if sid not in embeddings:
            continue
        lab = r.get('final_label')
        if _is_labeled(lab):
            seg_ids.append(sid)
            labels.append(_as_class(lab))
        elif n_classes >= 6:
            seg_ids.append(sid)
            labels.append(no_code_class)
        # else: 5-class drops No-code participant rows (cannot emit No-code)
    return seg_ids, labels, n_classes


def _resolve_folds(seg_ids: List[str], df_all, folds: Dict[str, int]
                   ) -> Tuple[Dict[str, int], List[int]]:
    """Map EVERY target seg_id -> a fold index (see module docstring).

    Honors any seg_id already present in ``folds``; otherwise inherits its
    participant's fold (folds are assumed participant-pure); a participant with no
    labeled row at all is assigned by deterministic round-robin over ``fold_list``.
    """
    part_of: Dict[str, str] = {
        str(r.get('segment_id')): str(r.get('participant_id'))
        for _, r in df_all.iterrows()
    }
    fold_list = sorted({int(v) for v in folds.values()}) or [0]

    # participant -> fold, learned from the harness folds (participant-pure assumption)
    p2f: Dict[str, int] = {}
    for sid, fi in folds.items():
        p = part_of.get(str(sid))
        if p is not None:
            p2f.setdefault(p, int(fi))

    fold_of: Dict[str, int] = {}
    orphan_parts: List[str] = []  # participants with no labeled fold (entirely No-code)
    for s in seg_ids:
        if s in folds:
            fold_of[s] = int(folds[s])
        else:
            p = part_of.get(s)
            if p in p2f:
                fold_of[s] = p2f[p]
            else:
                orphan_parts.append(p)

    # deterministic round-robin for entirely-unlabeled participants
    for i, p in enumerate(sorted({x for x in orphan_parts if x is not None})):
        p2f[p] = fold_list[i % len(fold_list)]
    for s in seg_ids:
        if s not in fold_of:
            fold_of[s] = p2f.get(part_of.get(s), fold_list[0])
    return fold_of, fold_list


def _iter_folds(seg_ids: List[str], fold_of: Dict[str, int], fold_list: List[int]):
    """Yield (fold_idx, train_ids, test_ids) for the grouped CV; shared by both arms."""
    sid_set = list(seg_ids)
    for f in fold_list:
        test = [s for s in sid_set if fold_of[s] == f]
        if not test:
            continue
        train = [s for s in sid_set if fold_of[s] != f]
        yield f, train, test


# ---------------------------------------------------------------------------
# probe fitting (shared)
# ---------------------------------------------------------------------------
class _ConstantClf:
    """Degenerate fallback when a training fold has <2 classes (predicts that class)."""
    def __init__(self, cls: int):
        import numpy as np
        self.classes_ = np.array([int(cls)])

    def predict_proba(self, X):
        import numpy as np
        return np.ones((len(X), 1), dtype=np.float64)


def _fit_probe(X_train, y_train, config, n_classes):
    """Fit the logistic-regression probe (+ optional StandardScaler) on one fold's train rows.

    Returns (scaler_or_None, classifier). ``class_weight`` follows
    ``config.vaamr_class_balance``.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    scaler = None
    Xt = X_train
    if _USE_STANDARD_SCALER:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X_train)
        Xt = scaler.transform(X_train)

    if len(np.unique(y_train)) < 2:
        return scaler, _ConstantClf(int(np.unique(y_train)[0]))

    cw = 'balanced' if getattr(config, 'vaamr_class_balance', False) else None
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight=cw)
    clf.fit(Xt, y_train)
    return scaler, clf


def _predict_proba_full(scaler, clf, X, n_classes):
    """predict_proba expanded to the full [n, n_classes] label space (zeros for unseen classes)."""
    import numpy as np
    Xt = scaler.transform(X) if scaler is not None else X
    proba = clf.predict_proba(Xt)
    full = np.zeros((len(X), n_classes), dtype=np.float64)
    for j, c in enumerate(clf.classes_):
        ci = int(c)
        if 0 <= ci < n_classes:
            full[:, ci] = proba[:, j]
    return full


# ===========================================================================
# Arm A1 / A1w / A1n — linear probe (no graph)
# ===========================================================================
def run_linear_probe(df_all, embeddings, folds, config) -> Dict[str, int]:
    """Logistic-regression probe on L2-normalized Qwen embeddings (grouped CV).

    See module docstring for the contract. Returns out-of-fold predicted classes
    for every labeled participant segment.
    """
    import numpy as np

    seg_ids, labels, n_classes = _prepare_labeled(df_all, embeddings, config)
    if not seg_ids:
        return {}
    label_of = dict(zip(seg_ids, labels))
    fold_of, fold_list = _resolve_folds(seg_ids, df_all, folds)

    preds: Dict[str, int] = {}
    for _f, train_ids, test_ids in _iter_folds(seg_ids, fold_of, fold_list):
        scaler, clf = _fit_probe(
            _stack_l2(embeddings, train_ids),
            [label_of[s] for s in train_ids], config, n_classes)
        proba = _predict_proba_full(scaler, clf, _stack_l2(embeddings, test_ids), n_classes)
        for s, row in zip(test_ids, proba):
            preds[s] = int(np.argmax(row))
    return preds


# ===========================================================================
# Arm A2 — Correct & Smooth on the probe (Huang et al., ICLR 2021)
# ===========================================================================
def _diffuse(signal, edge_index, edge_weight, n_nodes, alpha, n_iter):
    """Label-spreading recursion  F^{t+1} = (1-α)·F0 + α·(D^{-1}A)·F^{t}, F0=signal.

    Reuses ``gnn_layer.propagation.neighbour_weighted_mean`` (the row-normalized
    D^{-1}A operator). Returns the diffused signal (NOT row-normalized — residuals
    may be negative).
    """
    import numpy as np
    from gnn_layer.propagation import neighbour_weighted_mean
    F0 = np.asarray(signal, dtype=np.float64)
    F = F0.copy()
    a = float(alpha)
    for _ in range(int(n_iter)):
        F = (1.0 - a) * F0 + a * neighbour_weighted_mean(F, edge_index, edge_weight, n_nodes)
    return F


def run_correct_smooth(df_all, embeddings, folds, config) -> Dict[str, int]:
    """Correct-&-Smooth over the QRA kNN+temporal graph, on the probe's soft base.

    Transductive per fold: the probe (trained on the OTHER folds' labeled rows)
    produces a soft base for ALL graph nodes; the residual (true-base) is diffused
    from the TRAIN labeled participant nodes (Correct), then the corrected labels
    (with train nodes clamped to truth) are diffused again (Smooth). The held-out
    fold's labeled participant nodes are masked, so they inherit neighbours'
    corrected/true labels. Returns OOF argmax for every labeled participant segment.
    """
    import numpy as np
    from gnn_layer.graph_builder import build_graph

    seg_ids, labels, n_classes = _prepare_labeled(df_all, embeddings, config)
    if not seg_ids:
        return {}
    label_of = dict(zip(seg_ids, labels))
    fold_of, fold_list = _resolve_folds(seg_ids, df_all, folds)

    # ---- graph over ALL segments (participant + therapist) ----
    graph = build_graph(df_all, embeddings, config)
    index_of = graph.index_of
    n_nodes = len(graph.node_ids)
    X_nodes = _stack_l2_from_matrix(_np(graph.x))               # [N, D], L2-normalized
    ei = _np(graph.edge_index)
    ew = _np(graph.edge_weight)
    ei = ei.astype(np.int64) if ei is not None and ei.size else np.zeros((2, 0), dtype=np.int64)
    ew = ew.astype(np.float64) if ew is not None and ew.size else None

    preds: Dict[str, int] = {}
    for _f, train_ids, test_ids in _iter_folds(seg_ids, fold_of, fold_list):
        # base soft predictions for EVERY node (therapist / No-target nodes carry base preds)
        scaler, clf = _fit_probe(
            _stack_l2(embeddings, train_ids),
            [label_of[s] for s in train_ids], config, n_classes)
        Z = _predict_proba_full(scaler, clf, X_nodes, n_classes)   # [N, C]

        # train mask + one-hot truth on the TRAIN labeled participant nodes only
        train_mask = np.zeros(n_nodes, dtype=bool)
        Y = np.zeros((n_nodes, n_classes), dtype=np.float64)
        for s in train_ids:
            gi = index_of[s]
            train_mask[gi] = True
            Y[gi, label_of[s]] = 1.0

        # ---- (1) CORRECT: diffuse the residual, add to base ----
        E = np.zeros((n_nodes, n_classes), dtype=np.float64)
        E[train_mask] = Y[train_mask] - Z[train_mask]
        E_hat = _diffuse(E, ei, ew, n_nodes, _CS_ALPHA, _CS_ITERS)
        Z_corr = Z + E_hat

        # ---- (2) SMOOTH: clamp train nodes to truth, diffuse the corrected labels ----
        G = Z_corr.copy()
        G[train_mask] = Y[train_mask]
        G_hat = _diffuse(G, ei, ew, n_nodes, _CS_ALPHA, _CS_ITERS)

        for s in test_ids:
            preds[s] = int(np.argmax(G_hat[index_of[s]]))
    return preds


def _stack_l2_from_matrix(mat):
    """L2-normalize the rows of an already-stacked [N, D] matrix (defensive)."""
    import numpy as np
    from sklearn.preprocessing import normalize
    return normalize(np.asarray(mat, dtype=np.float64), norm='l2', axis=1)
