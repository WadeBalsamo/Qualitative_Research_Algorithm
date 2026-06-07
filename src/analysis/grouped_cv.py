"""
analysis/grouped_cv.py
----------------------
Participant-grouped cross-validation + clustered-bootstrap κ confidence intervals.

This is the production home of the *only* leak-free reliability protocol the QRA
classifier work established (methodology §5.3 correction): hold out WHOLE participants,
never random segments. Random k-fold over a transcript leaks — a held-out segment's
same-participant neighbours are visible in training — and inflated the pilot gate from a
participant-grouped κ ≈ 0.05 to a random-fold κ ≈ 0.25.

Both helpers were validated in the reliability/distillation campaigns
(``experiments/gnn_reliability/harness.py``) and are lifted here verbatim in behaviour so
production gates (the probe scaler, any future learned classifier) reproduce the campaign
numbers. They reuse the shipping stats primitives (``analysis.stats.cluster_bootstrap_ci``,
``analysis.irr_stats.cohen_kappa``) — nothing is re-derived.
"""

from typing import Dict, List, Optional

import numpy as np

from . import irr_stats
from . import stats as _stats


def labeled_participants(df_all):
    """Participant rows carrying an integer VAAMR ``final_label`` (the labeled subset).

    Returns a copy with ``final_label`` cast to int and ``segment_id`` / ``participant_id``
    as str — the same frame ``build_folds`` strata over.
    """
    part = df_all[df_all['speaker'] == 'participant'].copy()
    part = part[part['final_label'].notna()].copy()
    part['final_label'] = part['final_label'].astype(float).astype(int)
    part['segment_id'] = part['segment_id'].astype(str)
    part['participant_id'] = part['participant_id'].astype(str)
    return part


def build_folds(df_all, n_folds: int = 5, seed: int = 42,
                verbose: bool = False) -> Dict[str, int]:
    """Participant-grouped + stratified folds over the LABELED participant segments.

    ``sklearn.model_selection.StratifiedGroupKFold`` (groups=participant_id,
    y=final_label) so no participant is in train+test and the rare-stage mix is held
    fold-to-fold. Deterministic given ``seed``. Returns ``{segment_id: fold_idx}`` over
    the labeled participant segments.

    Falls back to a single fold when there are too few participants/classes to split
    (so callers degrade rather than raise on tiny / synthetic corpora).
    """
    from sklearn.model_selection import StratifiedGroupKFold

    lab = labeled_participants(df_all)
    if lab.empty:
        return {}
    sids = lab['segment_id'].to_numpy()
    y = lab['final_label'].to_numpy()
    groups = lab['participant_id'].to_numpy()

    n_groups = len(set(groups.tolist()))
    n_min_class = int(np.min(np.unique(y, return_counts=True)[1])) if len(y) else 0
    k = min(int(n_folds), n_groups, max(2, n_min_class))
    if k < 2:
        return {str(s): 0 for s in sids}

    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_of: Dict[str, int] = {}
    for fi, (_train_idx, test_idx) in enumerate(sgkf.split(sids, y, groups)):
        for i in test_idx:
            fold_of[str(sids[i])] = fi

    if verbose:
        print(f"[grouped_cv] {len(fold_of)} labeled participant segments, "
              f"{n_groups} participants, {k} folds (StratifiedGroupKFold, seed={seed})")
    return fold_of


def kappa_cluster_ci(a: List[int], b: List[int], clusters: List,
                     seed: int = 42, n_boot: int = 2000) -> dict:
    """Participant-clustered bootstrap 95% CI for Cohen's κ between aligned label lists.

    Resamples WHOLE clusters (participants) so the per-item (a, b) pairing is preserved:
    each pair is packed into one finite float, the statistic unpacks and computes κ via
    ``irr_stats.cohen_kappa``. ``point`` is the plain κ over all items. Returns
    ``{point, lo, hi, n, n_clusters}``. VAAMR labels live in [-1, 5]; we shift +1 into
    [0, 6] and pack as ``(a+1)*10 + (b+1)`` ∈ [0, 66].
    """
    if len(a) < 2:
        return {'point': irr_stats.cohen_kappa(a, b), 'lo': None, 'hi': None,
                'n': len(a), 'n_clusters': len(set(clusters))}
    a_arr = np.asarray(a, dtype=int)
    b_arr = np.asarray(b, dtype=int)
    packed = ((a_arr + 1) * 10 + (b_arr + 1)).astype(float)

    def _stat(arr: np.ndarray) -> float:
        codes = arr.astype(int)
        aa = (codes // 10) - 1
        bb = (codes % 10) - 1
        k = irr_stats.cohen_kappa(aa.tolist(), bb.tolist())
        return float('nan') if k is None else float(k)

    res = _stats.cluster_bootstrap_ci(packed, list(clusters), statistic=_stat,
                                      n_boot=n_boot, seed=seed)
    res['point'] = irr_stats.cohen_kappa(a, b)
    return res


def per_class_recall_precision(pairs: List, names: Optional[Dict[int, str]] = None
                               ) -> List[dict]:
    """Per-class support/recall/precision/binary-κ from (pred, ref) integer pairs.

    Mirrors ``gnn_layer.classifier.validation._per_class`` so probe per-stage recall is
    comparable to the GNN/IRR reports. ``names`` maps class id → display name.
    """
    names = names or {}
    rows = []
    classes = sorted(set(r for _, r in pairs) | set(p for p, _ in pairs))
    for c in classes:
        support = sum(1 for _, r in pairs if r == c)
        tp = sum(1 for p, r in pairs if p == c and r == c)
        pred_c = sum(1 for p, _ in pairs if p == c)
        rows.append({
            'class_id': c, 'class_name': names.get(c, str(c)),
            'support': support,
            'recall': (tp / support) if support else None,
            'precision': (tp / pred_c) if pred_c else None,
            'kappa': irr_stats.cohen_kappa([1 if p == c else 0 for p, _ in pairs],
                                           [1 if r == c else 0 for _, r in pairs]),
        })
    return rows
