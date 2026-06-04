"""
gnn_layer/ablation.py
---------------------
Capability D — principled ablation & sub-typing.

- run_ablation: remove a construct head from the objective set, retrain, and report
  the change in best training loss vs the full model (an empirical test of whether
  that construct family carries signal — e.g. "is VCE superfluous?").
- discover_subtypes: cluster segment embeddings within each VAAMR stage / PURER move
  to surface emergent sub-types from the data (not imposed).
"""

from copy import deepcopy
from typing import Dict, List, Optional


_ABLATE_TO_OBJECTIVE = {
    'vce': 'vce_multilabel',
    'purer': 'purer',
    'microskill': 'microskill_multilabel',
}


def run_ablation(graph, targets: dict, config, ablate: str,
                 n_vce: int = 0, n_microskill: int = 8) -> dict:
    """Retrain with one head removed; return loss deltas vs the full model."""
    from .train import train_model
    obj = _ABLATE_TO_OBJECTIVE.get(ablate)
    full_cfg = deepcopy(config)
    _, full_metrics = train_model(graph, targets, full_cfg, n_vce=n_vce, n_microskill=n_microskill)
    abl_cfg = deepcopy(config)
    if obj and obj in abl_cfg.objectives:
        abl_cfg.objectives = [o for o in abl_cfg.objectives if o != obj]
    _, abl_metrics = train_model(graph, targets, abl_cfg, n_vce=n_vce, n_microskill=n_microskill)
    full_loss = full_metrics.get('best_loss', float('nan'))
    abl_loss = abl_metrics.get('best_loss', float('nan'))
    return {
        'ablate': ablate,
        'objective_removed': obj,
        'best_loss_full': full_loss,
        'best_loss_ablated': abl_loss,
        'delta': (abl_loss - full_loss),
    }


def discover_subtypes(seg_embeddings_by_id: Dict[str, "object"], df_all,
                      by: str = 'vaamr_stage', k_per_group: int = 3,
                      config=None) -> dict:
    """Cluster segment embeddings within each stage/move to surface emergent sub-types."""
    import numpy as np
    from sklearn.cluster import KMeans

    if by == 'purer_move':
        col, speaker = 'purer_primary', 'therapist'
    else:
        col, speaker = 'final_label', 'participant'
    if col not in df_all.columns:
        return {}
    sub = df_all[df_all.get('speaker', speaker) == speaker] if 'speaker' in df_all.columns else df_all

    out: Dict[str, dict] = {}
    seed = int(getattr(config, 'seed', 42)) if config is not None else 42
    for group_val, grp in sub.groupby(col):
        ids = [str(s) for s in grp.get('segment_id', []).tolist()]
        embs = [seg_embeddings_by_id[i] for i in ids if i in seg_embeddings_by_id]
        kept = [i for i in ids if i in seg_embeddings_by_id]
        if len(embs) < 2:
            continue
        X = np.stack(embs, axis=0)
        k = max(1, min(int(k_per_group), len(embs)))
        labels = (np.zeros(len(embs), dtype=int) if k == 1
                  else KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X))
        clusters = {}
        for ci in np.unique(labels):
            members = [kept[j] for j in range(len(kept)) if labels[j] == ci]
            clusters[int(ci)] = {'n': len(members), 'exemplar_ids': members[:5]}
        out[str(group_val)] = {'n_total': len(embs), 'clusters': clusters}
    return out
