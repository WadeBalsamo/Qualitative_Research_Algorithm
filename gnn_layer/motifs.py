"""
gnn_layer/motifs.py
-------------------
Capability B — cue granularization + emergent-motif discovery.

Keeps the continuous cue-block embedding (instead of collapsing each block to one of
five PURER moves before measuring influence) to:
  1. cluster cue-block embeddings into emergent therapist-language motifs;
  2. score each motif's influence on forward VAAMR transitions, conditioned on
     from_stage (logistic regression with from_stage one-hot);
  3. flag influential motifs poorly explained by PURER/microskill (candidate new
     constructs) and surface exemplar cues.

sklearn imported lazily.
"""

from typing import Dict, List, Optional


def cluster_cue_motifs(cue_embeddings, config):
    """Cluster cue-block embeddings → integer motif id per block (np.ndarray)."""
    import numpy as np
    from sklearn.cluster import KMeans
    n = cue_embeddings.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=int)
    k = max(1, min(int(config.n_motif_clusters), n))
    if k == 1:
        return np.zeros(n, dtype=int)
    km = KMeans(n_clusters=k, n_init=10, random_state=int(config.seed))
    return km.fit_predict(cue_embeddings)


def score_motif_influence(cue_embeddings, from_stages, forward_outcome, motif_ids,
                          config) -> Dict[int, dict]:
    """Per-motif influence on forward transitions, conditioned on from_stage.

    Fits LogisticRegression(forward_outcome ~ [cue_embedding, from_stage one-hot]) and
    reports, per motif, the mean predicted P(forward) divided by the global base rate
    (a lift), plus block count.
    """
    import numpy as np
    out: Dict[int, dict] = {}
    n = cue_embeddings.shape[0]
    if n == 0:
        return out
    base_rate = float(np.mean(forward_outcome)) if n else 0.0

    # from_stage one-hot (5 stages)
    fs = np.asarray(from_stages, dtype=int)
    onehot = np.zeros((n, 5), dtype=np.float32)
    for i, s in enumerate(fs):
        if 0 <= s < 5:
            onehot[i, s] = 1.0
    feats = np.concatenate([cue_embeddings, onehot], axis=1)

    pred = None
    y = np.asarray(forward_outcome, dtype=int)
    if len(np.unique(y)) >= 2 and n >= 4:
        from sklearn.linear_model import LogisticRegression
        try:
            clf = LogisticRegression(max_iter=1000)
            clf.fit(feats, y)
            pred = clf.predict_proba(feats)[:, 1]
        except Exception:
            pred = None
    if pred is None:
        pred = y.astype(float)  # fallback: empirical

    for m in np.unique(motif_ids):
        sel = motif_ids == m
        cnt = int(sel.sum())
        mean_pred = float(np.mean(pred[sel])) if cnt else 0.0
        influence = (mean_pred / base_rate) if base_rate > 0 else 0.0
        out[int(m)] = {
            'influence': round(influence, 3),
            'mean_pred_forward': round(mean_pred, 3),
            'n_blocks': cnt,
        }
    return out


def annotate_label_purity(motif_ids, purer_labels, micro_labels) -> Dict[int, dict]:
    """Per motif: dominant PURER move + its purity, and dominant microskill + purity."""
    import numpy as np
    from collections import Counter
    out: Dict[int, dict] = {}
    motif_ids = np.asarray(motif_ids)
    for m in np.unique(motif_ids):
        sel = np.where(motif_ids == m)[0]
        pl = [purer_labels[i] for i in sel if i < len(purer_labels) and purer_labels[i] is not None]
        ml = []
        for i in sel:
            if i < len(micro_labels) and micro_labels[i]:
                ml.extend(micro_labels[i])
        info = {}
        if pl:
            c = Counter(pl); dom, cnt = c.most_common(1)[0]
            info['dominant_purer'] = int(dom); info['purer_purity'] = round(cnt / len(pl), 3)
        else:
            info['dominant_purer'] = None; info['purer_purity'] = 0.0
        if ml:
            c = Counter(ml); dom, cnt = c.most_common(1)[0]
            info['dominant_microskill'] = dom; info['microskill_purity'] = round(cnt / len(ml), 3)
        else:
            info['dominant_microskill'] = None; info['microskill_purity'] = 0.0
        out[int(m)] = info
    return out


def flag_emergent_motifs(motif_stats: Dict[int, dict], purity: Dict[int, dict],
                         config) -> List[int]:
    """Influential (>= min_motif_influence) but low PURER/microskill purity, enough blocks."""
    flagged = []
    for m, s in motif_stats.items():
        if s['n_blocks'] < int(config.motif_min_block_count):
            continue
        if s['influence'] < float(config.min_motif_influence):
            continue
        p = purity.get(m, {})
        low_purer = p.get('purer_purity', 0.0) < 0.5
        low_micro = p.get('microskill_purity', 0.0) < 0.5
        if low_purer and low_micro:
            flagged.append(m)
    return sorted(flagged)


def select_motif_exemplars(motif_ids, cue_embeddings, block_rows, k: int = 3) -> Dict[int, list]:
    """Per motif, pick the k blocks nearest the motif centroid; return their from/to + ids."""
    import numpy as np
    out: Dict[int, list] = {}
    motif_ids = np.asarray(motif_ids)
    for m in np.unique(motif_ids):
        sel = np.where(motif_ids == m)[0]
        if len(sel) == 0:
            continue
        centroid = cue_embeddings[sel].mean(axis=0)
        d = np.linalg.norm(cue_embeddings[sel] - centroid, axis=1)
        order = sel[np.argsort(d)][:k]
        out[int(m)] = [
            {
                'from_seg_id': block_rows[i]['from_seg_id'],
                'to_seg_id': block_rows[i]['to_seg_id'],
                'from_stage': block_rows[i]['from_stage'],
                'to_stage': block_rows[i]['to_stage'],
                'session_id': block_rows[i]['session_id'],
            }
            for i in order
        ]
    return out
