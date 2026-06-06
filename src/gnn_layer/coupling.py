"""
gnn_layer/coupling.py
---------------------
Capability E — inductive participant<->therapist coupling.

Decomposes cue-block (therapist) embeddings into latent factors and, where each
factor co-varies with subsequent participant forward movement, reports it. Factors
are optionally *named* against an inline Common-Factors / Intervention-Concepts
reference lexicon (defined here; NOT imported from any external dataset) — common
factors are rediscovered from the data, never imposed as labels.
"""

from typing import Dict, List, Optional


# Inline interpretive lexicon (used only to NAME discovered factors, never to classify).
CF_IC_REFERENCE = {
    'bond': "warmth, empathy, trust, relational connection between therapist and participant",
    'goal_alignment': "shared agreement on the goals or aims of the therapeutic work",
    'task_agreement': "shared agreement on the tasks and methods used to pursue the goals",
    'empathy_acceptance_regard': "empathic, accepting, non-judgmental positive regard",
    'collaboration_partnership': "collaborative, partnership-oriented, autonomy-supportive stance",
}


def extract_latent_factors(cue_embeddings, forward_outcome=None, config=None) -> dict:
    """PCA decomposition of cue embeddings; per-factor association with forward movement.

    PCA (not NMF) because embeddings are signed. Returns:
      {'components': [n_factors, D], 'block_scores': [n_blocks, n_factors],
       'factor_forward_corr': [n_factors] correlation with forward_outcome (or None)}
    """
    import numpy as np
    from sklearn.decomposition import PCA
    n = cue_embeddings.shape[0]
    if n < 2:
        return {'components': None, 'block_scores': None, 'factor_forward_corr': None}
    n_factors = max(1, min(int(getattr(config, 'n_latent_factors', 5)), n, cue_embeddings.shape[1]))
    pca = PCA(n_components=n_factors, random_state=int(getattr(config, 'seed', 42)))
    scores = pca.fit_transform(cue_embeddings)
    corr = None
    if forward_outcome is not None and len(forward_outcome) == n and len(np.unique(forward_outcome)) > 1:
        y = np.asarray(forward_outcome, dtype=float)
        corr = []
        for f in range(n_factors):
            s = scores[:, f]
            if s.std() > 0:
                corr.append(float(np.corrcoef(s, y)[0, 1]))
            else:
                corr.append(0.0)
    return {
        'components': pca.components_,
        'block_scores': scores,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'factor_forward_corr': corr,
    }


def factor_exemplars(scores, block_rows, k: int = 3) -> Dict[int, list]:
    """Top-k blocks loading most positively on each factor (for human interpretation)."""
    import numpy as np
    out: Dict[int, list] = {}
    if scores is None:
        return out
    for f in range(scores.shape[1]):
        order = np.argsort(scores[:, f])[::-1][:k]
        out[int(f)] = [
            {'from_stage': block_rows[i]['from_stage'], 'to_stage': block_rows[i]['to_stage'],
             'session_id': block_rows[i]['session_id'], 'from_seg_id': block_rows[i]['from_seg_id']}
            for i in order
        ]
    return out


def interpret_factors(factors: dict, exemplar_texts_by_factor: Optional[Dict[int, List[str]]],
                      config) -> dict:
    """Name each factor against CF_IC_REFERENCE by embedding similarity (best-effort).

    Requires the embedding model; if unavailable this returns an empty interpretation
    with a note rather than failing (the factors themselves remain available).
    """
    if not getattr(config, 'interpret_against_cf_ic', False) or not exemplar_texts_by_factor:
        return {'note': 'CF/IC interpretation skipped (disabled or no exemplar texts).'}
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        from .embeddings import embed_anchor_texts
        ref_keys = list(CF_IC_REFERENCE.keys())
        ref_vecs = embed_anchor_texts([CF_IC_REFERENCE[k] for k in ref_keys], config)
        out = {}
        for f, texts in exemplar_texts_by_factor.items():
            if not texts:
                continue
            fac_vec = embed_anchor_texts(texts, config).mean(axis=0, keepdims=True)
            sims = cosine_similarity(fac_vec, ref_vecs)[0]
            j = int(np.argmax(sims))
            out[int(f)] = {'nearest_cf_ic': ref_keys[j], 'similarity': round(float(sims[j]), 3)}
        return out
    except Exception as e:
        return {'note': f'CF/IC interpretation unavailable: {e}'}
