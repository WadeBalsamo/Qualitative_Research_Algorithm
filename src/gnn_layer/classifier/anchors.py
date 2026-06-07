"""
gnn_layer/anchors.py
--------------------
Construct-anchor nodes for the heterogeneous (Path-B) graph + the honest test of
whether they earn their place.

An anchor is a construct DEFINITION embedded in the same Qwen3 space as the
segments (a VAAMR stage, a PURER move, or a VCE code). Anchors attach to segments
by EMBEDDING SIMILARITY only — never by label — so they inject "this segment sits
near the Reappraisal definition" as graph topology WITHOUT leaking the LLM's
labels into the structure.

Cross-framework anchor<->anchor edges (VAAMR stage <-> VCE code) ARE derived from
the LLM-consensus co-occurrence lift, so they are NOT label-independent. They are
built only when ``include_vce_nodes`` is set, and exist for the ablation — never
for the independence substrate.

Critical guardrail (gnn-influence-to-execution.md, G2): anchors raise GNN<->LLM
agreement BY CONSTRUCTION — a segment near its own construct definition will tend
to predict that construct. Their value must therefore be judged on the
GNN<->HUMAN out-of-sample axis (see :func:`gnn_layer.ablation.anchor_contribution`),
never by a GNN<->LLM gain.

Anchor id scheme: ``anchor:vaamr:<theme_id>`` / ``anchor:purer:<theme_id>`` /
``anchor:vce:<code_id>``.
"""

from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Definition text — mirrors the per-construct block the LLM classifier sees.
# ---------------------------------------------------------------------------

def _theme_text(t) -> str:
    feats = '; '.join(getattr(t, 'prototypical_features', []) or [])
    return (
        f"{t.name}: {t.definition} "
        f"Prototypical features: {feats}. "
        f"Key distinction: {getattr(t, 'distinguishing_criteria', '') or ''}."
    ).replace('\n', ' ').replace('  ', ' ').strip()


def _code_text(c) -> str:
    return (
        f"{c.category}: {c.description} "
        f"Include when: {getattr(c, 'inclusive_criteria', '') or ''} "
        f"Exclude when: {getattr(c, 'exclusive_criteria', '') or ''}"
    ).replace('\n', ' ').replace('  ', ' ').strip()


def anchor_specs(config) -> List[Tuple[str, str]]:
    """Return ``[(anchor_id, definition_text), ...]`` for the enabled families.

    Families are gated by ``include_vaamr_nodes`` / ``include_purer_nodes`` /
    ``include_vce_nodes``. Each loader is guarded so a missing framework/codebook
    simply contributes no anchors rather than raising.
    """
    specs: List[Tuple[str, str]] = []

    if getattr(config, 'include_vaamr_nodes', True):
        try:
            from theme_framework.registry import load as _load
            fw = _load('vaamr')
            if fw is not None:
                for t in fw.themes:
                    specs.append((f"anchor:vaamr:{t.theme_id}", _theme_text(t)))
        except Exception:
            pass

    if getattr(config, 'include_purer_nodes', True):
        try:
            from theme_framework.registry import load as _load
            fw = _load('purer')
            if fw is not None:
                for t in fw.themes:
                    specs.append((f"anchor:purer:{t.theme_id}", _theme_text(t)))
        except Exception:
            pass

    if getattr(config, 'include_vce_nodes', False):
        try:
            from codebook.phenomenology_codebook import get_phenomenology_codebook
            for c in get_phenomenology_codebook().codes:
                specs.append((f"anchor:vce:{c.code_id}", _code_text(c)))
        except Exception:
            pass

    return specs


def build_anchor_features(config) -> Dict[str, "object"]:
    """Embed the enabled construct definitions → ``{anchor_id: vector}``.

    Uses the same embedding path as the segments (passage encoding), so anchors
    and segments live in one space. Returns ``{}`` when no families are enabled
    or the embedder is unavailable.
    """
    specs = anchor_specs(config)
    if not specs:
        return {}
    from .. import embeddings as _emb
    ids = [aid for aid, _ in specs]
    texts = [txt for _, txt in specs]
    vecs = _emb.embed_anchor_texts(texts, config)
    return {aid: vecs[i] for i, aid in enumerate(ids)}


def anchor_similarity_edges(anchor_features: Dict[str, "object"],
                            segment_embeddings: Dict[str, "object"],
                            top_m: int) -> List[Tuple[str, str, float]]:
    """Connect each anchor to its ``top_m`` most-similar segments by cosine.

    LABEL-FREE: similarity in embedding space only. Returns ``(anchor_id,
    segment_id, weight)`` triples with weight = max(0, cosine).
    """
    import numpy as np
    if not anchor_features or not segment_embeddings:
        return []
    seg_ids = list(segment_embeddings)
    S = np.stack([np.asarray(segment_embeddings[s], dtype=np.float32) for s in seg_ids], axis=0)
    Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
    m = max(1, min(int(top_m), len(seg_ids)))
    edges: List[Tuple[str, str, float]] = []
    for aid, vec in anchor_features.items():
        a = np.asarray(vec, dtype=np.float32)
        a = a / (np.linalg.norm(a) + 1e-8)
        sims = Sn @ a
        top = np.argsort(-sims)[:m]
        for j in top:
            edges.append((aid, seg_ids[int(j)], float(max(0.0, sims[int(j)]))))
    return edges


def cross_framework_anchor_edges(df_all, config) -> List[Tuple[str, str, float]]:
    """VAAMR-stage <-> VCE-code anchor edges weighted by LLM-consensus lift.

    LLM-DERIVED (uses ``final_label`` co-occurrence) — only built when
    ``include_vce_nodes`` is set, and excluded from the independence substrate.
    """
    if not getattr(config, 'include_vce_nodes', False):
        return []
    from .graph_builder import compute_cross_framework_lift
    lift = compute_cross_framework_lift(df_all, float(getattr(config, 'cross_framework_min_lift', 1.5)))
    edges: List[Tuple[str, str, float]] = []
    for (stage, code), w in (lift or {}).items():
        edges.append((f"anchor:vaamr:{stage}", f"anchor:vce:{code}", float(w)))
    return edges


def build_anchors(df_all, segment_embeddings: Dict[str, "object"], config
                  ) -> Tuple[Dict[str, "object"], List[Tuple[str, str, float]]]:
    """Assemble ``(anchor_features, anchor_edges)`` for :func:`build_graph`.

    ``anchor_edges`` = label-free anchor<->segment similarity edges, plus
    (when ``include_vce_nodes``) the LLM-derived cross-framework anchor<->anchor
    lift edges. Returns ``({}, [])`` when no families are enabled.
    """
    feats = build_anchor_features(config)
    if not feats:
        return {}, []
    top_m = int(getattr(config, 'anchor_knn_m', 8))
    edges = anchor_similarity_edges(feats, segment_embeddings, top_m)
    edges += cross_framework_anchor_edges(df_all, config)
    return feats, edges
