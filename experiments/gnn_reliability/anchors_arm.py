"""
experiments/gnn_reliability/anchors_arm.py
------------------------------------------
The pre-registered Path-B escalation, arm B1 (design_decisions.md §6): a
CFiCS-style **concept-anchor** GNN arm.

The homogeneous battery found the linear probe ties/beats the plain GNN at n≈205.
B1 asks whether adding VAAMR *construct-definition* anchor nodes to the segment
graph changes that — measured on the load-bearing GNN↔HUMAN axis, where anchors
cannot inflate κ (a segment near its own construct definition tends to predict
that construct, so the GNN↔LLM gain is by-construction; see ``gnn_layer/anchors.py``).

This is the anchored twin of ``harness.run_gnn_arm``: it replicates that engine's
grouped-CV loop EXACTLY (same masking, same ``train.train_model``, same out-of-fold
argmax over ``0..vaamr_n_classes-1``, same participant-grouped fold handling incl.
No-code-by-participant), with the SINGLE departure that the graph is built WITH the
5 VAAMR concept anchors + their label-free anchor↔segment similarity edges. Anchors
carry no VAAMR target, so they are never predicted/scored — the returned contract is
identical to ``run_gnn_arm``: ``{segment_id: pred_int}`` for participant segments.

``harness.py`` is intentionally NOT modified (its loop is inline in ``run_gnn_arm``,
so it is copied here to keep that file merge-clean). Anchor building + kNN are reused
from ``gnn_layer.anchors`` and ``gnn_layer.graph_builder.build_graph`` — nothing is
re-derived here.

Arm-runner contract (matches ``harness.run_gnn_arm`` + ``baselines.run_*``):
  - ``df_all``     : pandas frame, cols incl. ``segment_id`` (str), ``participant_id``,
                     ``speaker``, ``final_label`` (0..4, or NaN = "No code"), ``rater_votes``.
  - ``embeddings`` : ``{segment_id(str): np.ndarray[D]}`` for ALL segments. The anchor
                     definitions are embedded in the SAME space (see ``build_vaamr_anchors``),
                     so ``config`` MUST carry the embedding settings used to build these.
  - ``folds``      : ``{segment_id(str): fold_idx(int)}`` over the LABELED participant
                     segments (participant-grouped CV, built once by the harness).
  - ``config``     : a ``GnnLayerConfig``. Reads ``vaamr_n_classes`` (5 | 6),
                     ``label_mode``, ``anchor_knn_m``, ``knn_k``, ``seed`` + the loss/head
                     flags threaded through ``build_graph`` / ``build_soft_targets`` /
                     ``train_model``.
  - ``framework``  : forwarded to ``build_graph`` for parity with ``run_gnn_arm``
                     (``build_graph`` does not consume it); the VAAMR anchor texts come
                     from ``theme_framework.registry.load('vaamr')`` inside ``gnn_layer.anchors``.
  - RETURNS        : ``{segment_id: predicted_class_int}`` — out-of-fold predictions for
                     every participant segment carrying a VAAMR soft target (a superset of
                     the labeled 205); anchors are never included.
"""

import copy
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# VAAMR concept anchors (reuse gnn_layer.anchors; this is the VAAMR-only arm)
# ---------------------------------------------------------------------------

def _vaamr_anchor_config(config):
    """Shallow copy of ``config`` restricted to the VAAMR anchor family.

    B1 is the VAAMR concept-anchor arm, so only the 5 VAAMR stage definitions
    become anchors: PURER-move and VCE-code families (and the LLM-derived
    cross-framework anchor↔anchor lift edges) are switched off. Mirrors
    ``gnn_layer.train._replace_seed`` (copy.copy + guarded setattr) so the caller's
    config object is never mutated and embedding/loss flags are preserved.
    """
    c = copy.copy(config)
    for attr, val in (('include_vaamr_nodes', True),
                      ('include_purer_nodes', False),
                      ('include_vce_nodes', False)):
        try:
            setattr(c, attr, val)
        except Exception:
            pass
    return c


def build_vaamr_anchors(df_all, embeddings: Dict[str, "object"], config
                        ) -> Tuple[Dict[str, "object"], List[Tuple[str, str, float]]]:
    """``(anchor_features, anchor_edges)`` for the 5 VAAMR construct definitions.

    Reuses ``gnn_layer.anchors.build_anchors`` with a VAAMR-only view of ``config``:
      * the 5 VAAMR stage definitions are loaded from
        ``theme_framework.registry.load('vaamr')`` and embedded via
        ``gnn_layer.embeddings.embed_anchor_texts`` using ``config``'s embedding
        settings — so anchors live in the SAME space as ``embeddings``;
      * each anchor connects to its ``config.anchor_knn_m`` most-similar segments by
        cosine (LABEL-FREE — similarity only, never the LLM's labels);
      * no VCE cross-framework anchor↔anchor edges (that family is off).

    Returns ``({}, [])`` if the framework/embedder is unavailable (the arm then
    degrades to the plain ``run_gnn_arm`` graph).
    """
    from gnn_layer.classifier import anchors as _anchors
    return _anchors.build_anchors(df_all, embeddings, _vaamr_anchor_config(config))


# ---------------------------------------------------------------------------
# GNN measurement engine WITH VAAMR concept anchors (grouped-CV OOF predictions)
# ---------------------------------------------------------------------------

def run_anchored_gnn_arm(df_all, embeddings: Dict[str, "object"],
                         folds: Dict[str, int], config,
                         framework: Optional[dict] = None) -> Dict[str, int]:
    """Out-of-fold VAAMR predictions from the graph built WITH VAAMR concept anchors.

    Identical to ``harness.run_gnn_arm`` except the graph is built with
    ``anchor_features`` + ``anchor_edges`` (the 5 VAAMR construct anchors). Build the
    graph ONCE; for each participant-grouped fold, mask that fold's participant VAAMR
    soft targets, train a fresh model on the rest, and read the held-out argmax.
    Because the folds are participant-grouped, ALL of a held-out participant's segments
    (labeled AND "No code") are out-of-fold together. No-code participants not covered
    by ``folds`` are assigned a fold by participant deterministically.

    Anchors carry no VAAMR soft target (they are not participant segments), so they are
    never in ``vaamr_idx`` and are never predicted, scored, or returned. The result
    contract is identical to ``run_gnn_arm``: ``{segment_id: pred_int}`` (argmax in
    ``0..n_classes-1``) for every participant segment carrying a VAAMR soft target.
    """
    import torch
    import torch.nn.functional as F
    from gnn_layer.classifier import graph_builder as _gb
    from gnn_layer import soft_labels as _sl
    from gnn_layer.classifier import train as _train
    from gnn_layer.runner import _vocabs

    n_classes = int(getattr(config, 'vaamr_n_classes', 5))

    # ---- the ONLY departure from harness.run_gnn_arm: VAAMR concept anchors ----
    anchor_features, anchor_edges = build_vaamr_anchors(df_all, embeddings, config)
    graph = _gb.build_graph(df_all, embeddings, config, framework=framework,
                            anchor_features=anchor_features, anchor_edges=anchor_edges)

    soft = _sl.build_soft_targets(df_all, config.label_mode, n_stages=n_classes)
    vce_codes = _vocabs(config)
    targets = _train.assemble_targets(graph, soft, config, df_all=df_all,
                                      vce_codes=vce_codes or None)

    v_idx = targets.get('vaamr_idx')
    n_v = int(v_idx.numel()) if v_idx is not None else 0
    if n_v == 0:
        return {}
    node_ids = list(graph.node_ids)
    pos_sid = [str(node_ids[int(v_idx[i])]) for i in range(n_v)]

    # segment_id -> participant_id (for grouping the No-code segments by their participant)
    part_of = {str(r.get('segment_id')): str(r.get('participant_id'))
               for _, r in df_all.iterrows()}

    # participant -> fold from the (labeled) fold map; all of a participant's labeled
    # segments share a fold by construction, so first-seen is consistent.
    n_folds = (max(folds.values()) + 1) if folds else 1
    pfold: Dict[str, int] = {}
    for sid, f in folds.items():
        p = part_of.get(str(sid))
        if p is not None:
            pfold.setdefault(p, int(f))
    # Assign participants with NO labeled segment (pure No-code) deterministically.
    extra = sorted({part_of.get(s) for s in pos_sid} - set(pfold) - {None})
    for i, p in enumerate(extra):
        pfold[p] = i % n_folds

    pos_fold = [pfold.get(part_of.get(pos_sid[i])) for i in range(n_v)]

    p_idx = targets.get('purer_idx')
    n_p = int(p_idx.numel()) if p_idx is not None else 0
    keep_p = list(range(n_p))  # PURER is auxiliary here — never masked

    oof: Dict[str, int] = {}
    for f in range(n_folds):
        held = [i for i in range(n_v) if pos_fold[i] == f]
        if not held:
            continue
        keep_v = [i for i in range(n_v) if pos_fold[i] != f]
        sub = _train._subset_targets(targets, keep_v, keep_p)
        fold_cfg = _train._replace_seed(config, int(config.seed) + 1 + f)
        model, _ = _train.train_model(graph, sub, fold_cfg, n_vce=len(vce_codes))
        dev = _train._device(fold_cfg)
        x = graph.x.to(dev)
        ei = graph.edge_index.to(dev)
        ew = graph.edge_weight.to(dev) if graph.edge_weight is not None else None
        eti = (graph.edge_type_ids.to(dev)
               if getattr(graph, 'edge_type_ids', None) is not None else None)
        model.eval()
        with torch.no_grad():
            res = model(x, ei, ew, eti)
        probs = F.softmax(res['soft_vaamr'], dim=1).cpu().numpy()
        for pos in held:
            gi = int(v_idx[pos])
            oof[pos_sid[pos]] = int(probs[gi].argmax())
    return oof
