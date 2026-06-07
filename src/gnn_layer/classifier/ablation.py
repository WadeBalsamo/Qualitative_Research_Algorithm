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
}


def run_ablation(graph, targets: dict, config, ablate: str,
                 n_vce: int = 0) -> dict:
    """Retrain with one head removed; return loss deltas vs the full model."""
    from .train import train_model
    obj = _ABLATE_TO_OBJECTIVE.get(ablate)
    full_cfg = deepcopy(config)
    _, full_metrics = train_model(graph, targets, full_cfg, n_vce=n_vce)
    abl_cfg = deepcopy(config)
    if obj and obj in abl_cfg.objectives:
        abl_cfg.objectives = [o for o in abl_cfg.objectives if o != obj]
    _, abl_metrics = train_model(graph, targets, abl_cfg, n_vce=n_vce)
    full_loss = full_metrics.get('best_loss', float('nan'))
    abl_loss = abl_metrics.get('best_loss', float('nan'))
    return {
        'ablate': ablate,
        'objective_removed': obj,
        'best_loss_full': full_loss,
        'best_loss_ablated': abl_loss,
        'delta': (abl_loss - full_loss),
    }


def vce_vaamr_contribution(graph, df_all, config) -> dict:
    """Direct, falsifiable test of the VCE-on-VAAMR hypothesis (methodology §3.3 / §5.2).

    Question: does adding the granular 54-code VCE phenomenology codebook as a
    multi-task layer ON TOP of the coarse five-stage VAAMR arc improve the graph's
    ability to classify VAAMR — i.e., does fine-grained construct supervision sharpen
    the broader mindfulness-skill-development themes, or is it superfluous?

    Method: assemble targets and run the same held-out k-fold cross-validation used by
    the reliability gate TWICE on the identical graph — once with the ``vce_multilabel``
    head active (so the shared GraphSAGE embedding is also shaped to predict VCE codes)
    and once with it removed — then compare out-of-sample VAAMR κ vs the LLM consensus,
    overall and per stage. Everything else (folds, seeds, edges, other heads) is held
    constant, so the κ difference is attributable to the VCE layer alone.

    Interpretation:
      delta_kappa > 0   the VCE layer carries information that improves VAAMR
                        classification — evidence the granular codebook helps.
      delta_kappa ≈ 0   VCE adds nothing the VAAMR signal does not already contain.
      delta_kappa < 0   VCE supervision distracts the shared representation — evidence
                        to keep VCE OUT of the classifier of record.
    Returns a result dict (overall + per-stage κ for both arms, deltas, verdict).
    """
    from copy import deepcopy
    from .train import assemble_targets, crossval_predictions
    from .validation import evaluate_crossval
    from .. import soft_labels as _sl

    try:
        from codebook.phenomenology_codebook import get_phenomenology_codebook
        vce_codes = [c.code_id for c in get_phenomenology_codebook().codes]
    except Exception as e:  # pragma: no cover - defensive
        return {'status': f'skipped: VCE codebook unavailable ({e})'}
    if not vce_codes:
        return {'status': 'skipped: empty VCE codebook'}

    soft = _sl.build_soft_targets(df_all, config.label_mode)

    # --- WITH the VCE layer ---
    cfg_with = deepcopy(config)
    if 'vce_multilabel' not in cfg_with.objectives:
        cfg_with.objectives = list(cfg_with.objectives) + ['vce_multilabel']
    t_with = assemble_targets(graph, soft, cfg_with, df_all=df_all, vce_codes=vce_codes)
    m_with = evaluate_crossval(
        df_all, crossval_predictions(graph, t_with, cfg_with, n_vce=len(vce_codes)), cfg_with)

    # --- WITHOUT the VCE layer (identical graph, identical folds/seed) ---
    cfg_wo = deepcopy(config)
    cfg_wo.objectives = [o for o in cfg_wo.objectives if o != 'vce_multilabel']
    t_wo = assemble_targets(graph, soft, cfg_wo, df_all=df_all, vce_codes=None)
    m_wo = evaluate_crossval(
        df_all, crossval_predictions(graph, t_wo, cfg_wo, n_vce=0), cfg_wo)

    def _ov(m):
        return (m.get('vaamr_overall') or {}).get('cohen_kappa')

    k_with, k_wo = _ov(m_with), _ov(m_wo)
    pc_with = {r['class_id']: r for r in m_with.get('vaamr_per_class', [])}
    pc_wo = {r['class_id']: r for r in m_wo.get('vaamr_per_class', [])}
    per_stage = []
    for cid in sorted(set(pc_with) | set(pc_wo)):
        rw, ro = pc_with.get(cid, {}), pc_wo.get(cid, {})
        kw, ko = rw.get('kappa'), ro.get('kappa')
        per_stage.append({
            'class_id': cid,
            'class_name': rw.get('class_name') or ro.get('class_name') or str(cid),
            'support': rw.get('support') if rw.get('support') is not None else ro.get('support'),
            'kappa_with_vce': kw,
            'kappa_without_vce': ko,
            'delta_kappa': (kw - ko) if (kw is not None and ko is not None) else None,
        })
    delta = (k_with - k_wo) if (k_with is not None and k_wo is not None) else None
    if delta is None:
        verdict = 'inconclusive'
    elif delta >= 0.02:
        verdict = 'vce_helps'
    elif delta <= -0.02:
        verdict = 'vce_harms'
    else:
        verdict = 'vce_neutral'
    return {
        'n_vce_codes': len(vce_codes),
        'vaamr_kappa_with_vce': k_with,
        'vaamr_kappa_without_vce': k_wo,
        'delta_kappa': delta,
        'per_stage': per_stage,
        'verdict': verdict,
    }


def anchor_contribution(df_all, segment_embeddings: Dict[str, "object"], config,
                        framework=None) -> dict:
    """Path-B test (G2): do construct-anchor nodes earn their place?

    Builds the graph TWICE on identical folds/seed — once homogeneous (segments +
    temporal + kNN) and once WITH construct anchors attached by similarity — runs the
    held-out reliability cross-validation on each, and compares κ.

    The DECISIVE axis is GNN<->HUMAN on the blind-coded subset, NOT GNN<->LLM:
    anchors are seeded from the same construct definitions the LLM uses, so they raise
    GNN<->LLM agreement by construction. We therefore score Δκ(graph, human) and only
    REPORT Δκ(graph, LLM) as an explicitly inflated secondary number. When the human
    subset is too small to be decisive, the verdict is 'inconclusive' and the
    recommendation is to keep anchors OFF (the homogeneous default).
    """
    from copy import deepcopy
    from .train import assemble_targets, crossval_predictions
    from .validation import evaluate_crossval
    from .. import soft_labels as _sl
    from . import graph_builder as _gb
    from . import anchors as _anc

    vce_codes = []
    if getattr(config, 'include_vce_nodes', False):
        try:
            from codebook.phenomenology_codebook import get_phenomenology_codebook
            vce_codes = [c.code_id for c in get_phenomenology_codebook().codes]
        except Exception:
            vce_codes = []

    soft = _sl.build_soft_targets(df_all, config.label_mode)

    def _gate(graph):
        t = assemble_targets(graph, soft, config, df_all=df_all, vce_codes=vce_codes or None)
        cv = crossval_predictions(graph, t, config, n_vce=len(vce_codes))
        return evaluate_crossval(df_all, cv, config)

    # --- WITHOUT anchors (homogeneous) ---
    g_wo = _gb.build_graph(df_all, segment_embeddings, config, framework=framework)
    m_wo = _gate(g_wo)

    # --- WITH anchors (similarity-attached construct definitions) ---
    anchor_feats, anchor_edges = _anc.build_anchors(df_all, segment_embeddings, config)
    if not anchor_feats:
        return {'status': 'skipped: no anchor families enabled or embedder unavailable'}
    g_with = _gb.build_graph(df_all, segment_embeddings, config, framework=framework,
                             anchor_features=anchor_feats, anchor_edges=anchor_edges)
    m_with = _gate(g_with)

    def _human_k(m):
        return ((m.get('vaamr_human') or {}).get('graph_vs_human') or {}).get('cohen_kappa')

    def _human_n(m):
        return ((m.get('vaamr_human') or {}).get('graph_vs_human') or {}).get('n') or 0

    def _llm_k(m):
        return (m.get('vaamr_overall') or {}).get('cohen_kappa')

    kh_wo, kh_with = _human_k(m_wo), _human_k(m_with)
    n_h = min(_human_n(m_wo), _human_n(m_with))
    kl_wo, kl_with = _llm_k(m_wo), _llm_k(m_with)

    delta_human = (kh_with - kh_wo) if (kh_with is not None and kh_wo is not None) else None
    delta_llm = (kl_with - kl_wo) if (kl_with is not None and kl_wo is not None) else None

    min_h = int(getattr(config, 'anchor_min_human', 10))
    if delta_human is None or n_h < min_h:
        verdict = 'inconclusive'
        recommend_anchors = False
    elif delta_human >= 0.02:
        verdict = 'anchors_help'
        recommend_anchors = True
    elif delta_human <= -0.02:
        verdict = 'anchors_harm'
        recommend_anchors = False
    else:
        verdict = 'anchors_neutral'
        recommend_anchors = False

    return {
        'n_anchors': len(anchor_feats),
        'n_anchor_edges': len(anchor_edges),
        'human_n': n_h,
        'anchor_min_human': min_h,
        # decisive axis
        'human_kappa_without_anchors': kh_wo,
        'human_kappa_with_anchors': kh_with,
        'delta_kappa_human': delta_human,
        # inflated secondary axis (reported, never decisive)
        'llm_kappa_without_anchors': kl_wo,
        'llm_kappa_with_anchors': kl_with,
        'delta_kappa_llm': delta_llm,
        'verdict': verdict,
        'recommend_anchors': recommend_anchors,
    }


def precipitates_contribution(df_all, segment_embeddings: Dict[str, "object"], config,
                              framework=None) -> dict:
    """Track A1 checkpoint: do typed therapist->participant ``precipitates`` edges earn
    their place in the main graph?

    Builds the graph TWICE on identical folds/seed — once WITHOUT the precipitates family
    (temporal + kNN only) and once WITH it (and the learnable per-edge-type gate active) —
    runs the held-out reliability cross-validation on each, and compares out-of-sample κ.

    Unlike construct anchors, precipitates edges are EMPIRICAL (they connect a therapist
    cue segment to the next participant segment from the transcript's own temporal
    structure), NOT seeded from the construct definitions the LLM consumes. So BOTH axes
    are legitimate here: the LLM axis is distillation fidelity (relevant to LLM-free
    scaling) and the human axis is independent validity. The decision rule keeps the family
    ONLY if it raises κ on both (and never harms the human axis), per the master plan.
    """
    from copy import deepcopy
    from .train import assemble_targets, crossval_predictions
    from .validation import evaluate_crossval
    from .. import soft_labels as _sl
    from . import graph_builder as _gb

    vce_codes = []
    if getattr(config, 'include_vce_nodes', False):
        try:
            from codebook.phenomenology_codebook import get_phenomenology_codebook
            vce_codes = [c.code_id for c in get_phenomenology_codebook().codes]
        except Exception:
            vce_codes = []

    soft = _sl.build_soft_targets(df_all, config.label_mode)

    def _gate(graph, cfg):
        t = assemble_targets(graph, soft, cfg, df_all=df_all, vce_codes=vce_codes or None)
        cv = crossval_predictions(graph, t, cfg, n_vce=len(vce_codes))
        return evaluate_crossval(df_all, cv, cfg)

    # --- WITHOUT precipitates (temporal + kNN only; fixed-weight aggregation) ---
    cfg_wo = deepcopy(config)
    cfg_wo.precipitates_edges = False
    g_wo = _gb.build_graph(df_all, segment_embeddings, cfg_wo, framework=framework)
    m_wo = _gate(g_wo, cfg_wo)

    # --- WITH precipitates (typed edges + learnable per-edge-type gate) ---
    cfg_with = deepcopy(config)
    cfg_with.precipitates_edges = True
    g_with = _gb.build_graph(df_all, segment_embeddings, cfg_with, framework=framework)
    n_prec = int(g_with.meta.get('n_precipitates', 0))
    if n_prec == 0:
        return {'status': 'skipped: no precipitates edges built (no cue blocks in corpus)'}
    m_with = _gate(g_with, cfg_with)

    def _human_k(m):
        return ((m.get('vaamr_human') or {}).get('graph_vs_human') or {}).get('cohen_kappa')

    def _human_n(m):
        return ((m.get('vaamr_human') or {}).get('graph_vs_human') or {}).get('n') or 0

    def _llm_k(m):
        return (m.get('vaamr_overall') or {}).get('cohen_kappa')

    kh_wo, kh_with = _human_k(m_wo), _human_k(m_with)
    n_h = min(_human_n(m_wo), _human_n(m_with))
    kl_wo, kl_with = _llm_k(m_wo), _llm_k(m_with)

    delta_human = (kh_with - kh_wo) if (kh_with is not None and kh_wo is not None) else None
    delta_llm = (kl_with - kl_wo) if (kl_with is not None and kl_wo is not None) else None

    min_h = int(getattr(config, 'anchor_min_human', 10))
    thresh = 0.02
    # The family must raise the LLM (scaling) axis AND not harm — ideally raise — the
    # independent human axis. Without a decisive human subset we cannot confirm validity,
    # so we stay conservative and leave the family OFF (the homogeneous default).
    if delta_llm is None:
        verdict, recommend = 'inconclusive', False
    elif delta_human is None or n_h < min_h:
        verdict, recommend = 'inconclusive', False
    elif delta_llm >= thresh and delta_human >= thresh:
        verdict, recommend = 'precipitates_help', True
    elif delta_llm <= -thresh or delta_human <= -thresh:
        verdict, recommend = 'precipitates_harm', False
    else:
        verdict, recommend = 'precipitates_neutral', False

    return {
        'n_precipitates_edges': n_prec,
        'human_n': n_h,
        'min_human': min_h,
        'human_kappa_without': kh_wo,
        'human_kappa_with': kh_with,
        'delta_kappa_human': delta_human,
        'llm_kappa_without': kl_wo,
        'llm_kappa_with': kl_with,
        'delta_kappa_llm': delta_llm,
        'verdict': verdict,
        'recommend_precipitates': recommend,
    }


def write_precipitates_contribution_report(result: dict, output_dir: str) -> str:
    """Human-readable Track A1 precipitates-contribution report → 06_reports/06_gnn/."""
    import os
    from process import output_paths as _paths

    def _k(x):
        return 'n/a' if not isinstance(x, (int, float)) else f"{x:+.3f}"

    W = 78
    L = ["=" * W, "PRECIPITATES-EDGE CONTRIBUTION (Track A1)", "=" * W, ""]
    if result.get('status'):
        L.append(f"  {result['status']}")
        L.append("")
    else:
        L.append("Do typed therapist->participant 'precipitates' edges (each cue segment ->")
        L.append("the following participant segment) + a learnable per-edge-type gate improve")
        L.append("the graph? These edges are EMPIRICAL (transcript temporal structure), not")
        L.append("seeded from construct definitions, so BOTH axes below are legitimate. The")
        L.append("family is kept ONLY if it raises κ on both axes without harming the human one.")
        L.append("")
        L.append(f"  precipitates_edges={result.get('n_precipitates_edges')}"
                 f"  human_subset_n={result.get('human_n')} (min for decision: {result.get('min_human')})")
        L.append("")
        L.append("-" * W)
        L.append("  GNN vs HUMAN (blind-coded subset) — independent validity")
        L.append("-" * W)
        L.append(f"    κ without precipitates : {_k(result.get('human_kappa_without'))}")
        L.append(f"    κ with precipitates    : {_k(result.get('human_kappa_with'))}")
        L.append(f"    Δκ (human)             : {_k(result.get('delta_kappa_human'))}")
        L.append("")
        L.append("-" * W)
        L.append("  GNN vs LLM (out-of-sample) — distillation fidelity for scaling")
        L.append("-" * W)
        L.append(f"    κ without precipitates : {_k(result.get('llm_kappa_without'))}")
        L.append(f"    κ with precipitates    : {_k(result.get('llm_kappa_with'))}")
        L.append(f"    Δκ (LLM)               : {_k(result.get('delta_kappa_llm'))}")
        L.append("")
        L.append("=" * W)
        L.append(f"  VERDICT: {result.get('verdict')}    "
                 f"RECOMMEND precipitates_edges ON: {'YES' if result.get('recommend_precipitates') else 'NO'}")
        L.append("=" * W)
        if result.get('verdict') == 'inconclusive':
            L.append("  Not decisive (missing κ or human subset too small) — precipitates stay OFF")
            L.append("  (the homogeneous default). Re-run once more blind-coded data exists.")
    rep_dir = _paths.reports_gnn_dir(output_dir)
    os.makedirs(rep_dir, exist_ok=True)
    path = os.path.join(rep_dir, 'precipitates_contribution.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path


def write_anchor_contribution_report(result: dict, output_dir: str) -> str:
    """Human-readable Path-B anchor-contribution report → 06_reports/06_gnn/."""
    import os
    from process import output_paths as _paths

    def _k(x):
        return 'n/a' if not isinstance(x, (int, float)) else f"{x:+.3f}"

    W = 78
    L = ["=" * W, "CONSTRUCT-ANCHOR CONTRIBUTION (Path B / G2)", "=" * W, ""]
    if result.get('status'):
        L.append(f"  {result['status']}")
        L.append("")
    else:
        L.append("Does attaching construct-DEFINITION anchor nodes (by similarity, no labels)")
        L.append("to the graph improve it? The decisive axis is GNN-vs-HUMAN out-of-sample κ,")
        L.append("NOT GNN-vs-LLM: anchors are seeded from the same definitions the LLM uses, so")
        L.append("they inflate GNN-vs-LLM agreement by construction. The LLM axis is shown only")
        L.append("for transparency and is NEVER used to justify anchors.")
        L.append("")
        L.append(f"  anchors={result.get('n_anchors')}  anchor_edges={result.get('n_anchor_edges')}"
                 f"  human_subset_n={result.get('human_n')} (min for decision: {result.get('anchor_min_human')})")
        L.append("")
        L.append("-" * W)
        L.append("  DECISIVE — GNN vs HUMAN (blind-coded subset)")
        L.append("-" * W)
        L.append(f"    κ without anchors : {_k(result.get('human_kappa_without_anchors'))}")
        L.append(f"    κ with anchors    : {_k(result.get('human_kappa_with_anchors'))}")
        L.append(f"    Δκ (human)        : {_k(result.get('delta_kappa_human'))}")
        L.append("")
        L.append("-" * W)
        L.append("  SECONDARY — GNN vs LLM (INFLATED by construction; not decisive)")
        L.append("-" * W)
        L.append(f"    κ without anchors : {_k(result.get('llm_kappa_without_anchors'))}")
        L.append(f"    κ with anchors    : {_k(result.get('llm_kappa_with_anchors'))}")
        L.append(f"    Δκ (LLM)          : {_k(result.get('delta_kappa_llm'))}")
        L.append("")
        L.append("=" * W)
        L.append(f"  VERDICT: {result.get('verdict')}    "
                 f"RECOMMEND ANCHORS ON: {'YES' if result.get('recommend_anchors') else 'NO'}")
        L.append("=" * W)
        if result.get('verdict') == 'inconclusive':
            L.append("  Human subset too small to decide on the independent axis — anchors stay OFF")
            L.append("  (the homogeneous graph is the default). Re-run once more blind-coded data exists.")
    rep_dir = _paths.reports_gnn_dir(output_dir)
    os.makedirs(rep_dir, exist_ok=True)
    path = os.path.join(rep_dir, 'anchor_contribution.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path


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
