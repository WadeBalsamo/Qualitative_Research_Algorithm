"""
gnn_layer/triangulation.py
--------------------------
GNN ↔ LLM ↔ human construct triangulation.

The GNN is justified as an *independent measurement substrate*: it learns segment
representations from embeddings + graph structure, then predicts VAAMR stage and
PURER move from its own heads. Comparing those predictions against the LLM labels
(and the human-coded subset where available) tests construct validity —
convergence across independent substrates is stronger evidence than LLM↔LLM
agreement alone. Agreement is reported as Cohen's κ (chance-corrected) + raw
percent agreement.

Directional: the GNN was trained on LLM/ballot labels, so agreement is partly
expected; divergence flags genuinely ambiguous segments worth human review.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from process import output_paths as _paths


def _kappa(a: List[int], b: List[int]) -> Optional[float]:
    """Cohen's κ via sklearn; None if degenerate."""
    if len(a) < 2:
        return None
    try:
        from sklearn.metrics import cohen_kappa_score
        if len(set(a)) < 2 and len(set(b)) < 2 and set(a) == set(b):
            return 1.0
        return float(cohen_kappa_score(a, b))
    except Exception:
        return None


def _agreement(pred: pd.Series, ref: pd.Series) -> Dict[str, float]:
    """Percent agreement + κ between two aligned integer label series (NaNs dropped)."""
    d = pd.DataFrame({'p': pred, 'r': ref}).dropna()
    if len(d) == 0:
        return {'n': 0, 'percent_agreement': None, 'cohen_kappa': None}
    p = d['p'].astype(int).tolist()
    r = d['r'].astype(int).tolist()
    pa = float(np.mean([1 if pi == ri else 0 for pi, ri in zip(p, r)]))
    return {'n': int(len(d)), 'percent_agreement': round(pa, 4), 'cohen_kappa': _kappa(p, r)}


def compute_triangulation(head_preds: dict, df_all: pd.DataFrame) -> dict:
    """Compare GNN head predictions to LLM (and human) labels. Returns a nested summary."""
    preds = pd.DataFrame({k: v for k, v in head_preds.items() if isinstance(v, list)})
    if preds.empty or 'segment_id' not in preds.columns:
        return {}
    preds['segment_id'] = preds['segment_id'].astype(str)
    base = df_all.copy()
    base['segment_id'] = base['segment_id'].astype(str)
    merged = base.merge(preds, on='segment_id', how='inner', suffixes=('', '_gnn'))

    result: Dict[str, dict] = {}

    # VAAMR (participant segments): GNN vs LLM final_label, and vs human_label.
    if 'gnn_vaamr_pred' in merged.columns:
        part = merged[merged.get('node_type') == 'participant_segment'] \
            if 'node_type' in merged.columns else merged
        if 'final_label' in part.columns:
            result['vaamr_gnn_vs_llm'] = _agreement(part['gnn_vaamr_pred'], part['final_label'])
        if 'human_label' in part.columns and 'in_human_coded_subset' in part.columns:
            hsub = part[part['in_human_coded_subset'] == True]  # noqa: E712
            if len(hsub):
                result['vaamr_gnn_vs_human'] = _agreement(hsub['gnn_vaamr_pred'], hsub['human_label'])
                result['vaamr_llm_vs_human'] = _agreement(hsub['final_label'], hsub['human_label'])

    # PURER (therapist segments): GNN vs LLM purer_primary.
    if 'gnn_purer_pred' in merged.columns and 'purer_primary' in merged.columns:
        ther = merged[merged.get('node_type') == 'therapist_segment'] \
            if 'node_type' in merged.columns else merged
        result['purer_gnn_vs_llm'] = _agreement(ther['gnn_purer_pred'], ther['purer_primary'])

    return result


def write_triangulation_report(triangulation: dict, output_dir: str,
                               lift_summary: Optional[dict] = None) -> str:
    """Human-readable GNN↔LLM↔human triangulation report → 06_reports/."""
    L = []
    L.append("=" * 78)
    L.append("GNN ↔ LLM ↔ HUMAN CONSTRUCT TRIANGULATION")
    L.append("=" * 78)
    L.append("")
    L.append("The GNN predicts VAAMR stage and PURER move from its OWN heads (embeddings +")
    L.append("graph structure). Agreement with the LLM labels — and with the human-coded")
    L.append("subset — tests construct validity across independent substrates. κ is")
    L.append("chance-corrected. The GNN was trained on LLM/ballot labels, so some agreement")
    L.append("is expected; the value is in the human comparison and in flagged divergence.")
    L.append("")

    def _line(label, d):
        if not d or d.get('n', 0) == 0:
            return f"  {label:<26} (no overlap)"
        k = d.get('cohen_kappa')
        kk = f"κ={k:+.3f}" if isinstance(k, (int, float)) and k == k else "κ=n/a"
        return (f"  {label:<26} n={d['n']:<5} agreement={d['percent_agreement']*100:5.1f}%  {kk}")

    L.append("-" * 78)
    L.append("VAAMR stage (participant segments)")
    L.append("-" * 78)
    L.append(_line("GNN vs LLM", triangulation.get('vaamr_gnn_vs_llm')))
    if 'vaamr_gnn_vs_human' in triangulation:
        L.append(_line("GNN vs HUMAN", triangulation.get('vaamr_gnn_vs_human')))
        L.append(_line("LLM vs HUMAN", triangulation.get('vaamr_llm_vs_human')))
        L.append("  (GNN vs HUMAN approaching LLM vs HUMAN ⇒ the GNN is a comparably valid")
        L.append("   independent measure, not merely echoing the LLM.)")
    else:
        L.append("  (no human-coded subset available for the stronger human comparison.)")
    L.append("")
    L.append("-" * 78)
    L.append("PURER move (therapist segments)")
    L.append("-" * 78)
    L.append(_line("GNN vs LLM", triangulation.get('purer_gnn_vs_llm')))
    L.append("")

    if lift_summary:
        L.append("-" * 78)
        L.append("VAAMR × VCE lift convergence (GNN vs LLM)")
        L.append("-" * 78)
        L.append(f"  (stage, code) pairs compared: {lift_summary.get('n_pairs', 0)}")
        if lift_summary.get('n_both_elevated') is not None:
            L.append(f"  Pairs elevated under BOTH substrates: {lift_summary['n_both_elevated']}")
        L.append("")

    rep_dir = _paths.human_reports_dir(output_dir)
    os.makedirs(rep_dir, exist_ok=True)
    path = os.path.join(rep_dir, 'report_gnn_triangulation.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path
