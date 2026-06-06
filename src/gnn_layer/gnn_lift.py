"""
gnn_layer/gnn_lift.py
---------------------
Capability C — GNN as an independent (non-LLM) measurement substrate.

Recomputes (VAAMR stage x VCE code) lift from GNN-derived VAAMR assignments and
compares it side-by-side with the LLM-derived table. Convergence GNN<->LLM is
stronger evidence than LLM<->LLM (methodology Sec 5.2). lift = P(b | a) / P(b).
"""

from typing import Dict, List, Optional

from .graph_builder import _codes


def _lift_table(pairs_stage, pairs_codes, label_a='stage', label_b='code'):
    """Generic lift table. pairs_stage: list of a-values; pairs_codes: list of code-lists."""
    import pandas as pd
    from analysis.stats import lift_ratio
    n = len(pairs_stage)
    if n == 0:
        return pd.DataFrame(columns=[label_a, label_b, 'lift', 'count', 'p_b'])
    code_counts: Dict[str, int] = {}
    for cl in pairs_codes:
        for c in _codes(cl):
            code_counts[c] = code_counts.get(c, 0) + 1
    rows = []
    stages = sorted(set(pairs_stage))
    for a in stages:
        idx = [i for i in range(n) if pairs_stage[i] == a]
        m = len(idx)
        if m == 0:
            continue
        for c, total in code_counts.items():
            in_a = sum(1 for i in idx if c in _codes(pairs_codes[i]))
            if in_a == 0:
                continue
            p_b = total / n
            p_b_given_a = in_a / m
            lift = lift_ratio(p_b_given_a, p_b)
            rows.append({label_a: a, label_b: c, 'lift': round(lift, 3),
                         'count': in_a, 'p_b': round(p_b, 4)})
    return pd.DataFrame(rows)


def gnn_vaamr_vce_lift(df_all, gnn_stage_by_id: Dict[str, int]):
    """(GNN VAAMR stage x VCE code) lift over participant segments."""
    part = df_all[df_all.get('speaker', 'participant') == 'participant'] \
        if 'speaker' in df_all.columns else df_all
    stages, codes = [], []
    for _, r in part.iterrows():
        sid = str(r.get('segment_id'))
        if sid not in gnn_stage_by_id:
            continue
        stages.append(int(gnn_stage_by_id[sid]))
        codes.append(_codes(r.get('codebook_labels_ensemble')))
    return _lift_table(stages, codes, 'vaamr_stage', 'vce_code')


def llm_vaamr_vce_lift(df_all):
    """(LLM VAAMR final_label x VCE code) lift — the comparison baseline."""
    part = df_all[df_all.get('speaker', 'participant') == 'participant'] \
        if 'speaker' in df_all.columns else df_all
    stages, codes = [], []
    for _, r in part.iterrows():
        fl = r.get('final_label')
        try:
            fl = int(fl)
        except (ValueError, TypeError):
            continue
        stages.append(fl)
        codes.append(_codes(r.get('codebook_labels_ensemble')))
    return _lift_table(stages, codes, 'vaamr_stage', 'vce_code')


def compare_gnn_vs_llm(gnn_table, llm_table):
    """Join GNN-derived and LLM-derived (stage,code) lift for convergence reporting."""
    import pandas as pd
    if gnn_table.empty and llm_table.empty:
        return pd.DataFrame(columns=['vaamr_stage', 'vce_code', 'lift_gnn', 'lift_llm', 'both_elevated'])
    g = gnn_table.rename(columns={'lift': 'lift_gnn'})[['vaamr_stage', 'vce_code', 'lift_gnn']] \
        if not gnn_table.empty else pd.DataFrame(columns=['vaamr_stage', 'vce_code', 'lift_gnn'])
    l = llm_table.rename(columns={'lift': 'lift_llm'})[['vaamr_stage', 'vce_code', 'lift_llm']] \
        if not llm_table.empty else pd.DataFrame(columns=['vaamr_stage', 'vce_code', 'lift_llm'])
    merged = pd.merge(g, l, on=['vaamr_stage', 'vce_code'], how='outer')
    merged['lift_gnn'] = merged['lift_gnn'].fillna(0.0)
    merged['lift_llm'] = merged['lift_llm'].fillna(0.0)
    merged['both_elevated'] = (merged['lift_gnn'] >= 1.5) & (merged['lift_llm'] >= 1.5)
    return merged.sort_values(['both_elevated', 'lift_llm'], ascending=False).reset_index(drop=True)
