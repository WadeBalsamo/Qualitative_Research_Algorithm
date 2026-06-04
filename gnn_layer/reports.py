"""
gnn_layer/reports.py
--------------------
CSV / text-report writers for the GNN layer. Artifacts go under 03_analysis_data/gnn/
(machine-readable) and 06_reports/ (human-readable); nothing here touches frozen
segments or master_segments.
"""

import os
from datetime import date
from typing import Dict, List, Optional


def _gnn_dir(output_dir: str) -> str:
    from process import output_paths as _paths
    d = _paths.gnn_data_dir(output_dir)
    os.makedirs(d, exist_ok=True)
    return d


def _reports_dir(output_dir: str) -> str:
    from process import output_paths as _paths
    d = _paths.human_reports_dir(output_dir)
    os.makedirs(d, exist_ok=True)
    return d


def write_segment_positions(positions: dict, output_dir: str) -> str:
    """segment_id, node_type, progression_coord, vaamr_mix_0..4."""
    import numpy as np
    import pandas as pd
    d = _gnn_dir(output_dir)
    rows = {
        'segment_id': positions['segment_id'],
        'node_type': positions['node_type'],
        'progression_coord': np.asarray(positions['progression_coord']).round(4),
    }
    mix = positions.get('vaamr_mixture')
    df = pd.DataFrame(rows)
    if mix is not None:
        for k in range(mix.shape[1]):
            df[f'vaamr_mix_{k}'] = np.asarray(mix[:, k]).round(4)
    path = os.path.join(d, 'segment_positions.csv')
    df.to_csv(path, index=False)
    return path


def write_cue_motifs(motif_stats: dict, purity: dict, exemplars: dict, output_dir: str) -> str:
    import pandas as pd
    d = _gnn_dir(output_dir)
    rows = []
    for m, s in sorted(motif_stats.items()):
        p = purity.get(m, {})
        rows.append({
            'motif_id': m, 'n_blocks': s['n_blocks'], 'influence': s['influence'],
            'mean_pred_forward': s['mean_pred_forward'],
            'dominant_purer': p.get('dominant_purer'), 'purer_purity': p.get('purer_purity'),
            'dominant_microskill': p.get('dominant_microskill'),
            'microskill_purity': p.get('microskill_purity'),
            'n_exemplars': len(exemplars.get(m, [])),
        })
    path = os.path.join(d, 'cue_motifs.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_gnn_vs_llm_lift(comparison, output_dir: str) -> str:
    d = _gnn_dir(output_dir)
    path = os.path.join(d, 'gnn_vs_llm_lift.csv')
    comparison.to_csv(path, index=False)
    return path


def write_purer_microskill_lift(table, output_dir: str) -> str:
    d = _gnn_dir(output_dir)
    path = os.path.join(d, 'purer_microskill_lift.csv')
    table.to_csv(path, index=False)
    return path


def write_coupling_factors(factors: dict, exemplars: dict, interpretation: dict,
                           output_dir: str) -> str:
    import pandas as pd
    d = _gnn_dir(output_dir)
    rows = []
    evr = factors.get('explained_variance_ratio') or []
    corr = factors.get('factor_forward_corr') or []
    n = len(evr)
    for f in range(n):
        interp = interpretation.get(f, {}) if isinstance(interpretation, dict) else {}
        rows.append({
            'factor': f,
            'explained_variance_ratio': round(float(evr[f]), 4) if f < len(evr) else None,
            'forward_corr': round(float(corr[f]), 4) if f < len(corr) else None,
            'nearest_cf_ic': interp.get('nearest_cf_ic'),
            'cf_ic_similarity': interp.get('similarity'),
            'n_exemplars': len(exemplars.get(f, [])),
        })
    path = os.path.join(d, 'coupling_factors.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_emergent_motifs_report(flagged: List[int], motif_stats: dict, purity: dict,
                                 exemplars: dict, output_dir: str) -> str:
    d = _reports_dir(output_dir)
    path = os.path.join(d, 'report_gnn_emergent_motifs.txt')
    W = 72
    lines = ['=' * W, 'GNN EMERGENT THERAPIST-LANGUAGE MOTIFS', '=' * W,
             f'Generated : {date.today().isoformat()}',
             'Motifs that are influential on forward VAAMR transitions but poorly',
             'explained by PURER moves or microcounseling skills — candidate new',
             'therapeutic-language constructs for HUMAN REVIEW (directional evidence).',
             '']
    if not flagged:
        lines.append('No emergent motifs passed the flagging thresholds.')
    for m in flagged:
        s = motif_stats.get(m, {}); p = purity.get(m, {})
        lines.append('-' * W)
        lines.append(f'Motif {m}: influence={s.get("influence")}  n_blocks={s.get("n_blocks")}')
        lines.append(f'  dominant PURER={p.get("dominant_purer")} (purity {p.get("purer_purity")}); '
                     f'dominant microskill={p.get("dominant_microskill")} (purity {p.get("microskill_purity")})')
        for ex in exemplars.get(m, []):
            lines.append(f'    e.g. {ex["session_id"]}: stage {ex["from_stage"]}->{ex["to_stage"]} '
                         f'(from {ex["from_seg_id"]})')
        lines.append('')
    lines.append('=' * W)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    return path


def write_coupling_report(factors: dict, exemplars: dict, interpretation: dict,
                          output_dir: str) -> str:
    d = _reports_dir(output_dir)
    path = os.path.join(d, 'report_gnn_coupling.txt')
    W = 72
    lines = ['=' * W, 'GNN PARTICIPANT<->THERAPIST COUPLING (latent factors)', '=' * W,
             f'Generated : {date.today().isoformat()}',
             'Latent factors of therapist cue language and their correlation with',
             'subsequent participant forward movement. Factors are interpreted against',
             'a Common-Factors/Intervention-Concepts reference lexicon — alliance-like',
             'structure is DISCOVERED, not imposed. Directional evidence; needs review.',
             '']
    corr = factors.get('factor_forward_corr') or []
    evr = factors.get('explained_variance_ratio') or []
    for f in range(len(evr)):
        interp = interpretation.get(f, {}) if isinstance(interpretation, dict) else {}
        lines.append('-' * W)
        lines.append(f'Factor {f}: var={round(float(evr[f]),4)}  '
                     f'forward_corr={round(float(corr[f]),4) if f < len(corr) else "n/a"}')
        if interp.get('nearest_cf_ic'):
            lines.append(f'  nearest CF/IC: {interp["nearest_cf_ic"]} (sim {interp.get("similarity")})')
        for ex in exemplars.get(f, []):
            lines.append(f'    e.g. {ex["session_id"]}: stage {ex["from_stage"]}->{ex["to_stage"]}')
        lines.append('')
    if isinstance(interpretation, dict) and interpretation.get('note'):
        lines.append(interpretation['note'])
    lines.append('=' * W)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    return path
