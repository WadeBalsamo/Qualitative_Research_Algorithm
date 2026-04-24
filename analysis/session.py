"""
analysis/session.py
-------------------
Per-session analysis reports.

Each report covers one session: per-participant stage distributions,
group proportions, stage sequence and transition matrix, confidence
distribution, top codebook codes, and up to 3 exemplars per stage.
"""

import json
import os
from collections import defaultdict

import pandas as pd

from .exemplars import select_prototypical_exemplars, format_exemplar
from .loader import sort_session_ids


def _build_transition_matrix(stage_sequence: list, n_stages: int = 4) -> dict:
    """Build a from→to transition count matrix from an ordered stage sequence.

    Returns nested dict: {str(from_stage): {str(to_stage): int}}.
    """
    matrix = {str(i): {str(j): 0 for j in range(n_stages)} for i in range(n_stages)}
    for a, b in zip(stage_sequence[:-1], stage_sequence[1:]):
        try:
            matrix[str(int(a))][str(int(b))] += 1
        except (KeyError, ValueError):
            pass
    return matrix


def _narrative_for_session(
    session_id: str,
    n_segments: int,
    n_participants: int,
    group_props: dict,
    confidence_dist: dict,
    top_codes: list,
    framework: dict,
) -> str:
    """Compose a plain-English narrative for a session report."""
    if n_segments == 0:
        return f"Session {session_id}: no classified segments."

    # Dominant stage
    dominant = max(group_props, key=lambda k: group_props[k])
    dominant_name = framework.get(int(dominant), {}).get('short_name', f'Stage {dominant}')
    dominant_pct = round(group_props[dominant] * 100, 1)

    # Confidence
    high_pct = round(confidence_dist.get('high', {}).get('proportion', 0) * 100, 1)

    # Top code
    code_note = ''
    if top_codes:
        code_note = f" Most prevalent codebook code: {top_codes[0]['code_id']} ({top_codes[0]['count']} occurrences)."

    return (
        f"Session {session_id} had {n_participants} participant(s) producing "
        f"{n_segments} classified segments. "
        f"{dominant_name} dominated ({dominant_pct}%). "
        f"High-confidence classifications: {high_pct}%.{code_note}"
    )


def generate_session_analysis(
    df: pd.DataFrame,
    session_id: str,
    framework: dict,
    output_dir: str,
) -> dict:
    """Generate an analysis report for a single session.

    Parameters
    ----------
    df : DataFrame
        Full dataset (all sessions). Filtered internally.
    session_id : str
    framework : dict
        From loader.load_framework().
    output_dir : str
        Root pipeline output directory.

    Returns
    -------
    Report dict (also written to reports/analysis/sessions/session_{id}.json).
    """
    sdf = df[df['session_id'] == session_id].copy()
    if sdf.empty:
        return {}

    stage_ids = sorted(framework.keys())
    n_stages = len(stage_ids)

    session_number = int(sdf['session_number'].iloc[0])
    cohort_id = sdf['cohort_id'].dropna().iloc[0] if sdf['cohort_id'].notna().any() else None
    n_segments = len(sdf)
    participant_ids = sorted(sdf['participant_id'].unique().tolist())

    # Per-participant breakdown
    participants_detail = {}
    for pid in participant_ids:
        pdf = sdf[sdf['participant_id'] == pid]
        n_p = len(pdf)
        stage_counts = pdf['final_label'].value_counts().to_dict()
        props = {str(st): round(stage_counts.get(st, 0) / n_p, 4) for st in stage_ids}
        dominant = int(pdf['final_label'].mode().iloc[0]) if n_p > 0 else None
        participants_detail[pid] = {
            'n_segments': n_p,
            'stage_proportions': props,
            'dominant_stage': dominant,
            'dominant_stage_name': framework.get(dominant, {}).get('short_name', '') if dominant is not None else '',
        }

    # Group proportions (mean of per-participant proportions — equal weighting)
    if participants_detail:
        group_props = {}
        for st in stage_ids:
            k = str(st)
            group_props[k] = round(
                sum(v['stage_proportions'][k] for v in participants_detail.values()) / len(participants_detail),
                4,
            )
    else:
        stage_counts = sdf['final_label'].value_counts().to_dict()
        group_props = {str(st): round(stage_counts.get(st, 0) / n_segments, 4) for st in stage_ids}

    # Stage sequence in segment_index order
    sorted_sdf = sdf.sort_values('segment_index')
    stage_sequence = sorted_sdf['final_label'].dropna().astype(int).tolist()
    transition_matrix = _build_transition_matrix(stage_sequence, n_stages=n_stages)

    # Confidence distribution
    tier_counts = sdf['label_confidence_tier'].value_counts().to_dict()
    confidence_dist = {}
    for tier in ('high', 'medium', 'low'):
        cnt = tier_counts.get(tier, 0)
        confidence_dist[tier] = {
            'count': cnt,
            'proportion': round(cnt / n_segments, 4) if n_segments > 0 else 0.0,
        }

    # Top codebook codes
    top_codes = []
    if 'codebook_labels_ensemble' in sdf.columns:
        exploded = sdf.explode('codebook_labels_ensemble')
        exploded = exploded[exploded['codebook_labels_ensemble'].notna() &
                           (exploded['codebook_labels_ensemble'] != '')]
        if not exploded.empty:
            code_counts = exploded['codebook_labels_ensemble'].value_counts()
            for code_id, cnt in code_counts.head(10).items():
                top_codes.append({
                    'code_id': str(code_id),
                    'count': int(cnt),
                    'prevalence': round(int(cnt) / n_segments, 4),
                })

    # Stage exemplars (up to 3 per stage)
    stage_exemplars = {}
    for st in stage_ids:
        st_df = sdf[sdf['final_label'] == st]
        ex_rows = select_prototypical_exemplars(st_df, n=3)
        stage_exemplars[str(st)] = [format_exemplar(ex_rows.iloc[i])
                                     for i in range(len(ex_rows))]

    narrative = _narrative_for_session(
        session_id, n_segments, len(participant_ids),
        group_props, confidence_dist, top_codes, framework,
    )

    report = {
        'session_id': session_id,
        'session_number': session_number,
        'cohort_id': int(cohort_id) if cohort_id is not None and not pd.isna(cohort_id) else None,
        'n_segments': n_segments,
        'n_participants': len(participant_ids),
        'participant_ids': participant_ids,
        'participants': participants_detail,
        'group_stage_proportions': group_props,
        'stage_sequence': stage_sequence,
        'stage_transition_matrix': transition_matrix,
        'confidence_distribution': confidence_dist,
        'top_codebook_codes': top_codes,
        'stage_exemplars': stage_exemplars,
        'narrative_summary': narrative,
    }

    from process import output_paths as _paths
    out_dir = _paths.sessions_json_dir(output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'session_{session_id}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def generate_all_session_analyses(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
) -> list:
    """Generate session analysis reports for all unique sessions in df.

    Returns list of report dicts in longitudinal sort order.
    """
    session_ids = sort_session_ids(df['session_id'].unique().tolist())
    reports = []
    for sid in session_ids:
        try:
            report = generate_session_analysis(df, sid, framework, output_dir)
            if report:
                reports.append(report)
        except Exception as e:
            print(f"  Warning: session analysis failed for {sid}: {e}")
    return reports
