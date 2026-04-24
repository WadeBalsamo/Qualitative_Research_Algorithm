"""
analysis/figure_data.py
-----------------------
Export graph-ready CSV datasets for visualization in R/Python.

Graph-ready CSVs land in {output_dir}/04_analysis_data/graphing/.
"""

import os

import pandas as pd

from .loader import sort_session_ids
from process import output_paths as _paths


def _ensure_graphing_dir(output_dir: str) -> str:
    out = _paths.graphing_dir(output_dir)
    os.makedirs(out, exist_ok=True)
    return out


def export_theme_proportions_by_participant_session(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
) -> pd.DataFrame:
    """Export theme proportions per (participant, session) — one row each.

    Columns: participant_id, session_id, session_number, cohort_id,
             stage_0_pct, stage_1_pct, ..., n_segments, mean_confidence
    """
    stage_ids = sorted(framework.keys())
    rows = []

    for (pid, sid), group in df.groupby(['participant_id', 'session_id']):
        n = len(group)
        snum = int(group['session_number'].iloc[0])
        cid = group['cohort_id'].dropna().iloc[0] if group['cohort_id'].notna().any() else None

        row = {
            'participant_id': pid,
            'session_id': sid,
            'session_number': snum,
            'cohort_id': int(cid) if cid is not None and not pd.isna(cid) else None,
            'n_segments': n,
            'mean_confidence': round(float(group['llm_confidence_primary'].mean()), 4)
                if group['llm_confidence_primary'].notna().any() else None,
        }
        for st in stage_ids:
            cnt = int((group['final_label'] == st).sum())
            row[f'stage_{st}_pct'] = round(cnt / n, 4) if n > 0 else 0.0

        rows.append(row)

    result = pd.DataFrame(rows)

    # Sort by cohort, session_number, participant
    if not result.empty:
        result = result.sort_values(['cohort_id', 'session_number', 'participant_id'],
                                    na_position='last').reset_index(drop=True)

    out = _ensure_graphing_dir(output_dir)
    path = os.path.join(out, 'theme_proportions_by_participant_session.csv')
    result.to_csv(path, index=False)
    return result


def export_group_theme_trajectories(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
) -> pd.DataFrame:
    """Export group-level mean theme proportions per session.

    Averages are computed over per-participant proportions (equal weighting),
    not raw segment counts — avoids bias from variable participant segment counts.

    Columns: session_id, session_number, cohort_id,
             stage_0_mean, stage_1_mean, ..., n_participants
    """
    stage_ids = sorted(framework.keys())

    # First compute per-participant proportions
    part_rows = []
    for (pid, sid), group in df.groupby(['participant_id', 'session_id']):
        n = len(group)
        snum = int(group['session_number'].iloc[0])
        cid = group['cohort_id'].dropna().iloc[0] if group['cohort_id'].notna().any() else None
        row = {'participant_id': pid, 'session_id': sid, 'session_number': snum, 'cohort_id': cid}
        for st in stage_ids:
            cnt = int((group['final_label'] == st).sum())
            row[f'stage_{st}_pct'] = cnt / n if n > 0 else 0.0
        part_rows.append(row)

    part_df = pd.DataFrame(part_rows)
    if part_df.empty:
        return part_df

    # Average over participants per session
    stage_cols = [f'stage_{st}_pct' for st in stage_ids]
    agg = part_df.groupby(['session_id', 'session_number', 'cohort_id']).agg(
        n_participants=('participant_id', 'count'),
        **{f'stage_{st}_mean': (f'stage_{st}_pct', 'mean') for st in stage_ids},
    ).reset_index()

    # Sort longitudinally
    session_order = sort_session_ids(agg['session_id'].tolist())
    order_map = {sid: i for i, sid in enumerate(session_order)}
    agg['_order'] = agg['session_id'].map(order_map)
    agg = agg.sort_values('_order').drop(columns='_order').reset_index(drop=True)

    # Round stage mean columns
    for st in stage_ids:
        agg[f'stage_{st}_mean'] = agg[f'stage_{st}_mean'].round(4)

    out = _ensure_graphing_dir(output_dir)
    path = os.path.join(out, 'group_theme_trajectories.csv')
    agg.to_csv(path, index=False)
    return agg


def export_codebook_prevalence_by_participant_session(
    df: pd.DataFrame,
    output_dir: str,
) -> pd.DataFrame:
    """Export codebook code prevalence per (participant, session, code).

    Columns: participant_id, session_id, session_number, cohort_id,
             code_id, count, prevalence

    Silently returns empty DataFrame if no codebook data present.
    """
    if 'codebook_labels_ensemble' not in df.columns:
        return pd.DataFrame()

    rows = []
    for (pid, sid), group in df.groupby(['participant_id', 'session_id']):
        n = len(group)
        snum = int(group['session_number'].iloc[0])
        cid = group['cohort_id'].dropna().iloc[0] if group['cohort_id'].notna().any() else None

        exploded = group.explode('codebook_labels_ensemble')
        exploded = exploded[
            exploded['codebook_labels_ensemble'].notna() &
            (exploded['codebook_labels_ensemble'] != '')
        ]
        if exploded.empty:
            continue

        code_counts = exploded['codebook_labels_ensemble'].value_counts()
        for code_id, cnt in code_counts.items():
            rows.append({
                'participant_id': pid,
                'session_id': sid,
                'session_number': snum,
                'cohort_id': int(cid) if cid is not None and not pd.isna(cid) else None,
                'code_id': code_id,
                'count': int(cnt),
                'prevalence': round(int(cnt) / n, 4),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(['cohort_id', 'session_number', 'participant_id', 'code_id'],
                                    na_position='last').reset_index(drop=True)

    out = _ensure_graphing_dir(output_dir)
    path = os.path.join(out, 'codebook_prevalence_by_participant_session.csv')
    result.to_csv(path, index=False)
    return result


def export_confidence_distribution_by_stage(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
) -> pd.DataFrame:
    """Export confidence tier distribution per VA-MR stage.

    Columns: stage_id, stage_name, label_confidence_tier, count, proportion
    (proportion = count / total segments in that stage)
    """
    rows = []
    for stage_id in sorted(framework.keys()):
        stage_df = df[df['final_label'] == stage_id]
        n_stage = len(stage_df)
        stage_name = framework.get(stage_id, {}).get('short_name', f'Stage {stage_id}')

        for tier in ('high', 'medium', 'low'):
            cnt = int((stage_df['label_confidence_tier'] == tier).sum())
            rows.append({
                'stage_id': stage_id,
                'stage_name': stage_name,
                'label_confidence_tier': tier,
                'count': cnt,
                'proportion': round(cnt / n_stage, 4) if n_stage > 0 else 0.0,
            })

    result = pd.DataFrame(rows)

    out = _ensure_graphing_dir(output_dir)
    path = os.path.join(out, 'confidence_distribution_by_stage.csv')
    result.to_csv(path, index=False)
    return result


def export_participant_stage_dominant(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
) -> pd.DataFrame:
    """Export dominant stage per (participant, session) — one row each.

    Columns: participant_id, session_id, session_number, cohort_id,
             dominant_stage, dominant_stage_name
    """
    rows = []
    for (pid, sid), group in df.groupby(['participant_id', 'session_id']):
        snum = int(group['session_number'].iloc[0])
        cid = group['cohort_id'].dropna().iloc[0] if group['cohort_id'].notna().any() else None
        n = len(group)

        dominant = int(group['final_label'].mode().iloc[0]) if n > 0 else None
        dominant_name = framework.get(dominant, {}).get('short_name', '') if dominant is not None else ''

        rows.append({
            'participant_id': pid,
            'session_id': sid,
            'session_number': snum,
            'cohort_id': int(cid) if cid is not None and not pd.isna(cid) else None,
            'dominant_stage': dominant,
            'dominant_stage_name': dominant_name,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(['cohort_id', 'session_number', 'participant_id'],
                                    na_position='last').reset_index(drop=True)

    out = _ensure_graphing_dir(output_dir)
    path = os.path.join(out, 'participant_stage_dominant.csv')
    result.to_csv(path, index=False)
    return result


def export_combined_cohort_group_trajectories(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
) -> pd.DataFrame:
    """Export group-level mean theme proportions per session_number, combining all cohorts.

    Unlike export_group_theme_trajectories which keeps cohorts separate,
    this groups by session_number only — treating the same session number
    across cohorts as the same longitudinal timepoint.

    Columns: session_number, stage_0_mean, stage_1_mean, ..., n_participants
    """
    stage_ids = sorted(framework.keys())

    part_rows = []
    for (pid, snum), group in df.groupby(['participant_id', 'session_number']):
        n = len(group)
        row = {'participant_id': pid, 'session_number': int(snum)}
        for st in stage_ids:
            row[f'stage_{st}_pct'] = int((group['final_label'] == st).sum()) / n if n > 0 else 0.0
        part_rows.append(row)

    part_df = pd.DataFrame(part_rows)
    if part_df.empty:
        return part_df

    agg = part_df.groupby('session_number').agg(
        n_participants=('participant_id', 'count'),
        **{f'stage_{st}_mean': (f'stage_{st}_pct', 'mean') for st in stage_ids},
    ).reset_index()

    agg = agg.sort_values('session_number').reset_index(drop=True)

    for st in stage_ids:
        agg[f'stage_{st}_mean'] = agg[f'stage_{st}_mean'].round(4)

    out = _ensure_graphing_dir(output_dir)
    path = os.path.join(out, 'combined_cohort_group_trajectories.csv')
    agg.to_csv(path, index=False)
    return agg


def export_all_graphing_datasets(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
) -> list:
    """Orchestrate all graph-ready CSV exports.

    Returns list of file paths written.
    """
    out = _ensure_graphing_dir(output_dir)
    paths = []

    try:
        export_theme_proportions_by_participant_session(df, framework, output_dir)
        paths.append(os.path.join(out, 'theme_proportions_by_participant_session.csv'))
    except Exception as e:
        print(f"  Warning: theme_proportions CSV failed: {e}")

    try:
        export_group_theme_trajectories(df, framework, output_dir)
        paths.append(os.path.join(out, 'group_theme_trajectories.csv'))
    except Exception as e:
        print(f"  Warning: group_theme_trajectories CSV failed: {e}")

    try:
        export_combined_cohort_group_trajectories(df, framework, output_dir)
        paths.append(os.path.join(out, 'combined_cohort_group_trajectories.csv'))
    except Exception as e:
        print(f"  Warning: combined_cohort_group_trajectories CSV failed: {e}")

    try:
        export_codebook_prevalence_by_participant_session(df, output_dir)
        paths.append(os.path.join(out, 'codebook_prevalence_by_participant_session.csv'))
    except Exception as e:
        print(f"  Warning: codebook_prevalence CSV failed: {e}")

    try:
        export_confidence_distribution_by_stage(df, framework, output_dir)
        paths.append(os.path.join(out, 'confidence_distribution_by_stage.csv'))
    except Exception as e:
        print(f"  Warning: confidence_distribution CSV failed: {e}")

    try:
        export_participant_stage_dominant(df, framework, output_dir)
        paths.append(os.path.join(out, 'participant_stage_dominant.csv'))
    except Exception as e:
        print(f"  Warning: participant_stage_dominant CSV failed: {e}")

    return paths
