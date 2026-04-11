"""
analysis/stage_progression.py
-----------------------------
Per-participant session stage progression and state transition analysis.

Replaces the per-session-only logic previously in process/dataset_assembly.py
with proper (participant_id, session_id) granularity.
"""

import os

import numpy as np
import pandas as pd

from .loader import sort_session_ids


def compute_session_stage_progression(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
) -> pd.DataFrame:
    """Track theme/stage progression per (participant, session).

    Groups by (participant_id, session_id) so each participant gets their own
    row even when multiple participants share a session.

    Parameters
    ----------
    df : DataFrame
        Cleaned analysis DataFrame from loader.load_segments().
    framework : dict
        Maps int stage_id -> {short_name, ...}.
    output_dir : str
        Root pipeline output directory.

    Returns
    -------
    DataFrame with one row per (participant, session) containing:
        participant_id, session_id, session_number, cohort_id,
        n_segments, stage_sequence, stage_transitions,
        forward_transitions, backward_transitions, lateral_transitions,
        max_stage_reached, dominant_stage, dominant_stage_name,
        stage_distribution
    """
    rows = []

    for (pid, sid), group in df.groupby(['participant_id', 'session_id']):
        group = group.sort_values('segment_index')
        stages = group['final_label'].astype(int).tolist()

        if len(stages) == 0:
            continue

        transitions = []
        forward = backward = 0
        for i in range(1, len(stages)):
            fr, to = stages[i - 1], stages[i]
            transitions.append((fr, to))
            if to > fr:
                forward += 1
            elif to < fr:
                backward += 1

        stage_counts = {}
        for s in stages:
            stage_counts[s] = stage_counts.get(s, 0) + 1

        dominant = max(stage_counts, key=stage_counts.get)
        snum = int(group['session_number'].iloc[0])
        cid = group['cohort_id'].dropna().iloc[0] if group['cohort_id'].notna().any() else None

        rows.append({
            'participant_id': pid,
            'session_id': sid,
            'session_number': snum,
            'cohort_id': int(cid) if cid is not None and not pd.isna(cid) else None,
            'n_segments': len(stages),
            'stage_sequence': stages,
            'stage_transitions': transitions,
            'forward_transitions': forward,
            'backward_transitions': backward,
            'lateral_transitions': len(transitions) - forward - backward,
            'max_stage_reached': max(stages),
            'dominant_stage': dominant,
            'dominant_stage_name': framework.get(dominant, {}).get('short_name', str(dominant)),
            'stage_distribution': {
                framework.get(k, {}).get('short_name', str(k)): v
                for k, v in sorted(stage_counts.items())
            },
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ['cohort_id', 'session_number', 'participant_id'],
            na_position='last',
        ).reset_index(drop=True)

    out_dir = os.path.join(output_dir, 'reports', 'longitudinal')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'session_stage_progression.csv')
    result.to_csv(path, index=False)

    return result


def compute_cross_session_transitions(
    df: pd.DataFrame,
    framework: dict,
) -> tuple:
    """Compute between-session dominant-theme transitions per participant.

    For each participant, builds a sequence of (session_id, dominant_stage)
    ordered by session_number, then records transitions from session N to N+1.

    Parameters
    ----------
    df : DataFrame
        Cleaned analysis DataFrame.
    framework : dict
        Maps int stage_id -> {short_name, ...}.

    Returns
    -------
    (cross_session_matrix, participant_sequences) where:
        cross_session_matrix : DataFrame (n_stages × n_stages) of transition counts
        participant_sequences : dict {participant_id: [(session_id, stage_id, stage_name), ...]}
    """
    stage_ids = sorted(framework.keys())
    n = len(stage_ids)
    matrix = np.zeros((n, n), dtype=int)
    id_to_idx = {sid: i for i, sid in enumerate(stage_ids)}

    participant_sequences = {}

    for pid in sorted(df['participant_id'].unique()):
        pdf = df[df['participant_id'] == pid]
        sessions = sort_session_ids(pdf['session_id'].unique().tolist())

        seq = []
        for sid in sessions:
            sdf = pdf[pdf['session_id'] == sid]
            if sdf.empty:
                continue
            dominant = int(sdf['final_label'].mode().iloc[0])
            name = framework.get(dominant, {}).get('short_name', f'Stage {dominant}')
            seq.append((sid, dominant, name))

        participant_sequences[pid] = seq

        for a, b in zip(seq[:-1], seq[1:]):
            fa, fb = a[1], b[1]
            if fa in id_to_idx and fb in id_to_idx:
                matrix[id_to_idx[fa]][id_to_idx[fb]] += 1

    labels = [framework[sid].get('short_name', f'Stage {sid}') for sid in stage_ids]
    cross_session_matrix = pd.DataFrame(matrix, index=labels, columns=labels)
    return cross_session_matrix, participant_sequences


def compute_state_transition_matrix(
    df: pd.DataFrame,
    framework: dict,
) -> pd.DataFrame:
    """Aggregate from->to stage transition frequencies across all data.

    Parameters
    ----------
    df : DataFrame
        Cleaned analysis DataFrame.
    framework : dict
        Maps int stage_id -> {short_name, ...}.

    Returns
    -------
    DataFrame of shape (n_stages, n_stages) with stage short_names as
    both index and columns. Values are transition counts.
    """
    stage_ids = sorted(framework.keys())
    n = len(stage_ids)
    matrix = np.zeros((n, n), dtype=int)
    id_to_idx = {sid: i for i, sid in enumerate(stage_ids)}

    for (pid, sid), group in df.groupby(['participant_id', 'session_id']):
        stages = group.sort_values('segment_index')['final_label'].astype(int).tolist()
        for a, b in zip(stages[:-1], stages[1:]):
            if a in id_to_idx and b in id_to_idx:
                matrix[id_to_idx[a]][id_to_idx[b]] += 1

    labels = [framework[sid].get('short_name', f'Stage {sid}') for sid in stage_ids]
    return pd.DataFrame(matrix, index=labels, columns=labels)
