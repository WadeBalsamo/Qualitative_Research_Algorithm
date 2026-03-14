"""
dataset_assembly.py
-------------------
Stage 6: Dataset Assembly -- produces the five specified output files.

Output 1: Master Segment Dataset (JSONL + CSV)
Output 2: Session Adjacency Index (JSONL)
Output 3: Human Validation Report (JSON)
Output 4: Stage Definition File (JSON)
Output 5: Content Validity Test Set (JSONL)

Adapted from:
  - classify_ctl_results.py and classify_reddit_results.py which aggregate
    metrics across constructs and feature vectors into summary tables.
  - save_classification_performance() from the metrics_report module.
"""

import json
import os
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

from .data_structures import Segment
from .vamr_constructs import stage_definitions, get_stage_definitions_json
from .validation import compute_validation_metrics


# ---------------------------------------------------------------------------
# Output 1: Master Segment Dataset
# ---------------------------------------------------------------------------

def assemble_master_dataset(
    segments: List[Segment],
    output_path: str,
) -> pd.DataFrame:
    """
    Produce the master segment dataset.

    Computes final_label and label_confidence_tier following the
    priority logic specified in the task description.
    """
    rows = []
    for seg in segments:
        # Compute final_label
        if seg.adjudicated_label is not None:
            final_label = seg.adjudicated_label
            final_label_source = 'adjudicated'
        elif seg.human_label is not None and seg.human_label == seg.primary_stage:
            final_label = seg.human_label
            final_label_source = 'human_consensus'
        elif seg.primary_stage is not None:
            final_label = seg.primary_stage
            final_label_source = 'llm_zero_shot'
        else:
            final_label = None
            final_label_source = None

        # Compute confidence tier
        if (
            seg.llm_run_consistency == 3
            and seg.llm_confidence_primary is not None
            and seg.llm_confidence_primary > 0.8
        ):
            confidence_tier = 'high'
        elif (
            seg.llm_run_consistency is not None
            and seg.llm_run_consistency >= 2
            and seg.llm_confidence_primary is not None
            and seg.llm_confidence_primary > 0.6
        ):
            confidence_tier = 'medium'
        else:
            confidence_tier = 'low'

        rows.append({
            'segment_id': seg.segment_id,
            'trial_id': seg.trial_id,
            'participant_id': seg.participant_id,
            'session_id': seg.session_id,
            'session_number': seg.session_number,
            'segment_index': seg.segment_index,
            'start_time_ms': seg.start_time_ms,
            'end_time_ms': seg.end_time_ms,
            'total_segments_in_session': seg.total_segments_in_session,
            'speaker': seg.speaker,
            'text': seg.text,
            'word_count': seg.word_count,
            'primary_stage': seg.primary_stage,
            'secondary_stage': seg.secondary_stage,
            'llm_confidence_primary': seg.llm_confidence_primary,
            'llm_confidence_secondary': seg.llm_confidence_secondary,
            'llm_justification': seg.llm_justification,
            'llm_run_consistency': seg.llm_run_consistency,
            'human_label': seg.human_label,
            'human_secondary_label': seg.human_secondary_label,
            'adjudicated_label': seg.adjudicated_label,
            'in_human_coded_subset': seg.in_human_coded_subset,
            'label_status': seg.label_status,
            'final_label': final_label,
            'final_label_source': final_label_source,
            'label_confidence_tier': confidence_tier,
        })

    df = pd.DataFrame(rows)

    # Save as both CSV and JSONL
    csv_path = output_path.replace('.jsonl', '.csv')
    df.to_csv(csv_path, index=False)
    df.to_json(output_path, orient='records', lines=True)

    # Print class distribution
    participant_labeled = df[
        (df['speaker'] == 'participant') & (df['final_label'].notna())
    ]
    print("\nFinal label distribution:")
    if len(participant_labeled) > 0:
        print(participant_labeled['final_label'].value_counts().sort_index())
    print(f"\nTotal segments: {len(df)}")
    print(f"Participant segments with labels: {len(participant_labeled)}")

    return df


# ---------------------------------------------------------------------------
# Output 2: Session Adjacency Index
# ---------------------------------------------------------------------------

def build_session_adjacency_index(
    segments_df: pd.DataFrame,
    output_path: str,
) -> List[Dict]:
    """
    Produce the session adjacency index for CFiCS graph construction.

    For each session, computes the interleaved sequence of participant and
    therapist segments and pre-computes the edge pairs that CFiCS needs.
    """
    sessions = []

    for session_id, session_df in segments_df.groupby('session_id'):
        session_df = session_df.sort_values('segment_index')

        all_ids = session_df['segment_id'].tolist()
        participant_ids = session_df[
            session_df['speaker'] == 'participant'
        ]['segment_id'].tolist()
        therapist_ids = session_df[
            session_df['speaker'] == 'therapist'
        ]['segment_id'].tolist()

        # Therapist-to-participant pairs
        t_to_p_pairs = []
        for i in range(1, len(session_df)):
            current = session_df.iloc[i]
            previous = session_df.iloc[i - 1]
            if (
                current['speaker'] == 'participant'
                and previous['speaker'] == 'therapist'
            ):
                t_to_p_pairs.append([
                    previous['segment_id'],
                    current['segment_id'],
                ])

        # Participant sequential pairs
        p_sequential = []
        for i in range(1, len(participant_ids)):
            p_sequential.append([participant_ids[i - 1], participant_ids[i]])

        sessions.append({
            'session_id': session_id,
            'segment_sequence': all_ids,
            'participant_segments': participant_ids,
            'therapist_segments': therapist_ids,
            'therapist_to_participant_pairs': t_to_p_pairs,
            'participant_sequential_pairs': p_sequential,
        })

    with open(output_path, 'w') as f:
        for session in sessions:
            f.write(json.dumps(session) + '\n')

    return sessions


# ---------------------------------------------------------------------------
# Output 3: Human Validation Report
# ---------------------------------------------------------------------------

def generate_validation_report(
    segments_df: pd.DataFrame,
    content_validity_results: Dict,
    human_intercoder_kappa: float,
    output_path: str,
) -> Dict:
    """
    Produce the structured validation report.

    Adapted from classify_ctl_results.py and classify_reddit_results.py,
    which aggregate metrics across constructs into summary tables.
    """
    human_subset = segments_df[segments_df['in_human_coded_subset'] == True]

    if len(human_subset) == 0:
        print("Warning: no human-coded segments found")
        return {}

    y_true = human_subset['human_label'].values
    y_pred = human_subset['primary_stage'].values

    # Remove NaN pairs
    valid_mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true = y_true[valid_mask].astype(int)
    y_pred = y_pred[valid_mask].astype(int)

    if len(y_true) == 0:
        print("Warning: no valid human-LLM label pairs")
        return {}

    metrics = compute_validation_metrics(y_true, y_pred)

    # Triplicate consistency rate
    labeled_participants = segments_df[
        (segments_df['speaker'] == 'participant')
        & (segments_df['llm_run_consistency'].notna())
    ]
    consistency_rate = (
        (labeled_participants['llm_run_consistency'] == 3).mean()
        if len(labeled_participants) > 0
        else 0
    )

    # Stage distribution
    stage_dist = segments_df[
        segments_df['speaker'] == 'participant'
    ]['primary_stage'].value_counts().to_dict()

    report = {
        **metrics,
        'content_validity_sensitivity': content_validity_results,
        'triplicate_consistency_rate': round(float(consistency_rate), 4),
        'human_intercoder_kappa': human_intercoder_kappa,
        'total_segments_labeled': int(
            segments_df[
                (segments_df['speaker'] == 'participant')
                & (segments_df['primary_stage'].notna())
            ].shape[0]
        ),
        'total_human_coded': int(len(human_subset)),
        'segments_per_stage_distribution': {
            str(int(k)): int(v) for k, v in stage_dist.items()
            if not pd.isna(k)
        },
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    return report


# ---------------------------------------------------------------------------
# Output 4: Stage Definition File
# ---------------------------------------------------------------------------

def export_stage_definitions(output_path: str) -> None:
    """Produce Output 4: stage definitions in JSON format."""
    with open(output_path, 'w') as f:
        json.dump(get_stage_definitions_json(), f, indent=2)


# ---------------------------------------------------------------------------
# Output 5: Content Validity Test Set
# ---------------------------------------------------------------------------

def export_content_validity_test_set(
    test_items: List[Dict],
    output_path: str,
) -> None:
    """Produce Output 5: content validity test set as JSONL."""
    with open(output_path, 'w') as f:
        for item in test_items:
            f.write(json.dumps(item) + '\n')
