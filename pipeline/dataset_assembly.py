"""
dataset_assembly.py
-------------------
Dataset assembly: produces output files from classified segments.
"""

import json
from typing import List, Dict, Optional

import pandas as pd

from shared.data_structures import Segment
from theme_labeler.theme_schema import ThemeFramework


# ---------------------------------------------------------------------------
# Output 1: Master Segment Dataset
# ---------------------------------------------------------------------------

def assemble_master_dataset(
    segments: List[Segment],
    output_path: str,
    confidence_tiers: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Produce the master segment dataset.

    Computes final_label and label_confidence_tier following the priority:
    adjudicated > human_consensus > llm_zero_shot.
    """
    ct = confidence_tiers or {}
    high_consistency = ct.get('high_consistency', 3)
    high_confidence = ct.get('high_confidence', 0.8)
    medium_min_consistency = ct.get('medium_min_consistency', 2)
    medium_min_confidence = ct.get('medium_min_confidence', 0.6)

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
            seg.llm_run_consistency == high_consistency
            and seg.llm_confidence_primary is not None
            and seg.llm_confidence_primary > high_confidence
        ):
            confidence_tier = 'high'
        elif (
            seg.llm_run_consistency is not None
            and seg.llm_run_consistency >= medium_min_consistency
            and seg.llm_confidence_primary is not None
            and seg.llm_confidence_primary > medium_min_confidence
        ):
            confidence_tier = 'medium'
        else:
            confidence_tier = 'low'

        row = {
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
            # Theme labels
            'primary_stage': seg.primary_stage,
            'secondary_stage': seg.secondary_stage,
            'llm_confidence_primary': seg.llm_confidence_primary,
            'llm_confidence_secondary': seg.llm_confidence_secondary,
            'llm_justification': seg.llm_justification,
            'llm_run_consistency': seg.llm_run_consistency,
            # Codebook labels (if populated)
            'codebook_labels_embedding': seg.codebook_labels_embedding,
            'codebook_labels_llm': seg.codebook_labels_llm,
            'codebook_labels_ensemble': seg.codebook_labels_ensemble,
            'codebook_disagreements': seg.codebook_disagreements,
            # Validation
            'human_label': seg.human_label,
            'human_secondary_label': seg.human_secondary_label,
            'adjudicated_label': seg.adjudicated_label,
            'in_human_coded_subset': seg.in_human_coded_subset,
            'label_status': seg.label_status,
            # Final
            'final_label': final_label,
            'final_label_source': final_label_source,
            'label_confidence_tier': confidence_tier,
        }
        rows.append(row)

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
# Output 3: Theme Definition File
# ---------------------------------------------------------------------------

def export_theme_definitions(
    framework: ThemeFramework,
    output_path: str,
) -> None:
    """Export theme/stage definitions as JSON."""
    with open(output_path, 'w') as f:
        json.dump(framework.to_json(), f, indent=2)


# ---------------------------------------------------------------------------
# Output 5: Content Validity Test Set
# ---------------------------------------------------------------------------

def export_content_validity_test_set(
    test_items: List[Dict],
    output_path: str,
) -> None:
    """Export content validity test set as JSONL."""
    with open(output_path, 'w') as f:
        for item in test_items:
            f.write(json.dumps(item) + '\n')


# ---------------------------------------------------------------------------
# Longitudinal Stage Tracking
# ---------------------------------------------------------------------------

def compute_session_stage_progression(
    segments_df: pd.DataFrame,
    id_to_short: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """
    Track theme/stage progression within each session.

    Parameters
    ----------
    segments_df : pd.DataFrame
        Master segment dataset.
    id_to_short : dict, optional
        Mapping from theme_id to short name. If None, uses string
        representation of the IDs.
    """
    if id_to_short is None:
        id_to_short = {}

    session_progressions = []

    for session_id, session_df in segments_df.groupby('session_id'):
        participant_df = session_df[
            (session_df['speaker'] == 'participant')
            & (session_df['primary_stage'].notna())
        ].sort_values('segment_index')

        if len(participant_df) == 0:
            continue

        stages = participant_df['primary_stage'].astype(int).tolist()

        transitions = []
        forward = 0
        backward = 0
        for i in range(1, len(stages)):
            from_stage = stages[i - 1]
            to_stage = stages[i]
            transitions.append((from_stage, to_stage))
            if to_stage > from_stage:
                forward += 1
            elif to_stage < from_stage:
                backward += 1

        stage_counts = {}
        for s in stages:
            stage_counts[s] = stage_counts.get(s, 0) + 1

        dominant_stage = max(stage_counts, key=stage_counts.get)

        session_progressions.append({
            'session_id': session_id,
            'trial_id': participant_df['trial_id'].iloc[0],
            'participant_id': participant_df['participant_id'].iloc[0],
            'n_segments': len(stages),
            'stage_sequence': stages,
            'stage_transitions': transitions,
            'forward_transitions': forward,
            'backward_transitions': backward,
            'lateral_transitions': len(transitions) - forward - backward,
            'max_stage_reached': max(stages),
            'dominant_stage': dominant_stage,
            'dominant_stage_name': id_to_short.get(dominant_stage, str(dominant_stage)),
            'stage_distribution': {
                id_to_short.get(k, str(k)): v
                for k, v in sorted(stage_counts.items())
            },
        })

    return pd.DataFrame(session_progressions)
