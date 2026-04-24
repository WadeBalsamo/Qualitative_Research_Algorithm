"""Assembles the master segment dataset from classified segments."""

import json
from typing import List, Dict, Optional

import pandas as pd

from classification_tools.data_structures import Segment
from .. import output_paths as _paths
from ._common import _ms_to_hms, _fmt_conf, _theme_name_from, _summarize_rationales


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
            'cohort_id': seg.cohort_id,
            'session_variant': seg.session_variant,
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
            # Interrater-reliability fields (getattr guards against older Segment schemas)
            'rater_ids': (json.dumps(v) if (v := getattr(seg, 'rater_ids', None)) else None),
            'rater_votes': (json.dumps(v) if (v := getattr(seg, 'rater_votes', None)) else None),
            'agreement_level': getattr(seg, 'agreement_level', None),
            'agreement_fraction': getattr(seg, 'agreement_fraction', None),
            'needs_review': getattr(seg, 'needs_review', False),
            'consensus_vote': (
                json.dumps(cv) if isinstance(cv := getattr(seg, 'consensus_vote', None), str)
                else cv
            ),
            'tie_broken_by_confidence': getattr(seg, 'tie_broken_by_confidence', False),
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
