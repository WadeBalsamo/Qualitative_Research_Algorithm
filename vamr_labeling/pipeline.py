"""
pipeline.py
-----------
Top-level orchestrator for the data labeling pipeline.

Executes six sequential stages, each producing versioned, archived outputs:
    1. Transcript ingestion and segment boundary detection
    2. Construct operationalization
    3. Zero-shot LLM classification
    4. Response parsing and error handling
    5. Human validation set preparation
    6. Dataset assembly (5 output files)

This module ties together all sub-modules that can label the data to feed into AutoResearch-for-classification (BERT fine-tuning pipeline).
"""

import os
import datetime
from collections import Counter
from dataclasses import asdict
from typing import List, Dict, Optional

import pandas as pd

from .config import PipelineConfig, SegmentationConfig
from .data_structures import Segment
from .vamr_constructs import stage_definitions
from .transcript_ingestion import (
    TranscriptSegmenter,
    load_diarized_session,
    discover_session_files,
)
from .zero_shot_classifier import (
    classify_segments_zero_shot,
    create_content_validity_test_set,
)
from .response_parser import parse_all_results
from .validation import create_balanced_evaluation_set
from .dataset_assembly import (
    assemble_master_dataset,
    build_session_adjacency_index,
    generate_validation_report,
    export_stage_definitions,
    export_content_validity_test_set,
    compute_session_stage_progression,
)


def run_full_pipeline(config: PipelineConfig) -> pd.DataFrame:
    """
    Execute the complete data labeling pipeline.

    This is the top-level orchestrator that produces all five outputs.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration with paths, model settings, and thresholds.

    Returns
    -------
    pd.DataFrame
        The master segment dataset.
    """
    ts = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1: Transcript Ingestion and Segmentation
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STAGE 1: Transcript Ingestion and Segmentation")
    print("=" * 60)

    seg_config = {
        'embedding_model': config.segmentation.embedding_model,
        'min_segment_words': config.segmentation.min_segment_words,
        'max_segment_words': config.segmentation.max_segment_words,
        'silence_threshold_ms': config.segmentation.silence_threshold_ms,
        'semantic_shift_percentile': config.segmentation.semantic_shift_percentile,
    }
    segmenter = TranscriptSegmenter(seg_config)

    session_files = discover_session_files(config.transcript_dir)
    if not session_files:
        print(f"Warning: No session files found in {config.transcript_dir}")
        print("Looking for JSON files directly...")
        import glob
        session_files = sorted(glob.glob(
            os.path.join(config.transcript_dir, '**/*.json'), recursive=True
        ))

    all_segments: List[Segment] = []
    for session_file in session_files:
        session_data = load_diarized_session(session_file)
        metadata = session_data['metadata']
        # Ensure required metadata fields
        metadata.setdefault('trial_id', config.trial_id)
        metadata.setdefault('participant_id', 'unknown')
        metadata.setdefault('session_id', os.path.basename(os.path.dirname(session_file)))
        metadata.setdefault('session_number', 1)

        segments = segmenter.segment_session(
            session_data['sentences'], metadata
        )
        all_segments.extend(segments)

    # Set total_segments_in_session for each segment
    session_counts = Counter(s.session_id for s in all_segments)
    for seg in all_segments:
        seg.total_segments_in_session = session_counts[seg.session_id]

    print(f"Produced {len(all_segments)} segments from {len(session_files)} sessions")

    # ------------------------------------------------------------------
    # Stage 2: Construct Operationalization
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2: Construct Operationalization")
    print("=" * 60)

    content_validity_items = create_content_validity_test_set(stage_definitions)
    export_stage_definitions(
        os.path.join(output_dir, 'stage_definitions.json')
    )
    export_content_validity_test_set(
        content_validity_items,
        os.path.join(output_dir, 'content_validity_test_set.jsonl'),
    )
    print(f"Built {len(content_validity_items)} content validity test items")
    print("Exported stage_definitions.json and content_validity_test_set.jsonl")

    # ------------------------------------------------------------------
    # Stage 3: Zero-Shot LLM Classification
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 3: Zero-Shot LLM Classification")
    print("=" * 60)

    results_all, metadata_all = classify_segments_zero_shot(
        segments=all_segments,
        api_key=config.classification.api_key,
        model=config.classification.model,
        temperature=config.classification.temperature,
        n_runs=config.classification.n_runs,
        output_dir=os.path.join(output_dir, 'llm_raw'),
        randomize_codebook=config.classification.randomize_codebook,
        definitions=stage_definitions,
        backend=config.classification.backend,
        replicate_api_token=config.classification.replicate_api_token,
        max_new_tokens=config.classification.max_new_tokens,
        resume_from=config.resume_from,
    )

    # ------------------------------------------------------------------
    # Stage 4: Response Parsing
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 4: Response Parsing")
    print("=" * 60)

    all_segments, parse_stats = parse_all_results(results_all, all_segments)

    # ------------------------------------------------------------------
    # Stage 5: Preparing Human Validation Set
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 5: Preparing Human Validation Set")
    print("=" * 60)

    segments_df = pd.DataFrame([vars(s) for s in all_segments])
    participant_labeled = segments_df[
        (segments_df['speaker'] == 'participant')
        & (segments_df['primary_stage'].notna())
    ]

    if len(participant_labeled) > 0:
        eval_set = create_balanced_evaluation_set(
            participant_labeled,
            n_per_stage=config.validation.n_per_stage,
        )
        eval_set.to_csv(
            os.path.join(output_dir, 'human_coding_evaluation_set.csv'),
            index=False,
        )
        print(f"Exported {len(eval_set)} segments for human coding")
    else:
        print("No labeled participant segments available for evaluation set")

    # ------------------------------------------------------------------
    # Stage 6: Dataset Assembly
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 6: Dataset Assembly")
    print("=" * 60)

    # Output 1: Master segment dataset
    confidence_tier_config = asdict(config.confidence_tiers)
    master_df = assemble_master_dataset(
        all_segments,
        os.path.join(output_dir, f'master_segments_{ts}.jsonl'),
        confidence_tiers=confidence_tier_config,
    )

    # Output 2: Session adjacency index
    build_session_adjacency_index(
        master_df,
        os.path.join(output_dir, f'session_adjacency_{ts}.jsonl'),
    )

    # Output 3: Validation report (placeholder -- completed after human coding)
    # Output 4: Stage definitions (already exported in Stage 2)
    # Output 5: Content validity test set (already exported in Stage 2)

    # Output 6: Session stage progression (longitudinal tracking)
    progression_df = compute_session_stage_progression(master_df)
    if len(progression_df) > 0:
        progression_path = os.path.join(output_dir, f'session_stage_progression_{ts}.csv')
        progression_df.to_csv(progression_path, index=False)
        print(f"Exported session stage progression for {len(progression_df)} sessions")
        avg_forward = progression_df['forward_transitions'].mean()
        avg_backward = progression_df['backward_transitions'].mean()
        print(f"  Avg forward transitions per session: {avg_forward:.1f}")
        print(f"  Avg backward transitions per session: {avg_backward:.1f}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"All outputs in: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Complete human coding of {output_dir}/human_coding_evaluation_set.csv")
    print(f"  2. Run validation report generation")
    print(f"  3. Feed master_segments to AutoResearch's prepare.py")

    return master_df
