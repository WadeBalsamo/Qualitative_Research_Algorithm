"""
orchestrator.py
---------------
Top-level pipeline orchestrator.

Generalized from vamr_labeling/pipeline.py. Key changes:
- Accepts a ThemeFramework parameter (not hardcoded to VA-MR)
- Optionally runs codebook classification alongside theme labeling
- Uses generalized shared/ validation utilities
- Uses pipeline/ sub-modules instead of vamr_labeling/ modules
"""

import os
import datetime
from collections import Counter
from dataclasses import asdict
from typing import List, Optional

import pandas as pd

from shared.data_structures import Segment
from shared.llm_client import LLMClient, LLMClientConfig
from shared.validation import create_balanced_evaluation_set
from theme_labeler.theme_schema import ThemeFramework
from theme_labeler.zero_shot_classifier import (
    classify_segments_zero_shot,
    create_content_validity_test_set,
)
from theme_labeler.response_parser import parse_all_results

from .config import PipelineConfig
from .transcript_ingestion import (
    TranscriptSegmenter,
    load_diarized_session,
    discover_session_files,
)
from .dataset_assembly import (
    assemble_master_dataset,
    build_session_adjacency_index,
    export_theme_definitions,
    export_content_validity_test_set,
    compute_session_stage_progression,
)

from codebook_classifier.embedding_classifier import EmbeddingCodebookClassifier
from codebook_classifier.llm_classifier import LLMCodebookClassifier
from codebook_classifier.ensemble import CodebookEnsemble


def run_full_pipeline(
    config: PipelineConfig,
    framework: ThemeFramework,
    codebook=None,
) -> pd.DataFrame:
    """
    Execute the complete classification pipeline.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration with paths, model settings, and thresholds.
    framework : ThemeFramework
        The theme/stage framework to use for classification.

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
        metadata.setdefault('trial_id', config.trial_id)
        metadata.setdefault('participant_id', 'unknown')
        metadata.setdefault('session_id', os.path.basename(os.path.dirname(session_file)))
        metadata.setdefault('session_number', 1)

        segments = segmenter.segment_session(
            session_data['sentences'], metadata
        )
        all_segments.extend(segments)

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

    content_validity_items = create_content_validity_test_set(framework)
    export_theme_definitions(
        framework,
        os.path.join(output_dir, 'theme_definitions.json'),
    )
    export_content_validity_test_set(
        content_validity_items,
        os.path.join(output_dir, 'content_validity_test_set.jsonl'),
    )
    print(f"Built {len(content_validity_items)} content validity test items")
    print("Exported theme_definitions.json and content_validity_test_set.jsonl")

    # ------------------------------------------------------------------
    # Stage 3: Zero-Shot LLM Theme Classification
    # ------------------------------------------------------------------
    if config.run_theme_labeler:
        print("\n" + "=" * 60)
        print("STAGE 3: Zero-Shot LLM Theme Classification")
        print("=" * 60)

        theme_config = config.theme_classification
        theme_config.output_dir = os.path.join(output_dir, 'llm_raw')

        results_all, metadata_all = classify_segments_zero_shot(
            segments=all_segments,
            framework=framework,
            config=theme_config,
            resume_from=config.resume_from,
        )

        # ------------------------------------------------------------------
        # Stage 4: Response Parsing
        # ------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("STAGE 4: Response Parsing")
        print("=" * 60)

        name_to_id = framework.build_name_to_id_map()
        all_segments, parse_stats = parse_all_results(
            results_all, all_segments, name_to_id
        )
    else:
        print("\n  Skipping theme classification (run_theme_labeler=False)")

    # ------------------------------------------------------------------
    # Stage 3b: Codebook Classification (optional)
    # ------------------------------------------------------------------
    if config.run_codebook_classifier:
        print("\n" + "=" * 60)
        print("STAGE 3b: Codebook Classification")
        print("=" * 60)

        if codebook is None:
            from codebook_classifier.codebooks.phenomenology import get_phenomenology_codebook
            codebook = get_phenomenology_codebook()

        # Set up codebook output directory and exemplar export path
        codebook_output_dir = os.path.join(output_dir, 'codebook_raw')
        os.makedirs(codebook_output_dir, exist_ok=True)
        config.codebook_embedding.exemplar_export_path = os.path.join(
            codebook_output_dir, 'found_exemplar_utterances.json'
        )

        # Embedding classification
        print("  Running embedding-based classification...")
        embedding_classifier = EmbeddingCodebookClassifier(config.codebook_embedding)
        embedding_results = embedding_classifier.classify_segments(
            all_segments, codebook
        )

        # LLM classification
        print("  Running LLM-based classification...")
        theme_cfg = config.theme_classification
        llm_cfg = LLMClientConfig(
            backend=theme_cfg.backend,
            api_key=theme_cfg.api_key,
            replicate_api_token=theme_cfg.replicate_api_token,
            model=theme_cfg.model,
        )
        llm_client = LLMClient(llm_cfg)
        llm_classifier = LLMCodebookClassifier(llm_client, config.codebook_llm)
        llm_results = llm_classifier.classify_segments(
            all_segments, codebook, output_dir=codebook_output_dir,
        )

        # Ensemble reconciliation
        print("  Running ensemble reconciliation...")
        ensemble = CodebookEnsemble(config.codebook_ensemble)
        ensemble_results = ensemble.reconcile(embedding_results, llm_results)

        # Populate Segment codebook fields
        for seg in all_segments:
            if seg.segment_id in ensemble_results:
                ens = ensemble_results[seg.segment_id]
                seg.codebook_labels_embedding = sorted(
                    a.code_id for a in embedding_results.get(seg.segment_id, [])
                )
                seg.codebook_labels_llm = sorted(
                    a.code_id for a in llm_results.get(seg.segment_id, [])
                )
                seg.codebook_labels_ensemble = ens.final_codes
                seg.codebook_disagreements = [
                    d['code_id'] for d in ens.disagreement_details
                ]
                seg.codebook_confidence = {
                    a.code_id: a.confidence for a in ens.final_assignments
                }

        n_coded = sum(
            1 for s in all_segments
            if s.codebook_labels_ensemble and len(s.codebook_labels_ensemble) > 0
        )
        print(f"  Codebook classification complete: {n_coded} segments with codes")

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
            n_per_class=config.validation.n_per_class,
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

    confidence_tier_config = asdict(config.confidence_tiers)
    master_df = assemble_master_dataset(
        all_segments,
        os.path.join(output_dir, f'master_segments_{ts}.jsonl'),
        confidence_tiers=confidence_tier_config,
    )

    build_session_adjacency_index(
        master_df,
        os.path.join(output_dir, f'session_adjacency_{ts}.jsonl'),
    )

    id_to_short = framework.build_id_to_short_map()
    progression_df = compute_session_stage_progression(master_df, id_to_short)
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
