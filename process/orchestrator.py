"""
orchestrator.py
---------------
Top-level pipeline orchestrator.

Generalized from vamr_labeling/pipeline.py. Key changes:
- Accepts a ThemeFramework parameter (not hardcoded to VA-MR)
- Optionally runs codebook classification alongside theme labeling
- Uses generalized shared/ validation utilities
- Uses pipeline/ sub-modules instead of vamr_labeling/ modules
- Accepts optional observer (for UI feedback) and validator (for human-in-the-loop)
"""

import json
import os
import datetime
from collections import Counter
from dataclasses import asdict
from typing import List, Optional

import pandas as pd

from classification_tools.data_structures import Segment
from classification_tools.llm_client import LLMClient, LLMClientConfig
from classification_tools.validation import create_balanced_evaluation_set
from classification_tools.llm_classifier import (
    classify_segments_zero_shot,
    create_content_validity_test_set,
    LLMCodebookClassifier,
)
from classification_tools.response_parser import parse_all_results
from constructs.theme_schema import ThemeFramework

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
from .cross_validation import (
    compute_theme_codebook_cooccurrence,
    validate_codebook_hypothesis,
)
from .pipeline_hooks import PipelineObserver, SilentObserver

from codebook.embedding_classifier import EmbeddingCodebookClassifier
from codebook.ensemble import CodebookEnsemble


def run_full_pipeline(
    config: PipelineConfig,
    framework: ThemeFramework,
    codebook=None,
    observer: Optional[PipelineObserver] = None,
    validator=None,
) -> pd.DataFrame:
    """
    Execute the complete classification pipeline.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration with paths, model settings, and thresholds.
    framework : ThemeFramework
        The theme/stage framework to use for classification.
    codebook : optional
        Codebook instance for codebook classification.
    observer : PipelineObserver, optional
        Observer for pipeline events (UI feedback). Defaults to SilentObserver.
    validator : HumanValidator, optional
        Validator for interactive human-in-the-loop validation.

    Returns
    -------
    pd.DataFrame
        The master segment dataset.
    """
    if observer is None:
        observer = SilentObserver()

    ts = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    run_mode = getattr(config, 'run_mode', 'auto')

    # ------------------------------------------------------------------
    # Stage 1: Transcript Ingestion and Segmentation
    # ------------------------------------------------------------------
    observer.on_stage_start(
        "Transcript Ingestion and Segmentation", "1",
        explanation_key='ingestion',
    )

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
        observer.on_stage_progress(
            "Transcript Ingestion and Segmentation",
            f"Warning: No session files found in {config.transcript_dir}",
        )
        observer.on_stage_progress(
            "Transcript Ingestion and Segmentation",
            "Looking for JSON files directly...",
        )
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

    observer.on_stage_complete(
        "Transcript Ingestion and Segmentation",
        f"Produced {len(all_segments)} segments from {len(session_files)} sessions",
    )

    # ------------------------------------------------------------------
    # Stage 2: Construct Operationalization
    # ------------------------------------------------------------------
    observer.on_stage_start(
        "Construct Operationalization", "2",
        explanation_key='operationalization',
    )

    content_validity_items = create_content_validity_test_set(framework)
    export_theme_definitions(
        framework,
        os.path.join(output_dir, 'theme_definitions.json'),
    )
    export_content_validity_test_set(
        content_validity_items,
        os.path.join(output_dir, 'content_validity_test_set.jsonl'),
    )

    observer.on_stage_complete(
        "Construct Operationalization",
        f"Built {len(content_validity_items)} content validity test items; "
        "exported theme_definitions.json and content_validity_test_set.jsonl",
    )

    # ------------------------------------------------------------------
    # Stage 3: Zero-Shot LLM Theme Classification
    # ------------------------------------------------------------------
    if config.run_theme_labeler:
        observer.on_stage_start(
            "Zero-Shot LLM Theme Classification", "3",
            explanation_key='theme_classification',
        )

        theme_config = config.theme_classification
        theme_config.output_dir = os.path.join(output_dir, 'llm_raw')

        results_all, metadata_all = classify_segments_zero_shot(
            segments=all_segments,
            framework=framework,
            config=theme_config,
            resume_from=config.resume_from,
        )

        # Response Parsing
        observer.on_stage_progress(
            "Zero-Shot LLM Theme Classification",
            "Parsing LLM responses...",
        )

        name_to_id = framework.build_name_to_id_map()
        all_segments, parse_stats = parse_all_results(
            results_all, all_segments, name_to_id
        )

        observer.on_stage_complete(
            "Zero-Shot LLM Theme Classification",
            "Theme classification and response parsing complete",
        )

        # Human validation of uncertain theme classifications
        if validator and run_mode in ('interactive', 'review'):
            _validate_theme_classifications(all_segments, validator)
    else:
        observer.on_stage_progress(
            "Zero-Shot LLM Theme Classification",
            "Skipping theme classification (run_theme_labeler=False)",
        )

    # ------------------------------------------------------------------
    # Stage 3b: Codebook Classification (optional)
    # ------------------------------------------------------------------
    if config.run_codebook_classifier:
        observer.on_stage_start(
            "Codebook Classification", "3b",
            explanation_key='codebook_classification',
        )

        if codebook is None:
            from codebook.phenomenology_codebook import get_phenomenology_codebook
            codebook = get_phenomenology_codebook()

        # Set up codebook output directory and exemplar export path
        codebook_output_dir = os.path.join(output_dir, 'codebook_raw')
        os.makedirs(codebook_output_dir, exist_ok=True)
        config.codebook_embedding.exemplar_export_path = os.path.join(
            codebook_output_dir, 'found_exemplar_utterances.json'
        )

        # Embedding classification
        observer.on_stage_progress("Codebook Classification", "Running embedding-based classification...")
        embedding_classifier = EmbeddingCodebookClassifier(config.codebook_embedding)
        embedding_results = embedding_classifier.classify_segments(
            all_segments, codebook
        )

        # LLM classification
        observer.on_stage_progress("Codebook Classification", "Running LLM-based classification...")
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
        observer.on_stage_progress("Codebook Classification", "Running ensemble reconciliation...")
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

                # Human validation of disagreements
                if validator and run_mode in ('interactive', 'review') and ens.needs_human_review:
                    final_codes = validator.validate_codebook_disagreement(seg, ens)
                    if final_codes is not None:
                        seg.codebook_labels_ensemble = final_codes

        n_coded = sum(
            1 for s in all_segments
            if s.codebook_labels_ensemble and len(s.codebook_labels_ensemble) > 0
        )

        observer.on_stage_complete(
            "Codebook Classification",
            f"Codebook classification complete: {n_coded} segments with codes",
        )

    # ------------------------------------------------------------------
    # Stage 4: Cross-Validation (optional, when both theme and codebook)
    # ------------------------------------------------------------------
    if config.run_theme_labeler and config.run_codebook_classifier:
        observer.on_stage_start(
            "Cross-Validation (Theme <-> Codebook)", "4",
            explanation_key='cross_validation',
        )

        segments_df = pd.DataFrame([vars(s) for s in all_segments])
        cooccurrence = compute_theme_codebook_cooccurrence(
            segments_df, framework,
            codebook_label_column='codebook_labels_ensemble',
            theme_label_column='primary_stage',
        )
        validation_results = validate_codebook_hypothesis(
            cooccurrence, framework, min_lift=1.5
        )

        # Export results
        cv_output = os.path.join(output_dir, f'cross_validation_results_{ts}.json')
        with open(cv_output, 'w') as f:
            json.dump(validation_results, f, indent=2)
        observer.on_stage_progress(
            "Cross-Validation (Theme <-> Codebook)",
            f"Exported cross-validation results to {os.path.basename(cv_output)}",
        )

        # Report anomalies
        anomalies = _collect_cooccurrence_anomalies(validation_results)
        if anomalies:
            observer.on_stage_progress(
                "Cross-Validation (Theme <-> Codebook)",
                f"Found {len(anomalies)} co-occurrence anomalies",
            )
            anomaly_output = os.path.join(output_dir, f'cooccurrence_anomalies_{ts}.json')
            with open(anomaly_output, 'w') as f:
                json.dump(anomalies, f, indent=2)

        observer.on_stage_complete(
            "Cross-Validation (Theme <-> Codebook)",
            "Cross-validation complete",
        )

    # ------------------------------------------------------------------
    # Stage 5: Preparing Human Validation Set
    # ------------------------------------------------------------------
    observer.on_stage_start(
        "Preparing Human Validation Set", "5",
        explanation_key='human_validation_set',
    )

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
        observer.on_stage_complete(
            "Preparing Human Validation Set",
            f"Exported {len(eval_set)} segments for human coding",
        )
    else:
        observer.on_stage_complete(
            "Preparing Human Validation Set",
            "No labeled participant segments available for evaluation set",
        )

    # ------------------------------------------------------------------
    # Stage 6: Dataset Assembly
    # ------------------------------------------------------------------
    observer.on_stage_start(
        "Dataset Assembly", "6",
        explanation_key='dataset_assembly',
    )

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
        observer.on_stage_progress(
            "Dataset Assembly",
            f"Exported session stage progression for {len(progression_df)} sessions",
        )
        avg_forward = progression_df['forward_transitions'].mean()
        avg_backward = progression_df['backward_transitions'].mean()
        observer.on_stage_progress(
            "Dataset Assembly",
            f"Avg forward transitions per session: {avg_forward:.1f}",
        )
        observer.on_stage_progress(
            "Dataset Assembly",
            f"Avg backward transitions per session: {avg_backward:.1f}",
        )

    observer.on_stage_complete(
        "Dataset Assembly",
        f"Master dataset assembled with {len(master_df)} segments",
    )

    # ------------------------------------------------------------------
    # Pipeline Complete
    # ------------------------------------------------------------------
    observer.on_pipeline_complete(
        output_dir,
        total_segments=len(master_df),
    )

    # Export validation log if validator was used
    if validator and hasattr(validator, 'validation_log') and validator.validation_log:
        log_path = os.path.join(output_dir, f'human_validation_log_{ts}.jsonl')
        with open(log_path, 'w') as f:
            for entry in validator.validation_log:
                f.write(json.dumps(entry) + '\n')

    return master_df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_theme_classifications(all_segments: List[Segment], validator) -> None:
    """Identify and validate uncertain theme classifications."""
    uncertain = []

    for segment in all_segments:
        if segment.primary_stage is None:
            continue

        reasons = []
        if segment.llm_run_consistency and segment.llm_run_consistency < 3:
            reasons.append("inconsistency")
        if segment.llm_confidence_primary and segment.llm_confidence_primary < 0.6:
            reasons.append("low_confidence")

        if reasons:
            uncertain.append((segment, reasons))

    if not uncertain:
        print(f"  All {len(all_segments)} theme classifications are confident and consistent")
        return

    print(f"\n  Found {len(uncertain)} uncertain classifications needing validation")
    for segment, reasons in uncertain:
        result = validator.validate_theme_classification(
            segment,
            " + ".join(reasons),
        )
        if result:
            primary_id, secondary_id = result
            segment.primary_stage = primary_id
            segment.secondary_stage = secondary_id


def _collect_cooccurrence_anomalies(validation_results: dict) -> list:
    """Collect anomalies from cross-validation results."""
    anomalies = []
    for theme_key, results in validation_results.items():
        for anomaly in results.get('unexpected', []):
            anomalies.append({
                'theme': theme_key,
                'code': anomaly['code'],
                'type': 'unexpected_strong_association',
                'lift': anomaly['lift'],
            })
        for missing in results.get('unconfirmed', []):
            if missing['count'] > 0:
                anomalies.append({
                    'theme': theme_key,
                    'code': missing['code'],
                    'type': 'weak_association',
                    'lift': missing['lift'],
                })
    return anomalies
