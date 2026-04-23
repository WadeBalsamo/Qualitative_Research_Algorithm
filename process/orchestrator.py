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
    ConversationalSegmenter,
    load_diarized_session,
    load_vtt_session,
    discover_session_files,
    parse_session_id_metadata,
)
from .llm_segmentation import LLMSegmentationRefiner
from .process_logger import ProcessLogger
from .dataset_assembly import (
    assemble_master_dataset,
    export_theme_definitions,
    export_content_validity_test_set,
    export_coded_transcript,
    export_per_transcript_stats,
    export_cumulative_report,
    export_human_classification_forms,
    export_flagged_for_review,
    export_training_data,
)
from .cross_validation import (
    compute_theme_codebook_cooccurrence,
    validate_codebook_hypothesis,
)
from .pipeline_hooks import PipelineObserver, SilentObserver

from codebook.embedding_classifier import EmbeddingCodebookClassifier, ensure_embedding_model_ready
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
    # Pre-flight: ensure embedding model is downloaded before the pipeline
    # starts.  Both segmentation and codebook classification use the same
    # model; downloading it now avoids a surprise mid-run pause.
    # ------------------------------------------------------------------
    seg_emb_model = config.segmentation.embedding_model
    print(f"  Checking embedding model: {seg_emb_model}")
    ensure_embedding_model_ready(seg_emb_model)

    # Process logger (verbose segmentation mode)
    _verbose = getattr(config.segmentation, 'verbose_segmentation', False)
    meta_dir = os.path.join(output_dir, 'meta')
    os.makedirs(meta_dir, exist_ok=True)
    _plog_path = os.path.join(meta_dir, 'process_log.txt') if _verbose else None
    plog = ProcessLogger(_plog_path)

    # Load speaker anonymization key, with priority: meta/ > imported config
    speaker_key_path = os.path.join(meta_dir, 'speaker_anonymization_key.json')
    _existing_speaker_map: dict = {}
    _use_unknown_prefix: bool = False

    if os.path.exists(speaker_key_path):
        # Re-run: meta/ key always takes precedence for stability
        try:
            with open(speaker_key_path) as _f:
                _raw_key = json.load(_f)
            _existing_speaker_map = {
                name: (entry['role'], entry['anonymized_id'])
                for name, entry in _raw_key.items()
            }
        except Exception:
            _existing_speaker_map = {}
        # Mark that we should use unknown prefix if key was imported in config
        _use_unknown_prefix = bool(getattr(config, 'speaker_anonymization_key_path', None))

    elif getattr(config, 'speaker_anonymization_key_path', None):
        # First run with imported key: seed from it
        try:
            with open(config.speaker_anonymization_key_path) as _f:
                _raw_key = json.load(_f)
            _existing_speaker_map = {
                name: (entry['role'], entry['anonymized_id'])
                for name, entry in _raw_key.items()
            }
            _use_unknown_prefix = True
        except Exception as e:
            print(f"  Warning: Could not load speaker_anonymization_key_path: {e}")
            _existing_speaker_map = {}

    # ------------------------------------------------------------------
    # Stage 1: Transcript Ingestion and Segmentation
    # ------------------------------------------------------------------
    observer.on_stage_start(
        "Transcript Ingestion and Segmentation", "1",
        explanation_key='ingestion',
    )

    # Build speaker exclusion/isolation lists from SpeakerFilterConfig
    sf = config.speaker_filter
    excluded_speakers = sf.speakers if sf.mode == 'exclude' else []
    isolated_speakers = sf.speakers if sf.mode == 'isolate' else []

    seg_config = {
        'embedding_model': config.segmentation.embedding_model,
        'min_segment_words': config.segmentation.min_segment_words,
        'max_segment_words': config.segmentation.max_segment_words,
        'silence_threshold_ms': config.segmentation.silence_threshold_ms,
        'semantic_shift_percentile': config.segmentation.semantic_shift_percentile,
        'min_segment_words_conversational': config.segmentation.min_segment_words_conversational,
        'max_segment_words_conversational': config.segmentation.max_segment_words_conversational,
        # Advanced grouping parameters
        'max_gap_seconds': getattr(config.segmentation, 'max_gap_seconds', 30.0),
        'min_words_per_sentence': getattr(config.segmentation, 'min_words_per_sentence', 10),
        'max_segment_duration_seconds': getattr(config.segmentation, 'max_segment_duration_seconds', 300.0),
        # Speaker filtering applied at sentence level before segmentation
        'excluded_speakers': excluded_speakers,
        'isolated_speakers': isolated_speakers,
        'speaker_filter_mode': config.speaker_filter.mode,
        # Adaptive threshold / dual-window / clustering
        'use_adaptive_threshold': getattr(config.segmentation, 'use_adaptive_threshold', True),
        'min_prominence': getattr(config.segmentation, 'min_prominence', 0.05),
        'broad_window_size': getattr(config.segmentation, 'broad_window_size', 7),
        'use_topic_clustering': getattr(config.segmentation, 'use_topic_clustering', False),
        # Process logger
        'process_logger': plog,
        # Persistent speaker map — ensures stable participant_N IDs across runs
        'existing_speaker_map': _existing_speaker_map,
        # Use unknown prefix for new speakers when key is imported
        'use_unknown_prefix': _use_unknown_prefix,
    }
    use_conv = getattr(config.segmentation, 'use_conversational_segmenter', True)
    use_llm_refine = getattr(config.segmentation, 'use_llm_refinement', True)
    segmenter = ConversationalSegmenter(seg_config) if use_conv else TranscriptSegmenter(seg_config)

    # Lazily create LLM refiner if enabled (reuses theme classification backend)
    llm_refiner = None
    if use_llm_refine and use_conv:
        theme_cfg = config.theme_classification
        refiner_llm_cfg = LLMClientConfig(
            backend=theme_cfg.backend,
            api_key=theme_cfg.api_key,
            replicate_api_token=theme_cfg.replicate_api_token,
            model=theme_cfg.model,
            lmstudio_base_url=getattr(theme_cfg, 'lmstudio_base_url', 'http://127.0.0.1:1234/v1'),
            no_reasoning=True,  # Segmentation uses simple true/false prompts; CoT wastes tokens
            process_logger=plog,
        )
        llm_refiner = LLMSegmentationRefiner(
            LLMClient(refiner_llm_cfg),
            {
                'mode': getattr(config.segmentation, 'llm_refinement_mode', 'full'),
                'ambiguity_threshold': getattr(config.segmentation, 'llm_ambiguity_threshold', 0.15),
                'batch_size': getattr(config.segmentation, 'llm_batch_size', 5),
                'excluded_speakers': excluded_speakers,
                'max_context_words': config.segmentation.max_segment_words_conversational,
                'max_context_duration_s': getattr(config.segmentation, 'max_segment_duration_seconds', 300.0),
                'max_gap_seconds': getattr(config.segmentation, 'max_gap_seconds', 30.0),
                'embedding_model': config.segmentation.embedding_model,
                'process_logger': plog,
            },
            speaker_normalizer=segmenter.speaker_norm,
        )

    # When using conversational segmenter, default to participant-only classification
    # (therapist segments are filtered out early; set classify_all_segments=True to override)
    if use_conv and not hasattr(config.theme_classification, 'classify_all_segments'):
        config.theme_classification.classify_all_segments = False

    session_files = discover_session_files(config.transcript_dir)
    if not session_files:
        observer.on_stage_progress(
            "Transcript Ingestion and Segmentation",
            f"Warning: No session files found in {config.transcript_dir}",
        )
        observer.on_stage_progress(
            "Transcript Ingestion and Segmentation",
            "Looking for JSON and VTT files directly...",
        )
        import glob as _glob
        session_files = sorted(
            set(_glob.glob(os.path.join(config.transcript_dir, '**/*.json'), recursive=True))
            | set(_glob.glob(os.path.join(config.transcript_dir, '**/*.vtt'), recursive=True))
        )

    all_segments: List[Segment] = []
    for session_file in session_files:
        if session_file.lower().endswith('.vtt'):
            session_data = load_vtt_session(session_file)
        else:
            session_data = load_diarized_session(session_file)
        metadata = session_data['metadata']
        metadata.setdefault('trial_id', config.trial_id)

        # For VTT files, use the filename stem as session_id and parse cohort/session metadata
        if session_file.lower().endswith('.vtt'):
            stem = os.path.splitext(os.path.basename(session_file))[0]
            parsed = parse_session_id_metadata(stem)
            metadata.setdefault('session_id', stem)
            metadata.setdefault('session_number', parsed['session_number'])
            metadata.setdefault('cohort_id', parsed['cohort_id'])
            metadata.setdefault('session_variant', parsed['session_variant'])
        else:
            default_session_id = os.path.basename(os.path.dirname(session_file))
            parsed = parse_session_id_metadata(default_session_id)
            metadata.setdefault('session_id', default_session_id)
            metadata.setdefault('session_number', parsed['session_number'])
            metadata.setdefault('cohort_id', parsed['cohort_id'])
            metadata.setdefault('session_variant', parsed['session_variant'])
        metadata.setdefault('source_file', session_file)

        if llm_refiner and use_conv:
            result = segmenter.segment_session(
                session_data['sentences'], metadata,
                return_intermediates=True,
            )
            segments = result['segments']
            segments = llm_refiner.refine(
                segments,
                result['sentences'],
                result['sim_curve'],
                result['embeddings'],
                result.get('boundary_confidence'),
                original_sentences=result.get('original_sentences'),
            )
        else:
            segments = segmenter.segment_session(
                session_data['sentences'], metadata
            )
        all_segments.extend(segments)

    session_counts = Counter(s.session_id for s in all_segments)
    for seg in all_segments:
        seg.total_segments_in_session = session_counts[seg.session_id]

    plog.close()

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
        os.path.join(meta_dir, 'theme_definitions.json'),
    )
    export_content_validity_test_set(
        content_validity_items,
        os.path.join(meta_dir, 'content_validity_test_set.jsonl'),
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

        segments_to_classify = _apply_speaker_filter(all_segments, config.speaker_filter)
        n_filtered = len(all_segments) - len(segments_to_classify)
        if n_filtered:
            observer.on_stage_progress(
                "Zero-Shot LLM Theme Classification",
                f"Speaker filter ({config.speaker_filter.mode}): "
                f"{len(segments_to_classify)} of {len(all_segments)} segments selected "
                f"({n_filtered} excluded)",
            )

        results_all, metadata_all = classify_segments_zero_shot(
            segments=segments_to_classify,
            framework=framework,
            config=theme_config,
            resume_from=config.resume_from,
            process_logger=plog,
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

        # Persist codebook definitions so standalone `qra analyze` can build
        # the human-readable codebook reference report without the in-memory object.
        _cb_def_path = os.path.join(meta_dir, 'codebook_definitions.json')
        if not os.path.exists(_cb_def_path):
            _cb_defs = {
                'name': codebook.name,
                'version': codebook.version,
                'description': codebook.description,
                'codes': [
                    {
                        'code_id': c.code_id,
                        'category': c.category,
                        'domain': c.domain,
                        'description': c.description,
                        'subcodes': c.subcodes,
                        'inclusive_criteria': c.inclusive_criteria,
                        'exclusive_criteria': c.exclusive_criteria,
                        'exemplar_utterances': c.exemplar_utterances,
                    }
                    for c in codebook.codes
                ],
            }
            with open(_cb_def_path, 'w') as _f:
                json.dump(_cb_defs, _f, indent=2)

        # Set up codebook output directory and exemplar export path
        codebook_output_dir = os.path.join(output_dir, 'codebook_raw')
        os.makedirs(codebook_output_dir, exist_ok=True)
        config.codebook_embedding.exemplar_export_path = os.path.join(
            codebook_output_dir, 'found_exemplar_utterances.json'
        )

        # Embedding classification
        observer.on_stage_progress("Codebook Classification", "Running embedding-based classification...")
        cb_segments = _apply_speaker_filter(all_segments, config.speaker_filter)

        embedding_classifier = EmbeddingCodebookClassifier(config.codebook_embedding)
        embedding_results = embedding_classifier.classify_segments(
            cb_segments, codebook
        )

        # LLM classification — reuses the same backend/model as theme classification
        observer.on_stage_progress("Codebook Classification", "Running LLM-based classification...")
        theme_cfg = config.theme_classification
        # Use the primary model (not per-run models) for codebook LLM classification.
        # Per-run model rotation is a theme-classification feature; codebook uses one model.
        codebook_model = theme_cfg.model
        llm_cfg = LLMClientConfig(
            backend=theme_cfg.backend,
            api_key=theme_cfg.api_key,
            replicate_api_token=theme_cfg.replicate_api_token,
            model=codebook_model,
            temperature=theme_cfg.temperature,
            lmstudio_base_url=getattr(theme_cfg, 'lmstudio_base_url', 'http://127.0.0.1:1234/v1'),
            ollama_host=getattr(theme_cfg, 'ollama_host', '0.0.0.0'),
            ollama_port=getattr(theme_cfg, 'ollama_port', 11434),
            process_logger=plog,
        )
        llm_client = LLMClient(llm_cfg)
        # Set the output dir on the codebook_llm config so checkpoints land in the right place
        config.codebook_llm.output_dir = codebook_output_dir
        llm_classifier = LLMCodebookClassifier(llm_client, config.codebook_llm)
        llm_results = llm_classifier.classify_segments(
            cb_segments, codebook, output_dir=codebook_output_dir,
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

    observer.on_stage_complete(
        "Dataset Assembly",
        f"Master dataset assembled with {len(master_df)} segments",
    )

    # ------------------------------------------------------------------
    # Stage 7: Coded Transcript + Statistics Reports
    # ------------------------------------------------------------------
    observer.on_stage_start("Report Generation", "7", explanation_key='report_generation')

    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    # Write speaker anonymization key atomically with coded transcripts so both
    # always reflect the same run.  Writing here (not at Stage 1) means a
    # failed run between ingestion and report generation cannot leave a stale
    # key that mismatches older coded transcripts.
    speaker_key = {
        original: {'role': role, 'anonymized_id': anon_id}
        for original, (role, anon_id) in segmenter.speaker_norm.speaker_map.items()
    }
    with open(speaker_key_path, 'w') as _f:
        json.dump(speaker_key, _f, indent=2)
    observer.on_stage_progress(
        "Report Generation",
        f"  Speaker anonymization key: meta/speaker_anonymization_key.json",
    )

    # Build a lightweight LLM client for rationale summarization (reuses theme config)
    _sum_client = None
    if config.run_theme_labeler:
        tc = config.theme_classification
        try:
            _sum_client = LLMClient(LLMClientConfig(
                backend=tc.backend,
                api_key=tc.api_key,
                replicate_api_token=getattr(tc, 'replicate_api_token', ''),
                model=tc.model,
                temperature=0.0,
                lmstudio_base_url=getattr(tc, 'lmstudio_base_url', 'http://127.0.0.1:1234/v1'),
                ollama_host=getattr(tc, 'ollama_host', '0.0.0.0'),
                ollama_port=getattr(tc, 'ollama_port', 11434),
            ))
        except Exception:
            _sum_client = None

    # Per-session coded transcripts
    for session_id, session_df in master_df.groupby('session_id'):
        segs_for_session = [s for s in all_segments if s.session_id == session_id]
        transcript_path = os.path.join(reports_dir, f"coded_transcript_{session_id}.txt")
        export_coded_transcript(
            segs_for_session, framework, codebook, transcript_path,
            llm_client=_sum_client,
        )
        observer.on_stage_progress(
            "Report Generation",
            f"  Coded transcript: reports/coded_transcript_{session_id}.txt",
        )

    # Human classification forms (blind-coding, no results)
    export_human_classification_forms(all_segments, framework, reports_dir)
    observer.on_stage_progress(
        "Report Generation",
        "  Human classification forms: reports/validation/human_classification_<session>.txt",
    )

    # Dataset-wide flagged-for-review report
    validation_dir = os.path.join(reports_dir, 'validation')
    flagged_path = os.path.join(validation_dir, 'flagged_for_review.txt')
    export_flagged_for_review(all_segments, framework, flagged_path)
    observer.on_stage_progress(
        "Report Generation",
        "  Flagged for review: reports/validation/flagged_for_review.txt",
    )

    # Per-transcript stats (one JSON per session)
    stats_dir = os.path.join(reports_dir, 'per_transcript')
    export_per_transcript_stats(master_df, framework, codebook, stats_dir)
    observer.on_stage_progress(
        "Report Generation",
        f"  Per-transcript stats: reports/per_transcript/stats_<session>.json",
    )

    # Cumulative report across all transcripts
    cumulative_path = os.path.join(reports_dir, f'cumulative_report_{ts}.json')
    export_cumulative_report(master_df, framework, codebook, cumulative_path)
    observer.on_stage_progress(
        "Report Generation",
        f"  Cumulative report: reports/cumulative_report_{ts}.json",
    )

    # BERT training data export
    export_training_data(all_segments, framework, codebook, output_dir)
    observer.on_stage_progress(
        "Report Generation",
        "  Training data: trainingdata/theme_classification.jsonl + codebook_multilabel.jsonl",
    )

    observer.on_stage_complete(
        "Report Generation",
        f"Reports written to {reports_dir}",
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

    # ------------------------------------------------------------------
    # Optional: Post-pipeline Results Analysis
    # ------------------------------------------------------------------
    if getattr(config, 'auto_analyze', False):
        try:
            from analysis.runner import run_analysis
            observer.on_stage_start("Results Analysis", "8",
                                    explanation_key='results_analysis')
            analysis_result = run_analysis(output_dir, verbose=False)
            observer.on_stage_complete(
                "Results Analysis",
                f"Analysis complete: {len(analysis_result['files_generated'])} files "
                f"written to reports/analysis/",
            )
        except ImportError:
            pass  # analysis module not available — skip silently
        except Exception as e:
            print(f"\n  Warning: results analysis failed: {e}")
            print(f"  Run manually: python qra.py analyze --output-dir {output_dir}")

    return master_df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_speaker_filter(
    segments: List[Segment], filter_cfg
) -> List[Segment]:
    """
    Return the subset of segments that should be classified, based on
    SpeakerFilterConfig.

    'none'    — return all segments unchanged
    'exclude' — drop therapist segments (by role) from classification.
                Therapist sentences are already filtered before segmentation,
                but this catches any remaining therapist segments.
    """
    mode = getattr(filter_cfg, 'mode', 'none')
    speakers = set(getattr(filter_cfg, 'speakers', []))

    if mode == 'none' or not speakers:
        return segments

    if mode == 'exclude':
        return [
            s for s in segments
            if s.speaker != 'therapist'
        ]

    return segments


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
