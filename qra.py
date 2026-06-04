#!/usr/bin/env python3
"""
qra.py
------
Unified CLI entry point for the Qualitative Research Algorithm (QRA) - a computational phenomenology pipeline.

This script orchestrates the complete workflow for analyzing therapy transcripts from the Move-MORE Feasibility Trial 
using two classification frameworks:
- VAAMR (Vigilance-Avoidance-Attention Regulation-Metacognition-Reappraisal): Classifies participant segments across a five-stage developmental arc
- PURER (Phenomenological-Utilization-Reframing-Educate/Expectancy-Reinforcement): Classifies therapist segments across five guided-inquiry moves
- VCE Phenomenology Codebook: Optional multi-label construct enrichment (59 codes, 7 domains) applied to participant segments

The pipeline has 8 core stages that can be executed as a complete workflow or modularly:
1. Ingestion & Segmentation: Load and segment diarized transcripts into semantic units
2. Construct Operationalization: Build framework definitions from theme_framework/ (VAAMR/PURER)
3-3c. Theme Classification: Zero-shot LLM classification of VAAMR, PURER, and codebook constructs
4-5. Confidence Voting & Consensus: Multi-run model voting with confidence thresholds
6. Assembly: Join frozen segments with overlays into master_segments.jsonl
7. Validation Artifacts: Generate human worksheet, definition key, answer keys for validation
8. Analysis: Generate comprehensive reports on classification results

Command Structure:
  qra <command> [options]

Available Commands:

setup
  Interactive wizard that creates and saves a configuration file (qra_config.json)
  Output: ./data/output/02_meta/qra_config.json

run
  Execute the complete 8-stage pipeline from transcript ingestion to analysis
  Input: Transcript files in transcript-dir
  Output: Complete pipeline output in output-dir with all intermediate and final artifacts
  Generates:
    - 01_transcripts/segmented/ : Frozen segments (JSONL)
    - 02_meta/ : Configuration, logs
    - 03_classification/ : Theme, PURER, codebook classification overlays (JSONL)
    - 04_validation/ : Human worksheet, definition key, answer keys for validation
    - 05_validation/ : Test sets and content validity results
    - 06_reports/ : Analysis reports in HTML, CSV, PDF formats
    - master_segments.jsonl : Final integrated segments with all classifications

analyze
  Run post-hoc analysis on existing pipeline output directory
  Input: Existing output directory containing master_segments.jsonl
  Output: Comprehensive analysis reports in output_dir/06_reports/
    - participant_summary.html/csv : Summary of each participant's theme progression
    - session_summary.html/csv : Summary of each session by construct prevalence
    - longitudinal_analysis.csv : Longitudinal trends across sessions
    - framework_statistics.json : Statistical summaries of classification distributions
    - validation_metrics.json : Accuracy metrics for content validity test sets
    - heatmap_visualizations/ : Visual representations of code/theme frequencies

ingest
  Stage 1: Segment transcripts only (Phase 3, Stage 1)
  Input: Raw diarized transcripts in transcript-dir
  Output: Frozen segments written to output_dir/01_transcripts/segmented/<session_id>/segments.jsonl

classify
  Stages 3-3c: Run classifiers on frozen segments without re-segmenting (Phase 3)
  --what vaamr   : Classify VAAMR constructs only (participant segments)
  --what purer   : Classify PURER constructs only (therapist segments)
  --what codebook: Classify VCE phenomenology codes only
  --what all     : Run all enabled classifiers (default)
  Output: Classification overlays written to output_dir/03_classification/
    - theme_labels.jsonl : VAAMR classifications
    - purer_labels.jsonl : PURER classifications
    - codebook_labels.jsonl : Codebook classifications

assemble
  Stage 6: Join frozen segments with classification overlays into master_segments.jsonl (Phase 3, Stage 6)
  Input: Frozen segments + classification overlays
  Output: output_dir/01_transcripts/master_segments.jsonl (complete integrated dataset)

validate
  Refresh validation artifacts without re-classifying (Stage 7)
  Use after manual edits to human forms or when updating frameworks
  Updates:
    - content_validity_human_worksheet.txt
    - content_validity_definition_key.txt
    - content_validity_answer_key.txt
    - testset manifest files with updated answer keys

testset
  Manage validation test sets (Phase 2)
  create --kind vaamr/purer/codebook --name <name> : Create new frozen test set
  refresh --all/--name <name>                     : Refresh AI answer key for test set(s)
  list                                            : List existing test sets
  Output: output_dir/05_validation/testsets/<name>/
    - manifest.json : Test set metadata and segment IDs
    - human_worksheet.txt : Human-annotated answers (for validation)
    - ai_answer_key.json : AI-generated answer key for automated evaluation

 cv
  Manage content-validity test sets (Phase 2)
  create --framework vaamr/purer --name <name>   : Create new CV test set
  refresh --all/--name <name>                    : Refresh AI answer key using specified model(s)
  list                                          : List existing content-validity test sets
  Output: output_dir/04_validation/cv/<name>/
    - manifest.json : Test set metadata and segment IDs
    - human_worksheet.txt : Human-annotated answers (for validation)
    - ai_answer_key.json : AI-generated answer key for automated evaluation
    - definition_key.txt : Framework definitions used in test set

--test-zeroshot
  Run zero-shot LLM classification against content validity test sets only
  Bypasses full pipeline; skips ingestion and assembly stages
  Input: Content validity test set (created via cv create)
  Output: output_dir/05_validation/content_validity_zeroshot_results.txt
    - Detailed report with per-item results, consensus scores, confidence levels
    - Summary statistics by difficulty level (clear/subtle/adversarial)
    - Per-rater performance metrics

Configuration:
All commands accept configuration via --config <path> or direct CLI arguments.
CLI arguments override config file settings. Configuration is persisted to qra_config.json
via the setup wizard and can be reused across runs.

Framework & Codebook:
- VAAMR: Default theme framework for participant segments
- PURER: Theme framework for therapist segments (requires separate classification)
- VCE Phenomenology Codebook: Optional multi-label construct enrichment system
  
For more information on the neurophenomenological methodology, see methodology.md.
"""

import argparse
import json
import os
import sys
import traceback

# Ensure the package root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =========================================================================
# Shared argument helpers
# =========================================================================

def _add_common_args(parser: argparse.ArgumentParser):
    """Add arguments shared by `run` subcommand."""
    # Config file
    parser.add_argument(
        '--config', '-c',
        default=None,
        help='Path to saved config JSON (from `qra setup`)',
    )

    # Input/output
    parser.add_argument('--transcript-dir', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--trial-id', default=None)

    # Backend & model
    parser.add_argument(
        '--backend',
        default=None,
        choices=['openrouter', 'ollama', 'lmstudio'],
    )
    parser.add_argument(
        '--lmstudio-url', default=None,
        help='LM Studio server base URL (default: http://127.0.0.1:1234/v1)',
    )
    parser.add_argument('--model', default=None)
    parser.add_argument(
        '--models', nargs='+', default=None,
        help='Multiple model IDs for cross-referencing',
    )
    parser.add_argument('--api-key', default=None)

    # Framework & codebook
    parser.add_argument(
        '--framework', default=None,
        help='Theme framework: "vaamr" (default) or path to custom JSON',
    )
    parser.add_argument(
        '--codebook', default=None,
        help='Codebook: "phenomenology" (default) or path to custom JSON',
    )

    # Classification settings
    parser.add_argument('--n-runs', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=None)

    # Confidence thresholds
    parser.add_argument('--high-confidence-threshold', type=float, default=None)
    parser.add_argument('--medium-confidence-threshold', type=float, default=None)

    # Feature flags
    parser.add_argument('--no-theme-labeler', action='store_true')
    parser.add_argument('--run-codebook-classifier', action='store_true')
    parser.add_argument('--no-codebook-classifier', action='store_true')
    parser.add_argument(
        '--verbose-segmentation', action='store_true',
        help='Write process_log.txt with every LLM prompt/response and segmentation step',
    )

    # Embedding classifier settings
    parser.add_argument('--no-two-pass', action='store_true')
    parser.add_argument(
        '--embedding-model', default=None,
        help='Sentence-transformer model for codebook embedding classification '
             '(default: Qwen/Qwen3-Embedding-8B; use all-MiniLM-L6-v2 for lightweight)',
    )
    parser.add_argument('--exemplar-import-path', default=None)
    parser.add_argument('--criteria-weight', type=float, default=None)
    parser.add_argument('--exemplar-weight', type=float, default=None)
    parser.add_argument('--exemplar-confidence-threshold', type=float, default=None)
    parser.add_argument('--max-exemplar-tokens', type=int, default=None)

    # Speaker filtering
    parser.add_argument(
        '--speaker-filter-mode', default=None,
        choices=['none', 'exclude'],
        help='none: classify all | exclude: drop listed speakers',
    )
    parser.add_argument(
        '--exclude-speakers', nargs='+', default=None, metavar='SPEAKER',
        help='Speaker labels to exclude (use with --speaker-filter-mode exclude)',
    )

    # Checkpoint
    parser.add_argument('--resume-from', default=None)

    # Post-pipeline analysis (on by default; use --no-auto-analyze to skip)
    parser.add_argument(
        '--no-auto-analyze',
        action='store_true',
        help='Skip the results analysis stage after the pipeline completes',
    )
    parser.add_argument(
        '--zero-shot',
        action='store_true',
        help='Run classification with zero-shot prompts (no exemplar/subtle/adversarial '
             'utterances in prompt). Per-invocation; not persisted to config.',
    )

    # Zero-shot test mode
    parser.add_argument(
        '--test-zeroshot',
        action='store_true',
        help='Run zero-shot LLM classification against the content validity test set '
             '(skips full pipeline; writes graded report to 05_validation/)',
    )
    parser.add_argument(
        '--preset',
        choices=['small', 'production'],
        default=None,
        help='Use a predefined model ensemble for --test-zeroshot '
             '(small = nemotron-nano/gemma-e2b/qwen3-8b; '
             'production = qwen3-80b/gemma-31b/nemotron-super). '
             'Requires --lmstudio-url.',
    )

    # PHI text anonymization
    parser.add_argument(
        '--no-text-anonymization',
        action='store_true',
        help='Disable PHI name scrubbing from transcript text during ingestion',
    )


# =========================================================================
# Config building
# =========================================================================

def _load_framework(framework_arg):
    """Load a ThemeFramework from preset name or JSON path."""
    if framework_arg is None or framework_arg == 'vaamr':
        from theme_framework.registry import load as _registry_load_fw
        return _registry_load_fw('vaamr')
    if framework_arg == 'vammr':
        raise ValueError(
            'The "vammr" preset is deprecated. Use "vaamr" or a custom framework JSON.'
        )
    # Custom JSON path
    with open(framework_arg) as f:
        fw_data = json.load(f)
    from theme_framework.theme_schema import ThemeFramework, ThemeDefinition
    themes = []
    for t in fw_data.get('themes', []):
        themes.append(ThemeDefinition(
            theme_id=t['theme_id'],
            key=t['key'],
            name=t['name'],
            short_name=t.get('short_name', t['name']),
            prompt_name=t.get('prompt_name', t['name'].lower()),
            definition=t['definition'],
            prototypical_features=t.get('prototypical_features', []),
            distinguishing_criteria=t.get('distinguishing_criteria', ''),
            exemplar_utterances=t.get('exemplar_utterances', []),
        ))
    return ThemeFramework(
        name=fw_data.get('framework', 'custom'),
        version=fw_data.get('version', '1.0'),
        description=fw_data.get('description', ''),
        themes=themes,
    )


def _load_codebook(codebook_arg):
    """Load a Codebook from preset name or JSON path. Returns None if disabled."""
    if codebook_arg is None or codebook_arg == 'phenomenology':
        from codebook.phenomenology_codebook import get_phenomenology_codebook
        return get_phenomenology_codebook()

    with open(codebook_arg) as f:
        cb_data = json.load(f)
    from codebook.codebook_schema import Codebook, CodeDefinition
    codes = []
    for c in cb_data.get('codes', []):
        codes.append(CodeDefinition(
            code_id=c['code_id'],
            category=c['category'],
            domain=c['domain'],
            description=c['description'],
            subcodes=c.get('subcodes', []),
            inclusive_criteria=c.get('inclusive_criteria', ''),
            exclusive_criteria=c.get('exclusive_criteria', ''),
            exemplar_utterances=c.get('exemplar_utterances', []),
        ))
    return Codebook(
        name=cb_data.get('name', 'custom'),
        version=cb_data.get('version', '1.0'),
        description=cb_data.get('description', ''),
        codes=codes,
    )


def _build_config(args):
    """Build PipelineConfig from CLI args, optionally merging a config file."""
    from process.config import PipelineConfig

    # Start from config file if provided, or auto-detect one in the output directory
    if args.config:
        config_path = args.config
    else:
        _od = getattr(args, 'output_dir', None)
        _auto = os.path.join(_od, '02_meta', 'qra_config.json') if _od else None
        config_path = _auto if (_auto and os.path.isfile(_auto)) else None

    if config_path:
        with open(config_path) as f:
            file_data = json.load(f)
        # Use from_json for proper nested reconstruction
        config = PipelineConfig.from_json(_flatten_wizard_config(file_data))
    else:
        config = PipelineConfig()

    # CLI overrides — all accessed via getattr so this function is safe to call
    # from any subparser namespace (not all subparsers define every flag).
    _td = getattr(args, 'transcript_dir', None)
    if _td is not None:
        config.transcript_dir = _td
    _od = getattr(args, 'output_dir', None)
    if _od is not None:
        config.output_dir = _od
    _tid = getattr(args, 'trial_id', None)
    if _tid is not None:
        config.trial_id = _tid
    _rf = getattr(args, 'resume_from', None)
    if _rf is not None:
        config.resume_from = _rf

    # Speaker filter
    sf_mode = getattr(args, 'speaker_filter_mode', None)
    exclude_spk = getattr(args, 'exclude_speakers', None)
    if sf_mode is not None:
        config.speaker_filter.mode = sf_mode
    if exclude_spk is not None:
        config.speaker_filter.mode = 'exclude'
        config.speaker_filter.speakers = exclude_spk

    # Feature flags
    if getattr(args, 'no_theme_labeler', False):
        config.run_theme_labeler = False
    if getattr(args, 'run_codebook_classifier', False):
        config.run_codebook_classifier = True
    if getattr(args, 'no_codebook_classifier', False):
        config.run_codebook_classifier = False
    if getattr(args, 'verbose_segmentation', False):
        config.segmentation.verbose_segmentation = True

    # Backend & model
    tc = config.theme_classification
    _backend = getattr(args, 'backend', None)
    if _backend is not None:
        tc.backend = _backend
    _model = getattr(args, 'model', None)
    if _model is not None:
        tc.model = _model
    lmstudio_url = getattr(args, 'lmstudio_url', None)
    if lmstudio_url is not None:
        tc.lmstudio_base_url = lmstudio_url
    _models = getattr(args, 'models', None)
    if _models is not None:
        tc.models = _models
    _n_runs = getattr(args, 'n_runs', None)
    if _n_runs is not None:
        tc.n_runs = _n_runs
    _temp = getattr(args, 'temperature', None)
    if _temp is not None:
        tc.temperature = _temp

    # API keys: CLI > env > existing config
    _api_key = getattr(args, 'api_key', None)
    if _api_key is not None:
        tc.api_key = _api_key
    elif not tc.api_key:
        tc.api_key = os.environ.get('OPENROUTER_API_KEY', '')

    # Embedding settings
    emb = config.codebook_embedding
    if getattr(args, 'no_two_pass', False):
        emb.two_pass = False
    if getattr(args, 'embedding_model', None) is not None:
        emb.embedding_model = args.embedding_model
    if getattr(args, 'exemplar_import_path', None) is not None:
        emb.exemplar_import_path = args.exemplar_import_path
    if getattr(args, 'criteria_weight', None) is not None:
        emb.criteria_weight = args.criteria_weight
    if getattr(args, 'exemplar_weight', None) is not None:
        emb.exemplar_weight = args.exemplar_weight
    if getattr(args, 'exemplar_confidence_threshold', None) is not None:
        emb.exemplar_confidence_threshold = args.exemplar_confidence_threshold
    if getattr(args, 'max_exemplar_tokens', None) is not None:
        emb.max_exemplar_tokens = args.max_exemplar_tokens

    # Confidence thresholds
    if getattr(args, 'high_confidence_threshold', None) is not None:
        config.confidence_tiers.high_confidence = args.high_confidence_threshold
    if getattr(args, 'medium_confidence_threshold', None) is not None:
        config.confidence_tiers.medium_min_confidence = args.medium_confidence_threshold

    if getattr(args, 'no_auto_analyze', False):
        config.auto_analyze = False

    if getattr(args, 'no_text_anonymization', False):
        config.anonymize_transcript_text = False

    return config


def _flatten_wizard_config(data: dict) -> dict:
    """Flatten wizard-format config (nested pipeline/theme_classification keys)
    into PipelineConfig.from_json-compatible dict."""
    result = {}

    # Lift pipeline-level keys to top level
    pipeline = data.get('pipeline', {})
    for key in ('transcript_dir', 'output_dir', 'trial_id',
                'run_theme_labeler', 'run_codebook_classifier',
                'auto_analyze', 'speaker_anonymization_key_path',
                'anonymize_transcript_text', 'anonymize_text_model',
                'anonymize_text_confidence_threshold'):
        if key in pipeline:
            result[key] = pipeline[key]

    # Pass through sub-config dicts directly
    for key in ('segmentation', 'speaker_filter', 'theme_classification', 'codebook_embedding',
                'codebook_llm', 'codebook_ensemble', 'validation', 'confidence_tiers',
                'test_sets', 'content_validity', 'purer_classification', 'purer_cue',
                'therapist_cues', 'session_summaries', 'participant_summaries'):
        if key in data:
            result[key] = data[key]

    # Copy top-level keys that are already flat
    for key in ('resume_from', 'autoresearch_dir', 'run_purer_labeler',
                'auto_analyze', 'speaker_anonymization_key_path'):
        if key in data and key not in result:
            result[key] = data[key]

    return result


# =========================================================================
# Subcommands
# =========================================================================

def cmd_setup(args):
    """Interactive setup wizard."""
    from process.setup_wizard import SetupWizard, build_config_from_wizard_data

    wizard = SetupWizard()
    result = wizard.run()
    config_path = result['config_path']

    if _prompt_yes_no_simple("Run pipeline now?", True):
        config = build_config_from_wizard_data(result['config_data'])
        framework_spec = result['config_data'].get('framework', {})
        framework = _load_framework(framework_spec.get('custom_path') or framework_spec.get('preset', 'vaamr'))

        codebook = None
        if config.run_codebook_classifier:
            codebook_spec = result['config_data'].get('codebook', {})
            codebook = _load_codebook(codebook_spec.get('custom_path') or codebook_spec.get('preset', 'phenomenology'))

        _execute_pipeline(config, framework, codebook)


def cmd_run(args):
    """Execute pipeline."""
    if getattr(args, 'test_zeroshot', False):
        cmd_test_zeroshot(args)
        return

    config = _build_config(args)

    # --zero-shot: per-invocation override (applies to both VAAMR + PURER for `run`).
    if getattr(args, 'zero_shot', False):
        config.theme_classification.zero_shot_prompt = True
        config.purer_classification.zero_shot_prompt = True

    # Load framework
    framework_arg = args.framework
    if framework_arg is None and args.config:
        with open(args.config) as f:
            file_data = json.load(f)
        fw = file_data.get('framework', {})
        framework_arg = fw.get('custom_path') or fw.get('preset', 'vaamr')
    framework = _load_framework(framework_arg)

    # Load codebook
    codebook = None
    if config.run_codebook_classifier:
        codebook_arg = args.codebook
        if codebook_arg is None and args.config:
            with open(args.config) as f:
                file_data = json.load(f)
            cb = file_data.get('codebook', {})
            codebook_arg = cb.get('custom_path') or cb.get('preset', 'phenomenology')
        codebook = _load_codebook(codebook_arg)

    _execute_pipeline(config, framework, codebook)


def cmd_add_data(args):
    """Incremental data addition: segment + classify only new transcripts.

    Runs the interactive speaker walkthrough, segments only sessions not
    already present in 01_transcripts/segmented/, classifies only those new
    segments using the manifest's recorded config (hard-pinned), merges
    results into existing overlays, re-assembles the master dataset, and
    re-runs analysis. Frozen validation testsets are never mutated.

    Exits non-zero when no new sessions are detected.
    """
    from process.orchestrator import run_incremental_pipeline
    from process import segments_io, classifications_io as _cio
    from process.transcript_ingestion import discover_session_files
    from process.segments_io import resolve_session_id

    config = _build_config(args)

    if getattr(args, 'zero_shot', False):
        config.theme_classification.zero_shot_prompt = True
        config.purer_classification.zero_shot_prompt = True

    # Load framework
    framework_arg = args.framework
    if framework_arg is None and args.config:
        with open(args.config) as f:
            file_data = json.load(f)
        fw = file_data.get('framework', {})
        framework_arg = fw.get('custom_path') or fw.get('preset', 'vaamr')
    framework = _load_framework(framework_arg)

    # Load codebook (only if enabled)
    codebook = None
    if config.run_codebook_classifier:
        codebook_arg = args.codebook
        if codebook_arg is None and args.config:
            with open(args.config) as f:
                file_data = json.load(f)
            cb = file_data.get('codebook', {})
            codebook_arg = cb.get('custom_path') or cb.get('preset', 'phenomenology')
        codebook = _load_codebook(codebook_arg)

    # Sanity check: project must already exist (frozen segments + manifest present).
    existing_sids = segments_io.list_segmented_sessions(config.output_dir)
    manifest = _cio.read_classification_manifest(config.output_dir) or {}
    if not existing_sids:
        print(f"  No existing project at {config.output_dir}. Run `qra run` for the initial pass first.")
        sys.exit(2)
    if not manifest:
        print(f"  No classification manifest at {config.output_dir}/02_meta/classifications/.")
        print("  add-data requires at least one classifier to have run previously.")
        sys.exit(2)

    # Check for new sessions BEFORE invoking the walkthrough — but still proceed
    # to walkthrough=True so the user sees explicit confirmation.
    current_hash = segments_io.params_hash(config.segmentation)
    session_files = discover_session_files(config.transcript_dir)
    new_sids = []
    for sf in session_files:
        sid = resolve_session_id(sf)
        if not segments_io.is_segmentation_fresh(config.output_dir, sid, current_hash):
            new_sids.append(sid)
    if not new_sids:
        print(f"  No new transcripts detected in {config.transcript_dir}.")
        print(f"  Existing project has {len(existing_sids)} segmented session(s).")
        sys.exit(1)

    # LM Studio: wait until reachable
    if config.theme_classification.backend == 'lmstudio':
        _wait_for_lmstudio(
            getattr(config.theme_classification, 'lmstudio_base_url', 'http://127.0.0.1:1234/v1')
        )

    run_incremental_pipeline(config, framework, codebook=codebook, walkthrough=True)


def _resolve_test_rater_config(args) -> dict:
    """
    Build a rater config dict for --test-zeroshot.

    Preset mode uses setup_wizard presets (lmstudio, 3-model ensemble).
    Custom mode uses CLI --backend/--model/--models/--api-key args.
    """
    from process.setup_wizard import _PRESET_SMALL, _PRESET_PRODUCTION

    preset_name = getattr(args, 'preset', None)
    if preset_name in ('small', 'production'):
        preset = _PRESET_SMALL if preset_name == 'small' else _PRESET_PRODUCTION
        lmstudio_url = getattr(args, 'lmstudio_url', None) or 'http://127.0.0.1:1234/v1'
        api_key = getattr(args, 'api_key', None) or os.environ.get('OPENROUTER_API_KEY', '')
        return {
            'backend': 'lmstudio',
            'model': preset['primary_model'],
            'per_run_models': list(preset['per_run_models']),
            'n_runs': preset['n_runs'],
            'temperature': preset['temperature'],
            'api_key': api_key,
            'lmstudio_base_url': lmstudio_url,
            'preset_label': preset['label'],
        }

    # Custom: derive from CLI args
    backend = getattr(args, 'backend', None) or 'openrouter'
    model = getattr(args, 'model', None) or 'meta-llama/Llama-4-Maverick-17B-128E-Instruct'
    models = getattr(args, 'models', None) or []
    n_runs = getattr(args, 'n_runs', None)
    temperature = getattr(args, 'temperature', None) or 0.0
    api_key = getattr(args, 'api_key', None) or os.environ.get('OPENROUTER_API_KEY', '')
    lmstudio_url = getattr(args, 'lmstudio_url', None) or 'http://127.0.0.1:1234/v1'

    per_run_models: list = []
    if models and len(models) >= 2:
        per_run_models = list(models)
        resolved_n_runs = n_runs if n_runs is not None else len(per_run_models)
    else:
        per_run_models = []
        resolved_n_runs = 1

    return {
        'backend': backend,
        'model': model,
        'per_run_models': per_run_models,
        'n_runs': resolved_n_runs,
        'temperature': temperature,
        'api_key': api_key,
        'lmstudio_base_url': lmstudio_url,
        'preset_label': None,
    }


def _cv_items_to_segments(test_items: list):
    """Convert content validity test item dicts to minimal Segment objects."""
    from classification_tools.data_structures import Segment
    segments = []
    for item in test_items:
        seg = Segment(
            segment_id=item['test_item_id'],
            trial_id='cv_test',
            participant_id='cv_participant',
            session_id='cv_test',
            session_number=1,
            cohort_id=None,
            session_variant='',
            segment_index=int(item['test_item_id'].split('_')[1]),
            start_time_ms=0,
            end_time_ms=0,
            total_segments_in_session=len(test_items),
            speaker='participant',
            text=item['text'],
            word_count=len(item['text'].split()),
        )
        segments.append(seg)
    return segments


def _write_zeroshot_report(
    test_items: list,
    results_all: dict,
    rater_ids: list,
    framework,
    output_dir: str,
) -> str:
    """
    Write a human-readable graded report for the zero-shot content validity test.

    Format mirrors 05_validation/ artifacts (78-char separators, same header style).
    Returns the written file path.
    """
    import datetime as _dt
    import textwrap

    from process import output_paths as _paths

    _W = 78
    id_to_name = {t.theme_id: t.short_name for t in framework.themes} if framework else {}

    def _sname(stage_id):
        if stage_id is None:
            return 'ABSTAIN'
        return id_to_name.get(stage_id, str(stage_id))

    def _fconf(c):
        if c is None:
            return '?'
        return f'{float(c):.2f}'

    # ---------------------------------------------------------------------------
    # Pass-type helper: 'primary', 'secondary', or None.
    # A secondary match still counts as correct — if a rater assigned two
    # stages and the expected one is among them, that should not penalise the
    # model.  This mirrors pipeline behaviour where both primary and secondary
    # labels are recorded and considered meaningful.
    # ---------------------------------------------------------------------------
    def _pass_type(primary, secondary, expected):
        if primary is not None and primary == expected:
            return 'primary'
        if secondary is not None and secondary == expected:
            return 'secondary'
        return None

    # Build per-item result rows
    rows = []
    for item in test_items:
        iid = item['test_item_id']
        expected = item['expected_stage']
        difficulty = item['difficulty']
        text = item['text']

        raw = results_all.get(iid, {})
        rater_votes = raw.get('rater_votes', [])
        consensus = raw.get('consensus', {})

        # vote_single_label: top-level keys are 'primary_stage', 'secondary_stage',
        # 'consensus_vote' (not 'stage'/'vote')
        cons_primary = consensus.get('primary_stage')
        cons_secondary = consensus.get('secondary_stage')
        cons_cv = consensus.get('consensus_vote')  # int | 'ABSTAIN' | None (split)
        cons_pt = _pass_type(cons_primary, cons_secondary, expected)

        # Per-rater votes indexed by rater id
        votes_by_rater = {rv.get('rater', ''): rv for rv in rater_votes}

        rows.append({
            'item_id': iid,
            'expected': expected,
            'difficulty': difficulty,
            'text': text,
            'rater_votes': votes_by_rater,
            'cons_primary': cons_primary,
            'cons_secondary': cons_secondary,
            'cons_cv': cons_cv,
            'cons_pt': cons_pt,           # 'primary' | 'secondary' | None
            'cons_pass': cons_pt is not None,
        })

    # ---------------------------------------------------------------------------
    # Score helpers: count passes (primary OR secondary) and secondary-only passes
    # ---------------------------------------------------------------------------
    tiers = ['clear', 'subtle', 'adversarial']

    def _score(rows_subset, rater=None):
        """Returns (n_pass, n_secondary_only, n_total)."""
        n_pass = 0
        n_sec = 0
        total = len(rows_subset)
        for r in rows_subset:
            if rater is not None:
                rv = r['rater_votes'].get(rater, {})
                vote = rv.get('vote', 'ERROR')
                pt = _pass_type(
                    rv.get('stage') if vote == 'CODED' else None,
                    rv.get('secondary_stage') if vote == 'CODED' else None,
                    r['expected'],
                )
            else:
                pt = r['cons_pt']
            if pt == 'primary':
                n_pass += 1
            elif pt == 'secondary':
                n_pass += 1
                n_sec += 1
        return n_pass, n_sec, total

    def _cell(n_pass, n_sec, total):
        if total == 0:
            return 'N/A'
        pct = 100 * n_pass / total
        base = f'{n_pass}/{total} ({pct:.0f}%)'
        return base + (f' ~{n_sec}' if n_sec else '')

    # Pad rater ids to consistent display width
    max_rater_len = max((len(r) for r in rater_ids), default=10)
    rater_col_w = max(max_rater_len, len('CONSENSUS'))

    vdir = _paths.validation_dir(output_dir)
    os.makedirs(vdir, exist_ok=True)
    out_path = os.path.join(vdir, 'content_validity_zeroshot_results.txt')

    with open(out_path, 'w', encoding='utf-8') as fh:
        # ---- Header ----
        fh.write('=' * _W + '\n')
        fh.write('CONTENT VALIDITY ZERO-SHOT CLASSIFICATION RESULTS\n')
        fh.write('=' * _W + '\n')
        fh.write(f'Framework: {framework.name}   Version: {framework.version}   '
                 f'Themes: {len(framework.themes)}\n')
        fh.write(f'Generated: {_dt.date.today().isoformat()}\n')
        fh.write('=' * _W + '\n\n')

        # ---- Rater lineup ----
        fh.write('RATER LINEUP\n')
        fh.write('-' * _W + '\n')
        if len(rater_ids) == 1:
            fh.write(f'  Single rater: {rater_ids[0]}\n')
        else:
            for i, rid in enumerate(rater_ids, start=1):
                fh.write(f'  Run {i}: {rid}\n')
        fh.write('\n')

        # ---- Score summary table ----
        fh.write('SCORE SUMMARY\n')
        fh.write('-' * _W + '\n')
        fh.write(
            '  Scoring: [PASS] = expected stage is primary label; '
            '[PASS~] = expected stage\n'
            '  is secondary label (both count as correct). '
            '~N in a cell = secondary-match count.\n\n'
        )
        header_cols = ['Overall'] + tiers
        col_w = 16
        header_line = f'  {"Rater":{rater_col_w + 2}}'
        for h in header_cols:
            header_line += f'  {h:<{col_w}}'
        fh.write(header_line.rstrip() + '\n')
        fh.write('  ' + '-' * (rater_col_w + 2 + (col_w + 2) * len(header_cols)) + '\n')

        all_raters_for_summary = rater_ids + ['CONSENSUS']
        for rater in all_raters_for_summary:
            is_cons = (rater == 'CONSENSUS')
            label = rater if not is_cons else 'CONSENSUS'
            n_p, n_s, t = _score(rows, None if is_cons else rater)
            row_line = f'  {label:{rater_col_w + 2}}  {_cell(n_p, n_s, t):<{col_w}}'
            for tier in tiers:
                tier_rows = [r for r in rows if r['difficulty'] == tier]
                n_p2, n_s2, t2 = _score(tier_rows, None if is_cons else rater)
                row_line += f'  {_cell(n_p2, n_s2, t2):<{col_w}}'
            fh.write(row_line.rstrip() + '\n')

        fh.write('\n')
        fh.write('  By stage (consensus):\n')
        for theme in sorted(framework.themes, key=lambda t: t.theme_id):
            stage_rows = [r for r in rows if r['expected'] == theme.theme_id]
            n_p, n_s, t = _score(stage_rows, None)
            pct = f'{100*n_p/t:.0f}%' if t else '?'
            sec_note = f'  (~{n_s} secondary)' if n_s else ''
            fh.write(
                f'    Stage {theme.theme_id} {theme.short_name:<16} '
                f'{n_p}/{t} ({pct}){sec_note}\n'
            )
        fh.write('\n')

        # ---- Item-by-item ----
        fh.write('=' * _W + '\n')
        fh.write('ITEM-BY-ITEM RESULTS\n')
        fh.write('=' * _W + '\n\n')

        for row in rows:
            exp_name = _sname(row['expected'])
            if row['cons_pt'] == 'primary':
                cons_label = '[PASS]'
            elif row['cons_pt'] == 'secondary':
                cons_label = '[PASS~]'
            else:
                cons_label = '[FAIL]'

            fh.write('=' * _W + '\n')
            fh.write(
                f"[{row['item_id']}]  Tier: {row['difficulty']:<12}  "
                f"Expected: {exp_name}   Consensus: {cons_label}\n"
            )
            fh.write('-' * _W + '\n')
            for line in textwrap.wrap(
                f'"{row["text"]}"', width=_W - 2,
                initial_indent='  ', subsequent_indent='  ',
            ) or ['  ']:
                fh.write(line + '\n')
            fh.write('\n')

            for rid in rater_ids:
                rv = row['rater_votes'].get(rid, {})
                vote = rv.get('vote', 'ERROR')
                primary = rv.get('stage') if vote == 'CODED' else None
                secondary = rv.get('secondary_stage') if vote == 'CODED' else None
                conf = rv.get('confidence')
                sec_conf = rv.get('secondary_confidence')
                just = (rv.get('justification') or '').strip()

                pt = _pass_type(primary, secondary, row['expected'])
                result_tag = '[PASS]' if pt == 'primary' else ('[PASS~]' if pt == 'secondary' else '[FAIL]')

                if vote == 'CODED':
                    stage_str = _sname(primary)
                    if secondary is not None:
                        stage_str += f' / {_sname(secondary)}'
                    conf_str = f'  conf={_fconf(conf)}'
                    if secondary is not None and sec_conf is not None:
                        conf_str += f'/{_fconf(sec_conf)}'
                else:
                    stage_str = vote
                    conf_str = ''

                label_col = f'[{rid}]'
                fh.write(
                    f'  {label_col:{rater_col_w + 2}}  '
                    f'{stage_str:<22}{conf_str}  {result_tag}\n'
                )
                # Show justification when this item has any failure (rater or consensus)
                if just and (pt != 'primary' or not row['cons_pass']):
                    for line in textwrap.wrap(
                        just, width=_W - 8,
                        initial_indent='      → ', subsequent_indent='        ',
                    ):
                        fh.write(line + '\n')

            # Consensus line with primary + secondary
            if row['cons_primary'] is not None:
                cons_stage_str = _sname(row['cons_primary'])
                if row['cons_secondary'] is not None:
                    cons_stage_str += f' / {_sname(row["cons_secondary"])}'
            elif row['cons_cv'] == 'ABSTAIN':
                cons_stage_str = 'ABSTAIN'
            elif row['cons_cv'] is None:
                cons_stage_str = 'SPLIT'
            else:
                cons_stage_str = 'ERROR'

            fh.write(
                f'  {"CONSENSUS":{rater_col_w + 2}}  '
                f'{cons_stage_str:<22}              {cons_label}\n'
            )
            fh.write('\n')

    return out_path


def cmd_test_zeroshot(args):
    """Run zero-shot LLM classification against the content validity test set."""
    from classification_tools.llm_classifier import (
        create_content_validity_test_set,
        classify_segments_zero_shot,
    )
    from theme_framework.config import ThemeClassificationConfig

    framework_arg = getattr(args, 'framework', None)
    framework = _load_framework(framework_arg)

    rater_cfg = _resolve_test_rater_config(args)

    output_dir = getattr(args, 'output_dir', None) or './data/zeroshot_test'
    os.makedirs(output_dir, exist_ok=True)

    print('\n' + '=' * 70)
    print('QRA CONTENT VALIDITY ZERO-SHOT TEST')
    print('=' * 70)
    print(f'  Framework : {framework.name} v{framework.version} ({len(framework.themes)} themes)')
    if rater_cfg.get('preset_label'):
        print(f'  Preset    : {rater_cfg["preset_label"]}')
    if rater_cfg['per_run_models']:
        print(f'  Raters    : {len(rater_cfg["per_run_models"])} models')
        for i, m in enumerate(rater_cfg['per_run_models'], start=1):
            print(f'    Run {i}: {m}')
    else:
        print(f'  Model     : {rater_cfg["model"]}')
    print(f'  Backend   : {rater_cfg["backend"]}')
    print()

    # Build test items and fake segments
    test_items = create_content_validity_test_set(framework)
    print(f'  Test items: {len(test_items)} '
          f'({sum(1 for i in test_items if i["difficulty"]=="clear")} clear, '
          f'{sum(1 for i in test_items if i["difficulty"]=="subtle")} subtle, '
          f'{sum(1 for i in test_items if i["difficulty"]=="adversarial")} adversarial)')
    print()

    segments = _cv_items_to_segments(test_items)

    # Write static companion documents before classification begins
    from process.assembly import (
        export_content_validity_human_worksheet,
        export_content_validity_definition_key,
        export_content_validity_answer_key,
    )
    export_content_validity_human_worksheet(test_items, framework, output_dir)
    export_content_validity_definition_key(framework, output_dir)
    export_content_validity_answer_key(test_items, framework, output_dir)
    print('  Companion documents written to 04_validation/:')
    print('    content_validity_human_worksheet.txt')
    print('    content_validity_definition_key.txt')
    print('    content_validity_answer_key.txt')
    print()

    # Build classification config
    checkpoint_dir = os.path.join(output_dir, '_cv_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    tc = ThemeClassificationConfig(
        model=rater_cfg['model'],
        per_run_models=rater_cfg['per_run_models'],
        n_runs=rater_cfg['n_runs'],
        temperature=rater_cfg['temperature'],
        backend=rater_cfg['backend'],
        api_key=rater_cfg['api_key'],
        lmstudio_base_url=rater_cfg['lmstudio_base_url'],
        output_dir=checkpoint_dir,
        context_window_segments=0,
        min_classifiable_words=0,
        randomize_codebook=False,
    )

    # LM Studio: wait for server before starting
    if tc.backend == 'lmstudio':
        _wait_for_lmstudio(tc.lmstudio_base_url)

    results_all, _ = classify_segments_zero_shot(segments, framework, tc)

    # Determine rater IDs that appear in results
    seen_raters = []
    if rater_cfg['per_run_models']:
        seen_raters = list(rater_cfg['per_run_models'])
    else:
        seen_raters = [rater_cfg['model']]

    report_path = _write_zeroshot_report(
        test_items=test_items,
        results_all=results_all,
        rater_ids=seen_raters,
        framework=framework,
        output_dir=output_dir,
    )

    # Quick summary to stdout
    from process import output_paths as _paths
    n_pass = n_secondary = 0
    for item in test_items:
        raw = results_all.get(item['test_item_id'], {})
        cons = raw.get('consensus', {})
        exp = item['expected_stage']
        primary = cons.get('primary_stage')
        secondary = cons.get('secondary_stage')
        if primary == exp:
            n_pass += 1
        elif secondary is not None and secondary == exp:
            n_pass += 1
            n_secondary += 1
    n_total = len(test_items)
    pct = 100 * n_pass / n_total if n_total else 0
    sec_note = f'  ({n_secondary} as secondary label)' if n_secondary else ''
    print(f'\n  Consensus score: {n_pass}/{n_total} ({pct:.0f}%){sec_note}')
    print(f'  Report: {os.path.relpath(report_path)}')


def cmd_analyze(args):
    """Run post-hoc results analysis on an existing pipeline output directory."""
    from analysis.runner import run_analysis

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        print(f"Error: output directory not found: {output_dir}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("QRA RESULTS ANALYSIS")
    print("=" * 70)
    print(f"  Output dir: {output_dir}")
    print()

    # --gnn / --no-gnn override the config's gnn_layer.enabled for this run.
    force_gnn = None
    if getattr(args, 'gnn', False):
        force_gnn = True
    elif getattr(args, 'no_gnn', False):
        force_gnn = False

    result = run_analysis(output_dir, verbose=True, force_gnn=force_gnn)

    print(f"\nAnalysis complete.")
    print(f"  {result['n_segments']} segments | "
          f"{result['n_participants']} participants | "
          f"{result['n_sessions']} sessions")
    print(f"  Reports: {output_dir}/06_reports/ and {output_dir}/03_analysis_data/")
    print(f"  Files generated: {len(result['files_generated'])}")


# =========================================================================
# Phase 3 stage subcommands: ingest, classify, assemble
# =========================================================================

def cmd_ingest(args):
    """qra ingest — segment transcripts and freeze them to disk."""
    from process.orchestrator import stage_ingest, SilentObserver

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    config = _build_config(args)
    config.output_dir = output_dir

    force_reingest = getattr(args, 'reingest', None)
    force_reingest_all = getattr(args, 'reingest_all', False)

    print(f"\nQRA INGEST")
    print(f"  Input:  {config.transcript_dir}")
    print(f"  Output: {output_dir}")
    if force_reingest:
        print(f"  Force re-ingest: {force_reingest}")
    elif force_reingest_all:
        print(f"  Force re-ingest: ALL sessions")

    segments = stage_ingest(
        config,
        output_dir=output_dir,
        observer=SilentObserver(),
        force_reingest=force_reingest,
        force_reingest_all=force_reingest_all,
    )
    n_sess = len(set(s.session_id for s in segments))
    print(f"\nDone. {len(segments)} segments across {n_sess} sessions.")
    print("Run `qra classify` to classify, then `qra assemble` to build master_segments.")


def cmd_apply_anonymization(args):
    """qra apply-anonymization — retroactively scrub PHI names from frozen segment text."""
    from process import segments_io as _sio
    from process.speaker_anonymization import load_speaker_map as _load_speaker_map
    from process.text_anonymization import scrub_segments as _scrub
    from process import output_paths as _paths

    output_dir = args.output_dir
    sessions = _sio.list_segmented_sessions(output_dir)
    if not sessions:
        print(f"No frozen segments found in {output_dir}.")
        print("Run `qra ingest` first to create segmented transcripts.")
        sys.exit(1)

    sessions_to_process = [args.session] if args.session else sessions
    if args.session and args.session not in sessions:
        print(f"Session {args.session!r} not found. Available: {sessions}")
        sys.exit(1)

    # Load speaker map — prefer explicit key path, fall back to project key
    meta_dir = _paths.meta_dir(output_dir)
    config = _build_config(args)
    speaker_map, _ = _load_speaker_map(meta_dir, config)

    if not speaker_map and not getattr(args, 'force', False):
        print(
            f"Warning: no speaker_anonymization_key.json found in {meta_dir}.\n"
            "  Known names cannot be mapped to anonymized IDs.\n"
            "  Unknown names will still be replaced with (NAME).\n"
            "  Use --force to proceed anyway, or provide --config pointing to a key."
        )
        sys.exit(1)

    use_transformer = not getattr(args, 'no_spacy', False)
    model_name = getattr(args, 'model', None) or 'obi/deid_roberta_i2b2'
    confidence = getattr(args, 'confidence', None) or 0.6

    print(f"\nQRA APPLY-ANONYMIZATION")
    print(f"  Output dir  : {output_dir}")
    print(f"  Sessions    : {len(sessions_to_process)}")
    print(f"  Known names : {len(speaker_map)} entries in anonymization key")
    if use_transformer:
        print(f"  NLP engine  : Presidio (spaCy + optional transformer de-id)")
        print(f"  Model       : {model_name}")
        print(f"  Confidence  : {confidence}")
    else:
        print("  Name detect : regex heuristics only (--no-spacy)")

    if not getattr(args, 'yes', False):
        print()
        confirm = input(
            f"  This will OVERWRITE segments.jsonl for {len(sessions_to_process)} session(s).\n"
            "  segmentation_meta.json will NOT be modified.\n"
            "  Type 'yes' to continue: "
        ).strip().lower()
        if confirm != 'yes':
            print("Aborted.")
            sys.exit(0)

    print()
    total_k = total_u = total_m = 0
    for sid in sessions_to_process:
        segs = _sio.read_session_segments(output_dir, sid)
        scrubbed, stats = _scrub(
            segs, speaker_map,
            use_transformer=use_transformer,
            confidence_threshold=confidence,
            model_name=model_name,
        )
        _sio.overwrite_segment_texts(output_dir, sid, scrubbed)
        total_k += stats['n_known']
        total_u += stats['n_unknown']
        total_m += stats['n_segments_modified']
        backend = stats.get('engine_backend', '?')
        print(
            f"  {sid}: {stats['n_known']} known + {stats['n_unknown']} unknown "
            f"replacements in {stats['n_segments_modified']} segments [{backend}]"
        )

    print(
        f"\nDone. {total_k} known-name + {total_u} unknown-name replacements "
        f"across {total_m} modified segments."
    )
    print("Run `qra assemble` to rebuild master_segments with the updated text.")


def cmd_classify(args):
    """qra classify — run classifier(s) on frozen segments and write overlays."""
    from process import segments_io as _segments_io
    from process.orchestrator import (
        stage_classify_theme,
        stage_classify_purer,
        stage_classify_codebook,
        stage_classify_microskill,
        stage_cross_validation,
    )
    from process._freeze import FrozenArtifactError

    output_dir = args.output_dir
    what = getattr(args, 'what', 'all') or 'all'
    valid = {'vaamr', 'purer', 'codebook', 'microskill', 'cross-validation', 'all'}
    if what not in valid:
        print(f"Error: --what must be one of {sorted(valid)}, got {what!r}")
        sys.exit(2)

    # Guard: frozen segments must exist
    sessions = _segments_io.list_segmented_sessions(output_dir)
    if not sessions:
        print(
            f"Error: no frozen segments found in {output_dir}.\n"
            "  Run `qra ingest` (or `qra run`) first."
        )
        sys.exit(1)

    config = _build_config(args)
    framework = _load_framework(getattr(args, 'framework', None))

    # --zero-shot: per-invocation override scoped by --what.
    if getattr(args, 'zero_shot', False):
        if what in ('vaamr', 'all'):
            config.theme_classification.zero_shot_prompt = True
        if what in ('purer', 'all'):
            config.purer_classification.zero_shot_prompt = True
        print("  Zero-shot prompting: ON (no exemplars in prompt)")

    print(f"\nQRA CLASSIFY  --what {what}")
    print(f"  Output: {output_dir}")
    print(f"  Sessions: {len(sessions)}")

    to_run = {what} if what != 'all' else {'vaamr', 'purer', 'codebook', 'microskill', 'cross-validation'}

    # Load frozen segments once (raw); apply overlays selectively per stage below.
    from process import classifications_io as _cio
    segments = _segments_io.load_segments_for_stage(output_dir, apply=())
    # Apply all existing overlays up-front so each stage sees the current on-disk state.
    by_id = {s.segment_id: s for s in segments}
    _cio.apply_overlays(output_dir, by_id, keys=('theme', 'purer', 'codebook', 'microskill', 'cv'))

    if 'vaamr' in to_run:
        print("  Running VAAMR classifier...")
        stage_classify_theme(config, framework, segments=segments, output_dir=output_dir)
        print(f"  theme_labels.jsonl written ({len(segments)} segments)")

    if 'purer' in to_run:
        print("  Running PURER classifier...")
        stage_classify_purer(config, segments=segments, output_dir=output_dir)
        print(f"  purer_labels.jsonl written")

    if 'codebook' in to_run:
        codebook = _load_codebook(getattr(args, 'codebook', None))
        print("  Running codebook classifier...")
        stage_classify_codebook(config, codebook, segments=segments, output_dir=output_dir)
        print(f"  codebook_labels.jsonl written")

    if 'microskill' in to_run:
        from codebook.microcounseling_codebook import get_microcounseling_codebook
        micro_cb = get_microcounseling_codebook()
        print("  Running microcounseling-skill classifier...")
        stage_classify_microskill(config, micro_cb, segments=segments, output_dir=output_dir)
        print(f"  microskill_labels.jsonl written")

    if 'cross-validation' in to_run:
        print("  Running cross-validation...")
        stage_cross_validation(config, framework, segments=segments, output_dir=output_dir)
        print(f"  cross_validation_labels.jsonl written")

    print("\nClassification overlay(s) written.")

    if not getattr(args, 'no_downstream', False):
        print("\nRunning downstream pipeline...")
        cmd_assemble(args)
        cmd_testset_refresh(args)
        cmd_analyze(args)


def cmd_reclassify_run(args):
    """qra reclassify-run — redo a single classification run without re-processing the others."""
    import glob
    from classification_tools.classification_loop import patch_runs_checkpoint
    from process.orchestrator import (
        stage_classify_theme,
        stage_assemble,
        stage_validation_artifacts,
    )
    from analysis.runner import run_analysis

    output_dir = args.output_dir
    run_number = args.run          # 1-indexed (user-facing)
    new_model = getattr(args, 'model', None)
    explicit_checkpoint = getattr(args, 'checkpoint', None)

    # -- Load config -------------------------------------------------------
    config_path = getattr(args, 'config', None)
    if not config_path:
        candidate = os.path.join(output_dir, '02_meta', 'qra_config.json')
        if os.path.isfile(candidate):
            config_path = candidate

    if config_path:
        with open(config_path) as f:
            file_data = json.load(f)
        config = _build_config_from_file(file_data)
    else:
        from process.config import PipelineConfig
        config = PipelineConfig()

    tc = config.theme_classification
    n_runs = tc.n_runs
    run_idx = run_number - 1

    if not (1 <= run_number <= n_runs):
        print(f"Error: --run must be between 1 and {n_runs} (got {run_number}).")
        sys.exit(2)

    # -- Locate checkpoint -------------------------------------------------
    if explicit_checkpoint:
        checkpoint_path = explicit_checkpoint
    else:
        checkpoints_dir = os.path.join(
            output_dir, '02_meta', 'auditable_logs', 'checkpoints'
        )
        glob_pattern = os.path.join(checkpoints_dir, 'llm_results_*_runs.json')
        candidates = sorted(glob.glob(glob_pattern))
        if not candidates:
            print(
                f"Error: no *_runs.json checkpoint found in {checkpoints_dir}.\n"
                "  Pass --checkpoint <path> to specify one explicitly."
            )
            sys.exit(1)
        checkpoint_path = candidates[-1]  # latest by filename timestamp

    print(f"\nQRA RECLASSIFY-RUN")
    print(f"  Output dir   : {output_dir}")
    print(f"  Checkpoint   : {os.path.basename(checkpoint_path)}")
    print(f"  Run          : {run_number}/{n_runs} (index {run_idx})")
    if new_model:
        print(f"  New model    : {new_model}")
    else:
        current_model = (tc.per_run_models[run_idx]
                         if run_idx < len(tc.per_run_models) else tc.model)
        print(f"  Model        : {current_model} (unchanged)")

    # -- Patch checkpoint --------------------------------------------------
    updated_per_run_models = patch_runs_checkpoint(
        checkpoint_path, run_idx, new_model=new_model
    )

    # Apply model update to config so classify stage uses the right model
    if new_model and run_idx < len(tc.per_run_models):
        tc.per_run_models[run_idx] = new_model

    # Tell the classifier to resume from this checkpoint (so other runs are skipped)
    config.resume_from = checkpoint_path

    # -- Re-run theme classification (updates consensus + theme_labels.jsonl) --
    framework = _load_framework(
        getattr(config, 'participant_framework', None) or 'vaamr'
    )
    print(f"\nStep 1/5 — Theme classification (run {run_number} only)...")
    stage_classify_theme(config, framework, output_dir=output_dir)
    print(f"  theme_labels.jsonl updated.")

    # -- Re-assemble master dataset ----------------------------------------
    print("\nStep 2/5 — Assembling master dataset...")
    stage_assemble(config, output_dir=output_dir)
    print(f"  master_segments.jsonl updated.")

    # -- Refresh testset AI answer keys + CV testsets ----------------------
    # create_missing=False so we only refresh existing artifacts, not create new ones
    print("\nStep 3/5 — Refreshing testset + CV answer keys...")
    try:
        stage_validation_artifacts(
            config, framework,
            output_dir=output_dir,
            create_missing=False,
        )
        print(f"  Testset and CV answer keys refreshed.")
    except Exception as e:
        print(f"  Warning: validation artifact refresh failed: {e}")

    # -- Re-run analysis (figures, reports) --------------------------------
    print("\nStep 4/5 — Re-running analysis and figures...")
    try:
        result = run_analysis(output_dir, verbose=False)
        print(
            f"  Analysis complete: {result['n_segments']} segments | "
            f"{result['n_participants']} participants | "
            f"{result['n_sessions']} sessions"
        )
        print(f"  Reports: {output_dir}/06_reports/")
    except Exception as e:
        print(f"  Warning: analysis failed: {e}")

    # -- Done --------------------------------------------------------------
    print(f"\nStep 5/5 — Complete.")
    print(f"  Run {run_number} re-classified with: "
          f"{updated_per_run_models[run_idx] if run_idx < len(updated_per_run_models) else new_model or 'unchanged model'}")
    print(f"  per_run_models: {updated_per_run_models}")


def _build_config_from_file(file_data: dict):
    """Load a PipelineConfig from a raw JSON dict (wizard or flat format)."""
    from process.config import PipelineConfig
    return PipelineConfig.from_json(_flatten_wizard_config(file_data))


def cmd_assemble(args):
    """qra assemble — join frozen segments + overlays into master_segments."""
    from process import segments_io as _segments_io, output_paths as _paths
    from process.orchestrator import stage_assemble
    from process import classifications_io as _cio

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Guard: at least one overlay must exist
    has_overlay = any(
        os.path.isfile(_cio.overlay_path(output_dir, key))
        for key in ('theme', 'purer', 'codebook', 'cv')
    )
    has_frozen = bool(_segments_io.list_segmented_sessions(output_dir))

    if not has_frozen and not has_overlay:
        print(
            "Error: no frozen segments or classification overlays found.\n"
            "  Run `qra ingest` then `qra classify` first."
        )
        sys.exit(1)

    if not has_overlay:
        print(
            "Error: no classification overlays found in {output_dir}.\n"
            "  Run `qra classify` first."
        )
        sys.exit(1)

    config = _build_config(args)

    print(f"\nQRA ASSEMBLE")
    print(f"  Output: {output_dir}")

    master_df = stage_assemble(config, output_dir=output_dir)

    ms_dir = _paths.master_segments_dir(output_dir)
    print(f"  master_segments.jsonl: {len(master_df)} segments")
    print(f"  Written to: {ms_dir}")

    # Refresh human forms, testsets, and CV answer keys (no new testset creation).
    from process.orchestrator import stage_validation_artifacts
    framework = _load_framework(getattr(args, 'framework', None))
    codebook = None
    try:
        codebook = _load_codebook(getattr(args, 'codebook', None))
    except Exception:
        pass
    stage_validation_artifacts(
        config, framework, codebook,
        output_dir=output_dir,
        create_missing=False,
    )
    print("Done. Run `qra analyze` for post-hoc analysis.")


def cmd_validate(args):
    """qra validate — refresh validation artifacts (human forms, testsets, CV) without re-classifying."""
    from process.orchestrator import stage_validation_artifacts
    from process import segments_io as _segments_io

    output_dir = args.output_dir
    sessions = _segments_io.list_segmented_sessions(output_dir)
    if not sessions:
        print(f"Error: no frozen segments in {output_dir}. Run `qra ingest` first.")
        sys.exit(1)

    config = _build_config(args)
    framework = _load_framework(getattr(args, 'framework', None))
    codebook = None
    try:
        codebook = _load_codebook(getattr(args, 'codebook', None))
    except Exception:
        pass

    print(f"\nQRA VALIDATE")
    print(f"  Output: {output_dir}")

    stage_validation_artifacts(
        config, framework, codebook,
        output_dir=output_dir,
        create_missing=False,
    )
    print("Done. Human forms, flagged-for-review, and testset/CV answer keys refreshed.")


def cmd_testsets(args):
    """Generate or refresh validation test set worksheets from existing pipeline output."""
    from process.assembly import generate_or_refresh_validation_testsets
    from process import segments_io as _segments_io

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        print(f"Error: output directory not found: {output_dir}")
        sys.exit(1)

    print(f"\nLoading segments from: {output_dir}")
    try:
        segments = _segments_io.read_master_segments(output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    participant_segs = [s for s in segments if s.speaker == 'participant']
    print(f"Loaded {len(segments)} segments ({len(participant_segs)} participant)")

    # Load framework
    framework_arg = args.framework
    if framework_arg is None and args.config:
        with open(args.config) as f:
            fw_data = json.load(f)
        fw = fw_data.get('framework', {})
        framework_arg = fw.get('custom_path') or fw.get('preset', 'vaamr')
    framework = _load_framework(framework_arg)

    # Determine codebook status
    codebook_enabled = args.codebook_enabled
    if not codebook_enabled and args.config:
        with open(args.config) as f:
            cfg_data = json.load(f)
        codebook_enabled = cfg_data.get('pipeline', {}).get('run_codebook_classifier', False)

    testset_dirs = generate_or_refresh_validation_testsets(
        segments,
        framework,
        output_dir,
        n_sets=args.n_sets,
        fraction_per_set=args.fraction,
        random_seed=args.seed,
        codebook_enabled=codebook_enabled,
    )

    print(f"\nGenerated/refreshed {len(testset_dirs)} testset(s):")
    for p in testset_dirs:
        print(f"  {os.path.relpath(p, output_dir)}")


# =========================================================================
# testset subcommand group (Phase 2)
# =========================================================================

def _load_segments_from_output(output_dir: str):
    """Load master segments from an existing pipeline output directory."""
    from process import segments_io as _segments_io
    if not os.path.isdir(output_dir):
        print(f"Error: output directory not found: {output_dir}")
        sys.exit(1)
    try:
        return _segments_io.read_master_segments(output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_testset_create(args):
    """qra testset create — create a flat numbered testset worksheet."""
    import re as _re
    from process.assembly.human_forms import (
        create_frozen_testset, refresh_testset_answer_key,
        _pool_purer, _pool_codebook, _collect_used_segment_ids,
        _worksheet_is_legacy_import, _sync_worksheet_header,
    )
    from process import output_paths as _paths

    segments = _load_segments_from_output(args.output_dir)
    segments_by_id = {s.segment_id: s for s in segments}
    framework = _load_framework(args.framework)

    kind = args.kind
    # Validate pool is non-empty for kind
    if kind == 'purer':
        pool = _pool_purer(segments)
        if not pool:
            print("Error: no therapist segments with purer_primary labels found. "
                  "Run qra run with PURER classifier enabled first.")
            sys.exit(2)
    elif kind == 'codebook':
        pool = _pool_codebook(segments)
        if not pool:
            print("Error: no participant segments with codebook_labels_ensemble found. "
                  "Run qra run with codebook classifier enabled first.")
            sys.exit(2)

    # Collect segments already in existing testsets to prevent overlap
    used_ids = _collect_used_segment_ids(args.output_dir)

    human_path = create_frozen_testset(
        segments, framework, args.output_dir,
        kind=kind, n_sets=1, set_index=1,
        fraction_per_set=args.fraction, random_seed=args.seed,
        codebook_enabled=(kind == 'codebook'),
        exclude_segment_ids=used_ids,
    )
    print(f"Created testset: {os.path.relpath(human_path, args.output_dir)}")

    # Refresh all testsets (including newly created) to sync 'N of M' headers and AI keys
    ts_dir = _paths.testsets_dir(args.output_dir)
    pattern = _re.compile(r'^human_classification_testset_worksheet_(\d+)\.txt$')
    all_nums = sorted(int(m.group(1)) for f in os.listdir(ts_dir) if (m := pattern.match(f)))
    n_total = len(all_nums)
    new_n = int(os.path.basename(human_path).rsplit('_', 1)[-1].split('.')[0])
    force = getattr(args, 'force', False) or getattr(args, 'yes', False)
    for num in all_nums:
        # Imported legacy testsets keep their human-validated content & AI classifications
        # frozen; only the 'N of M' header is synced (both files) so the set reads
        # consistently. Regenerating a validated legacy AI key is the explicit job of
        # `testset refresh`, not a side effect of creating a new testset.
        if _worksheet_is_legacy_import(_paths.testset_meta_path(args.output_dir, num)):
            _sync_worksheet_header(_paths.testset_human_flat_path(args.output_dir, num), num, n_total)
            _sync_worksheet_header(_paths.testset_ai_flat_path(args.output_dir, num), num, n_total)
            continue
        try:
            ai_path = refresh_testset_answer_key(
                segments_by_id, framework, args.output_dir, num,
                codebook_enabled=(kind == 'codebook'),
                n_total=n_total,
                force=force,
            )
            if num != new_n:
                print(f"Updated header: {os.path.relpath(ai_path, args.output_dir)}")
        except Exception as e:
            print(f"Warning: could not refresh testset #{num}: {e}")


def cmd_testset_refresh(args):
    """qra testset refresh — refresh AI answer key(s) for flat numbered testsets.

    Regenerates the AI answer key (model codes + justifications) for every testset,
    including imported legacy ones, from the current master segments. The human coding
    worksheet is never rewritten (only its 'N of M' header is synced). Each segment's
    text is validated against the SHA256 frozen at testset creation; drift is a hard
    error (use --force to regenerate against changed text).
    """
    import re as _re
    from process.assembly.human_forms import (
        refresh_testset_answer_key, _worksheet_is_legacy_import,
    )
    from process import output_paths as _paths

    segments = _load_segments_from_output(args.output_dir)
    segments_by_id = {s.segment_id: s for s in segments}
    framework = _load_framework(getattr(args, 'framework', None))

    testsets_root = _paths.testsets_dir(args.output_dir)
    if not os.path.isdir(testsets_root):
        print("No testsets directory found.")
        sys.exit(0)

    pattern = _re.compile(r'^human_classification_testset_worksheet_(\d+)\.txt$')
    nums = sorted(
        int(m.group(1)) for f in os.listdir(testsets_root)
        if (m := pattern.match(f))
    )

    if not nums:
        print("No testsets found.")
        return

    n_total = len(nums)
    force = getattr(args, 'force', False) or getattr(args, 'yes', False)
    failed = []
    for n in nums:
        is_legacy = _worksheet_is_legacy_import(_paths.testset_meta_path(args.output_dir, n))
        try:
            path = refresh_testset_answer_key(
                segments_by_id, framework, args.output_dir, n,
                codebook_enabled=False,
                n_total=n_total,
                force=force,
            )
            tag = " (legacy validated set)" if is_legacy else ""
            print(f"Refreshed{tag}: {os.path.relpath(path, args.output_dir)}")
        except Exception as e:
            print(f"Error refreshing testset #{n}: {e}")
            failed.append(n)

    if failed:
        print(f"\n{len(failed)} testset(s) could not be refreshed: "
              f"{', '.join(f'#{n}' for n in failed)}")
        sys.exit(1)


def cmd_testset_list(args):
    """qra testset list — list existing flat numbered testsets."""
    import re as _re
    from process.assembly.human_forms import _detect_worksheet_kind
    from process import output_paths as _paths

    testsets_root = _paths.testsets_dir(args.output_dir)
    if not os.path.isdir(testsets_root):
        print("No testsets found.")
        return

    pattern = _re.compile(r'^human_classification_testset_worksheet_(\d+)\.txt$')
    rows = []
    for fname in sorted(os.listdir(testsets_root)):
        m = pattern.match(fname)
        if not m:
            continue
        n = int(m.group(1))
        human_path = _paths.testset_human_flat_path(args.output_dir, n)
        ai_path = _paths.testset_ai_flat_path(args.output_dir, n)
        kind = _detect_worksheet_kind(human_path)
        has_ai = os.path.isfile(ai_path)
        rows.append({'n': n, 'kind': kind, 'has_ai': has_ai})

    if not rows:
        print("No testsets found.")
        return

    print(f"{'#':>4} {'Kind':<10} {'AI Key'}")
    print('-' * 30)
    for r in rows:
        ai_status = 'yes' if r['has_ai'] else 'missing'
        print(f"{r['n']:>4}  {r['kind']:<10} {ai_status}")


# =========================================================================
# cv subcommand group (Phase 2)
# =========================================================================

def cmd_cv_create(args):
    """qra cv create — create a content-validity testset."""
    from process.assembly.content_validity import create_frozen_content_validity_testset
    from process._freeze import FrozenArtifactError

    kind = args.framework
    if kind == 'codebook':
        print("Error: codebook content-validity is not yet supported — "
              "codebook codes have no exemplar utterances. Skipping.")
        sys.exit(2)

    if kind == 'purer':
        from theme_framework.registry import load as _registry_load_fw
        framework = _registry_load_fw('purer')
    else:
        framework = _load_framework(None)  # vaamr

    name = args.name or f'cv_{kind}_v1'
    try:
        d = create_frozen_content_validity_testset(
            framework, args.output_dir,
            name=name, kind=kind, force=args.force,
        )
        print(f"Created content-validity testset: {os.path.relpath(d, args.output_dir)}")
    except FrozenArtifactError as e:
        print(f"Error: {e}\n  Use --force to overwrite.")
        sys.exit(2)
    except NotImplementedError as e:
        print(f"Error: {e}")
        sys.exit(2)


def cmd_cv_refresh(args):
    """qra cv refresh — refresh AI answer key(s) for content-validity testsets."""
    from process.assembly.content_validity import refresh_cv_answer_key
    from process import output_paths as _paths

    cv_root = _paths.cv_testsets_dir(args.output_dir)
    if not os.path.isdir(cv_root):
        print("No content-validity testsets found.")
        sys.exit(0)

    if getattr(args, 'name', None) and not args.all:
        names = [args.name]
    else:
        names = sorted(
            e.name for e in os.scandir(cv_root)
            if e.is_dir() and os.path.isfile(os.path.join(e.path, 'manifest.json'))
        )

    if not names:
        print("No content-validity testsets found.")
        return

    # Build ThemeClassificationConfig from config file or CLI args.
    config = _build_config(args)
    tc = config.theme_classification if config is not None else None
    _backend = getattr(args, 'backend', None)
    _model = getattr(args, 'model', None)
    _lmurl = getattr(args, 'lmstudio_url', None)
    if _backend or _model or _lmurl:
        from theme_framework.config import ThemeClassificationConfig as _TCC
        tc = _TCC(
            backend=_backend or (tc.backend if tc else 'lmstudio'),
            model=_model or (tc.model if tc else None),
            lmstudio_base_url=_lmurl or (getattr(tc, 'lmstudio_base_url', None) if tc else 'http://127.0.0.1:1234/v1'),
        )

    for name in names:
        mp = _paths.cv_testset_manifest_path(args.output_dir, name)
        with open(mp) as f:
            m = json.load(f)
        kind = m.get('kind', 'vaamr')
        if kind == 'purer':
            from theme_framework.registry import load as _registry_load_fw
            framework = _registry_load_fw('purer')
        else:
            framework = _load_framework(None)
        try:
            path = refresh_cv_answer_key(args.output_dir, name, tc, framework)
            print(f"Refreshed: {os.path.relpath(path, args.output_dir)}")
        except Exception as e:
            print(f"Error refreshing {name!r}: {e}")


def cmd_cv_list(args):
    """qra cv list — list existing content-validity testsets."""
    from process.assembly.content_validity import list_content_validity_testsets

    results = list_content_validity_testsets(args.output_dir)
    if not results:
        print("No content-validity testsets found.")
        return

    print(f"{'Name':<30} {'Kind':<10} {'Items':>6} {'Framework':<15} {'Created'}")
    print('-' * 70)
    for r in results:
        fw = r.get('framework', {}).get('name', '?')
        print(f"{r['name']:<30} {r['kind']:<10} {r['n_items']:>6}   {fw:<15} {r['created_at'][:10]}")


# =========================================================================
# Pipeline execution
# =========================================================================

def _wait_for_lmstudio(base_url: str, timeout_s: int = 32000, poll_s: int = 5) -> None:
    """
    Block until the LM Studio server responds to GET /v1/models, or timeout.
    """
    import time
    import requests

    url = base_url.rstrip('/') + '/models'
    waited = 0
    first_fail = True

    while waited < timeout_s:
        try:
            r = requests.get(url, timeout=4)
            if r.status_code < 500:
                if not first_fail:
                    print(" connected.")
                return
        except Exception:
            pass

        if first_fail:
            print(
                f"\n  LM Studio not yet reachable at {base_url}\n"
                f"  Waiting up to {timeout_s}s for the server to start",
                end='', flush=True,
            )
            first_fail = False
        else:
            print('.', end='', flush=True)

        time.sleep(poll_s)
        waited += poll_s

    print(f"\n  Warning: LM Studio did not respond within {timeout_s}s — proceeding anyway.")


def _execute_pipeline(config, framework, codebook=None):
    """Run the pipeline with the given config and framework."""
    from process.orchestrator import run_full_pipeline

    # LM Studio: wait until the server is reachable before starting
    if config.theme_classification.backend == 'lmstudio':
        _wait_for_lmstudio(
            getattr(config.theme_classification, 'lmstudio_base_url', 'http://127.0.0.1:1234/v1')
        )

    # Print header
    print("\n" + "=" * 70)
    print("QRA CLASSIFICATION PIPELINE")
    print("=" * 70)
    tc = config.theme_classification
    print(f"  Framework: {framework.name} ({framework.num_themes} themes)")
    per_run = getattr(tc, 'per_run_models', []) or []
    use_per_run = len(per_run) >= 2 and len(per_run) == tc.n_runs
    if use_per_run:
        print(f"  Models:    {tc.n_runs} distinct raters (per-run interrater)")
        for i, m in enumerate(per_run):
            print(f"             Run {i + 1}: {m}")
    else:
        print(f"  Model:     {tc.model}")
    print(f"  Backend:   {tc.backend}")
    if tc.backend == 'lmstudio':
        print(f"  LM Studio: {getattr(tc, 'lmstudio_base_url', 'http://127.0.0.1:1234/v1')}")
    if config.run_codebook_classifier:
        print(f"  Codebook:  enabled")
    print()

    master_df = run_full_pipeline(config, framework, codebook=codebook)

    # Summary
    print(f"\nTotal segments: {len(master_df)}")
    if 'primary_stage' in master_df.columns:
        labeled = master_df[master_df['primary_stage'].notna()]
        print(f"Successfully labeled: {len(labeled)}")

    return master_df


def _prompt_yes_no_simple(label: str, default: bool = True) -> bool:
    default_str = 'Y/n' if default else 'y/N'
    raw = input(f"\n{label} [{default_str}]: ").strip().lower()
    if not raw:
        return default
    return raw in ('y', 'yes')


# =========================================================================
# Main
# =========================================================================

def _build_parser() -> tuple:
    """Build and return the top-level ArgumentParser (extracted for testability)."""
    parser = argparse.ArgumentParser(
        prog='qra',
        description='QRA: Qualitative Research Algorithm pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive setup wizard
  python qra.py setup

  # Full pipeline with saved config
  python qra.py run --config ./qra_config.json

  # Add new transcripts (incremental — walks through new speakers interactively)
  python qra.py add-data --config ./qra_config.json

  # Full pipeline with inline model
  python qra.py run --backend lmstudio --model nvidia/nemotron-3-super -o ./data/output/

  # Modular re-classification workflow (existing / legacy project):
  python qra.py ingest -o ./data/output/
  python qra.py classify --what vaamr --backend lmstudio --model <new_model> -o ./data/output/
  python qra.py classify --what purer --backend lmstudio --model <new_model> -o ./data/output/
  python qra.py assemble -o ./data/output/
  python qra.py validate -o ./data/output/
  python qra.py analyze -o ./data/output/

  # Re-classify everything with new models (no config file needed):
  python qra.py classify --what vaamr --backend lmstudio --model <m> -o ./data/output/
  python qra.py assemble -o ./data/output/

  # Testset management
  python qra.py testset create -o ./data/output/ --kind purer --name purer_irr_1
  python qra.py testset refresh --all -o ./data/output/
  python qra.py testset list -o ./data/output/

  # Content-validity management
  python qra.py cv create -o ./data/output/ --framework purer --name cv_purer_v1
  python qra.py cv refresh --all --model <new_model> -o ./data/output/
  python qra.py cv list -o ./data/output/

  # Refresh only validation artifacts (after manual label edits)
  python qra.py validate -o ./data/output/

  # Zero-shot content validity test (skips full pipeline)
  python qra.py run --test-zeroshot --preset small --lmstudio-url http://10.0.0.58:1234/v1 -o ./data/output/
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # ---- setup ----
    subparsers.add_parser('setup', help='Interactive configuration wizard')

    # ---- ingest (Phase 3) ----
    ingest_parser = subparsers.add_parser(
        'ingest',
        help='Segment transcripts and freeze to disk (Phase 3)',
    )
    ingest_parser.add_argument('--output-dir', '-o', required=True)
    ingest_parser.add_argument('--config', '-c', default=None)
    ingest_parser.add_argument('--transcript-dir', default=None)
    ingest_parser.add_argument('--framework', default=None)
    ingest_parser.add_argument('--reingest', default=None, metavar='SESSION_ID',
                               help='Force re-segmentation of one session')
    ingest_parser.add_argument('--reingest-all', action='store_true',
                               help='Force re-segmentation of all sessions')
    ingest_parser.add_argument('--no-text-anonymization', action='store_true',
                               help='Disable PHI name scrubbing during ingestion')
    # Accept (and ignore) extra common args so existing configs work
    for _a in ('--backend', '--model', '--api-key', '--trial-id'):
        try:
            ingest_parser.add_argument(_a, default=None)
        except argparse.ArgumentError:
            pass

    # ---- classify (Phase 3) ----
    classify_parser = subparsers.add_parser(
        'classify',
        help='Run classifier(s) on frozen segments and write overlay files (Phase 3)',
    )
    classify_parser.add_argument('--output-dir', '-o', required=True)
    classify_parser.add_argument('--config', '-c', default=None)
    classify_parser.add_argument('--codebook', default=None)
    classify_parser.add_argument(
        '--what',
        default='all',
        choices=['vaamr', 'purer', 'codebook', 'microskill', 'cross-validation', 'all'],
        help=(
            'Which classifier to run (default: all).\n'
            '  vaamr            — VAAMR participant-stage classification\n'
            '  purer            — PURER therapist-move classification\n'
            '  codebook         — VCE phenomenology codebook\n'
            '  microskill       — therapist microcounseling-skill codebook\n'
            '  cross-validation — cross-validation overlay\n'
            '  all              — run every enabled classifier'
        ),
    )
    classify_parser.add_argument('--backend', default=None)
    classify_parser.add_argument('--model', default=None)
    classify_parser.add_argument('--api-key', default=None)
    classify_parser.add_argument(
        '--zero-shot',
        action='store_true',
        help='Run classification with zero-shot prompts (definitions only, no exemplar/'
             'subtle/adversarial utterances). Per-invocation; not persisted to config. '
             'Scope follows --what (vaamr → VAAMR only, purer → PURER only, all → both).',
    )
    classify_parser.add_argument(
        '--no-downstream',
        action='store_true',
        help='Skip automatic assemble → testset refresh → analyze after classification.',
    )

    # ---- assemble (Phase 3) ----
    assemble_parser = subparsers.add_parser(
        'assemble',
        help='Join frozen segments + overlays into master_segments (Phase 3)',
    )
    assemble_parser.add_argument('--output-dir', '-o', required=True)
    assemble_parser.add_argument('--config', '-c', default=None)

    # ---- run ----
    run_parser = subparsers.add_parser(
        'run', help='Execute the classification pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_args(run_parser)

    # ---- add-data ----
    add_data_parser = subparsers.add_parser(
        'add-data',
        help='Incrementally add new transcripts: segment + classify only new sessions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            'Adds new transcripts to an existing project without disturbing\n'
            'frozen segments, frozen validation testsets, or content-validity\n'
            'worksheets. Walks the user through extending the speaker\n'
            'anonymization key for any newly-discovered speakers.\n\n'
            'Requires an existing project (frozen segments + classification\n'
            'manifest). For a brand-new project, use `qra run` first.'
        ),
    )
    _add_common_args(add_data_parser)

    # ---- analyze ----
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Run results analysis on an existing pipeline output directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            'Reads master_segment_dataset.csv and theme_definitions.json from OUTPUT_DIR\n'
            'and produces per-session, per-participant, per-construct, and longitudinal\n'
            'reports in OUTPUT_DIR/reports/analysis/.'
        ),
    )
    analyze_parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='Path to pipeline output directory containing master_segment_dataset.csv',
    )
    _gnn_grp = analyze_parser.add_mutually_exclusive_group()
    _gnn_grp.add_argument(
        '--gnn', action='store_true',
        help='Force-enable the GNN representation-and-discovery layer for this run '
             '(overrides config.gnn_layer.enabled).',
    )
    _gnn_grp.add_argument(
        '--no-gnn', action='store_true',
        help='Force-disable the GNN layer for this run.',
    )

    # ---- validate ----
    validate_parser = subparsers.add_parser(
        'validate',
        help='Refresh validation artifacts (human forms, testsets, CV) without re-classifying',
    )
    validate_parser.add_argument('--output-dir', '-o', required=True, help='Pipeline output directory')
    validate_parser.add_argument('--config', '-c', default=None, help='Config JSON path')
    validate_parser.add_argument('--framework', default=None, help='Framework preset or JSON path')
    validate_parser.add_argument('--codebook', default=None, help='Codebook preset or JSON path')

    # ---- testsets (deprecated alias for testset refresh --all) ----
    testsets_parser = subparsers.add_parser(
        'testsets',
        help='[DEPRECATED] Use "testset refresh --all" instead',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    testsets_parser.add_argument('--output-dir', '-o', required=True)
    testsets_parser.add_argument('--config', '-c', default=None)
    testsets_parser.add_argument('--framework', default=None)
    testsets_parser.add_argument('--n-sets', type=int, default=2)
    testsets_parser.add_argument('--fraction', type=float, default=0.10)
    testsets_parser.add_argument('--seed', type=int, default=42)
    testsets_parser.add_argument('--codebook-enabled', action='store_true')

    # ---- testset (new subcommand group) ----
    testset_parser = subparsers.add_parser(
        'testset',
        help='Manage frozen validation testsets (create / refresh / list)',
    )
    testset_sub = testset_parser.add_subparsers(dest='testset_command')

    _ts_create = testset_sub.add_parser('create', help='Create a new flat numbered testset')
    _ts_create.add_argument('--output-dir', '-o', required=True)
    _ts_create.add_argument('--kind', required=True, choices=['vaamr', 'purer', 'codebook'])
    _ts_create.add_argument('--framework', default=None)
    _ts_create.add_argument('--fraction', type=float, default=0.10)
    _ts_create.add_argument('--seed', type=int, default=42)
    _ts_create.add_argument('--force', '--yes', '-y', dest='force', action='store_true',
                            help="Regenerate an AI key even if a segment's text no longer "
                                 "matches the frozen human worksheet (--yes is a deprecated alias)")

    _ts_refresh = testset_sub.add_parser('refresh', help='Refresh AI answer key(s)')
    _ts_refresh.add_argument('--output-dir', '-o', required=True)
    _ts_refresh.add_argument('--framework', default=None)
    _ts_refresh.add_argument('--force', '--yes', '-y', dest='force', action='store_true',
                             help="Regenerate an AI key even if a segment's text no longer "
                                  "matches the frozen human worksheet (--yes is a deprecated alias)")

    _ts_list = testset_sub.add_parser('list', help='List existing frozen testsets')
    _ts_list.add_argument('--output-dir', '-o', required=True)

    # ---- cv (content-validity subcommand group) ----
    cv_parser = subparsers.add_parser(
        'cv',
        help='Manage content-validity testsets (create / refresh / list)',
    )
    cv_sub = cv_parser.add_subparsers(dest='cv_command')

    _cv_create = cv_sub.add_parser('create', help='Create a content-validity testset')
    _cv_create.add_argument('--output-dir', '-o', required=True)
    _cv_create.add_argument('--framework', required=True, choices=['vaamr', 'purer', 'codebook'])
    _cv_create.add_argument('--name', default=None)
    _cv_create.add_argument('--force', action='store_true')

    _cv_refresh = cv_sub.add_parser('refresh', help='Refresh AI answer key(s)')
    _cv_refresh.add_argument('--output-dir', '-o', required=True)
    _cv_refresh.add_argument('--config', '-c', default=None)
    _cv_refresh.add_argument('--name', default=None)
    _cv_refresh.add_argument('--all', action='store_true')
    _cv_refresh.add_argument('--backend', default=None, help='LLM backend override')
    _cv_refresh.add_argument('--model', default=None, help='Model override')
    _cv_refresh.add_argument('--lmstudio-url', dest='lmstudio_url', default=None)

    _cv_list = cv_sub.add_parser('list', help='List content-validity testsets')
    _cv_list.add_argument('--output-dir', '-o', required=True)

    # ---- reclassify-run ----
    reclassify_parser = subparsers.add_parser(
        'reclassify-run',
        help='Re-classify a single run (e.g. to fix wrong model) without redoing other runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            'Patches the existing model-first checkpoint to remove one run\'s results,\n'
            'then re-classifies only that run using the specified model.\n\n'
            'Example: run 3 accidentally used gemma instead of nemotron:\n'
            '  qra reclassify-run -o ./data/MMORE_Processed --run 3 --model nvidia/nemotron-3-nano-30b\n'
        ),
    )
    reclassify_parser.add_argument('--output-dir', '-o', required=True,
                                   help='Pipeline output directory')
    reclassify_parser.add_argument('--run', type=int, required=True,
                                   help='1-indexed run number to redo (e.g. 3 for run 3/3)')
    reclassify_parser.add_argument('--model', default=None,
                                   help='New model for this run (overrides config and checkpoint)')
    reclassify_parser.add_argument('--checkpoint', default=None, metavar='PATH',
                                   help='Explicit path to *_runs.json checkpoint (auto-detected '
                                        'from output-dir if omitted)')
    reclassify_parser.add_argument('--config', '-c', default=None,
                                   help='Path to qra_config.json (auto-detected if omitted)')

    # ---- apply-anonymization ----
    anon_parser = subparsers.add_parser(
        'apply-anonymization',
        help='Retroactively scrub PHI names from frozen segment text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            'Applies PHI name scrubbing to already-frozen segments.jsonl files.\n'
            'Known names (from speaker_anonymization_key.json) are replaced with\n'
            '{anonymized_id} tokens; unrecognized names become (NAME).\n'
            'segmentation_meta.json is NOT modified.\n'
            'Run `qra assemble` afterward to rebuild master_segments.'
        ),
    )
    anon_parser.add_argument('--output-dir', '-o', required=True,
                             help='Project output directory')
    anon_parser.add_argument('--config', '-c', default=None,
                             help='Path to qra_config.json (to locate speaker key)')
    anon_parser.add_argument('--session', default=None, metavar='SESSION_ID',
                             help='Process one session only (default: all sessions)')
    anon_parser.add_argument('--yes', '-y', action='store_true',
                             help='Skip interactive confirmation prompt')
    anon_parser.add_argument('--no-spacy', action='store_true',
                             help='Use regex heuristics only (skip NLP engine)')
    anon_parser.add_argument('--force', action='store_true',
                             help='Proceed even without a speaker_anonymization_key.json')
    anon_parser.add_argument(
        '--model', default=None,
        help='HF de-id model name for unknown-name detection '
             '(default: obi/deid_roberta_i2b2)',
    )
    anon_parser.add_argument(
        '--confidence', type=float, default=None,
        help='Confidence threshold for model/Presidio predictions (default: 0.6)',
    )

    return parser, testset_parser, cv_parser


def main():
    parser, testset_parser, cv_parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        from process.interactive_tui import run_tui
        run_tui()
        sys.exit(0)

    try:
        if args.command == 'setup':
            cmd_setup(args)
        elif args.command == 'ingest':
            cmd_ingest(args)
        elif args.command == 'classify':
            cmd_classify(args)
        elif args.command == 'assemble':
            cmd_assemble(args)
        elif args.command == 'run':
            cmd_run(args)
        elif args.command == 'add-data':
            cmd_add_data(args)
        elif args.command == 'analyze':
            cmd_analyze(args)
        elif args.command == 'validate':
            cmd_validate(args)
        elif args.command == 'testsets':
            import warnings
            warnings.warn(
                "'qra testsets' is deprecated — use 'qra testset refresh --all' instead.",
                DeprecationWarning,
                stacklevel=1,
            )
            cmd_testsets(args)
        elif args.command == 'testset':
            if args.testset_command == 'create':
                cmd_testset_create(args)
            elif args.testset_command == 'refresh':
                cmd_testset_refresh(args)
            elif args.testset_command == 'list':
                cmd_testset_list(args)
            else:
                testset_parser.print_help()
        elif args.command == 'cv':
            if args.cv_command == 'create':
                cmd_cv_create(args)
            elif args.cv_command == 'refresh':
                cmd_cv_refresh(args)
            elif args.cv_command == 'list':
                cmd_cv_list(args)
            else:
                cv_parser.print_help()
        elif args.command == 'reclassify-run':
            cmd_reclassify_run(args)
        elif args.command == 'apply-anonymization':
            cmd_apply_anonymization(args)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
