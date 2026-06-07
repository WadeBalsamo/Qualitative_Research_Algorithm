#!/usr/bin/env python3
"""
qra.py
------
Unified CLI entry point for the Qualitative Research Algorithm (QRA) - a computational phenomenology pipeline.

This script orchestrates the complete workflow for analyzing therapy transcripts from the Move-MORE Feasibility Trial 
using two classification frameworks:
- VAAMR (Vigilance-Avoidance-Attention Regulation-Metacognition-Reappraisal): Classifies participant segments across a five-stage developmental arc
- PURER (Phenomenological-Utilization-Reframing-Educate/Expectancy-Reinforcement): Classifies therapist segments across five guided-inquiry moves
- VCE Phenomenology Codebook: Optional multi-label construct enrichment (54 codes, 6 domains) applied to participant segments

The pipeline has 8 stages (+2 optional sub-stages), sequenced by process/orchestrator.py:
1.  Ingestion & segmentation       -> 01_transcripts/segmented/<sid>/ (frozen)
2.  Construct operationalization    -> theme definitions + content-validity set
3.  VAAMR zero-shot classification  (participant segments; multi-run consensus)
3b. Codebook classification         (optional; participant segments)
3c. PURER cue-block classification  (optional; therapist segments)
4.  Cross-validation                (optional; VAAMR x VCE lift)
5.  Human validation set            -> 04_validation/
6.  Dataset assembly                -> 02_meta/training_data/master_segments.csv
7.  Report generation               -> 01_transcripts/coded/ + 04_validation/
8.  Results analysis                (optional) -> 03_analysis_data/, 05_figures/, 06_reports/

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
    - 01_transcripts/ : segmented/<sid>/ (frozen) + coded/ transcripts
    - 02_meta/ : config, classifications/ overlays, training_data/master_segments.csv
    - 04_validation/ : worksheets, definition/answer keys, testsets/, content_validity/
    - 05_figures/ : PNG visualization figures
    - 06_reports/ : tiered human-readable text reports (00_* through 06_gnn/)

analyze
  Run post-hoc analysis on existing pipeline output directory
  Input: Existing output dir with 02_meta/training_data/master_segments.csv
  Output:
    - 06_reports/ : tiered text reports — 00_executive_summary.txt, 01_outcomes/,
      02_mechanism/, 03_per_session/, 04_per_participant/, 05_per_stage/, 06_gnn/
    - 03_analysis_data/ : per-session / participant / theme JSON + graphing CSVs
    - 05_figures/ : heatmaps, trajectories, transition + GNN figures (PNG)

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
  Output: Classification overlays written to output_dir/02_meta/classifications/
    - theme_labels.jsonl : VAAMR classifications
    - purer_labels.jsonl : PURER classifications
    - codebook_labels.jsonl : Codebook classifications

assemble
  Stage 6: Join frozen segments with classification overlays into the master dataset
  Input: Frozen segments + classification overlays
  Output: output_dir/02_meta/training_data/master_segments.csv (integrated dataset)

add-data
  Incrementally segment + classify only NEW transcripts, then re-assemble and
  re-analyze. Frozen validation testsets are never mutated; exits non-zero if no
  new sessions are detected.

reclassify-run
  Re-run a single classifier over existing frozen segments (overlay refresh).

apply-anonymization
  Retroactively scrub PHI names from frozen segment text (segments.jsonl).

edit-anonymization
  Edit the speaker anonymization key (interactive TUI or flags) and cascade the
  change across frozen segments, overlays, checkpoints, worksheets, and reports.

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
  Output: output_dir/04_validation/testsets/<name>/
    - manifest.json : Test set metadata and segment IDs
    - human_worksheet.txt : Human-annotated answers (for validation)
    - ai_answer_key.json : AI-generated answer key for automated evaluation

 cv
  Manage content-validity test sets (Phase 2)
  create --framework vaamr/purer --name <name>   : Create new CV test set
  refresh --all/--name <name>                    : Refresh AI answer key using specified model(s)
  list                                          : List existing content-validity test sets
  Output: output_dir/04_validation/content_validity/<name>/
    - manifest.json : Test set metadata and segment IDs
    - human_worksheet.txt : Human-annotated answers (for validation)
    - ai_answer_key.json : AI-generated answer key for automated evaluation
    - definition_key.txt : Framework definitions used in test set

--test-zeroshot
  Run zero-shot LLM classification against content validity test sets only
  Bypasses full pipeline; skips ingestion and assembly stages
  Input: Content validity test set (created via cv create)
  Output: output_dir/04_validation/content_validity_zeroshot_results.txt
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
  
For more information on the neurophenomenological methodology, see docs/methodology.md.
"""

import argparse
import json
import os
import sys
import traceback

# Ensure src/ (where packages live) is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))


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
             '(skips full pipeline; writes graded report to 04_validation/)',
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

    # Pass through every top-level sub-config dict directly, except the ones
    # handled specially above ('pipeline') or consumed elsewhere ('framework',
    # 'codebook'). This keeps upgraded configs forward-compatible: newly-added
    # sub-config blocks (e.g. superposition, efficacy, gnn_layer) load fully
    # without needing to be enumerated here.
    for key, val in data.items():
        if isinstance(val, dict) and key not in ('pipeline', 'framework', 'codebook'):
            result[key] = val

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

    from classification_tools.zeroshot_reporting import write_zeroshot_report
    report_path = write_zeroshot_report(
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

    if getattr(args, 'therapist_only', False):
        from process.orchestrator import stage_resegment_therapist
        print(f"\nQRA INGEST (therapist-only re-segmentation)")
        print(f"  Output: {output_dir}")
        r = stage_resegment_therapist(config, output_dir)
        print(f"\nDone. Re-segmented {r.get('sessions', 0)} session(s).")
        print(f"  Therapist segments: {r.get('old_therapist', 0)} -> {r.get('new_therapist', 0)}")
        print(f"  Participant segments preserved: {r.get('participant_preserved', 0)}")
        print("Run `qra classify --what purer`, then `qra assemble` to rebuild master_segments.")
        return

    force_reingest = getattr(args, 'reingest', None)
    force_reingest_all = getattr(args, 'reingest_all', False) or getattr(args, 'fresh', False)

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
            f"  This will rewrite segment text for {len(sessions_to_process)} session(s)\n"
            "  in the project database (qra.db). Segmentation boundaries and metadata\n"
            "  are preserved; only the text of each frozen segment is updated.\n"
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


def _split_kv(spec: str, flag: str):
    """Split an OLD=NEW CLI spec on the first '='. Raises ValueError on bad input."""
    if '=' not in spec:
        raise ValueError(f"{flag} expects OLD=NEW, got {spec!r}")
    left, right = spec.split('=', 1)
    left, right = left.strip(), right.strip()
    if not left or not right:
        raise ValueError(f"{flag} expects non-empty OLD=NEW, got {spec!r}")
    return left, right


def cmd_edit_anonymization(args):
    """qra edit-anonymization — edit the speaker anonymization key and cascade downstream.

    With no edit flags this launches an interactive TUI.  With edit flags it
    applies them non-interactively, then propagates the change across frozen
    segments, classification overlays, checkpoints, validation worksheets, and
    regenerated master/analysis artifacts.
    """
    from process import anonymization_editor as _ae

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        print(f"Error: output directory not found: {output_dir}")
        sys.exit(1)

    old_key = _ae.load_key(output_dir)

    renames = getattr(args, 'rename', None) or []
    rename_raws = getattr(args, 'rename_raw', None) or []
    set_roles = getattr(args, 'set_role', None) or []
    merges = getattr(args, 'merge', None) or []
    remove_names = getattr(args, 'remove_names', False)
    has_edits = bool(renames or rename_raws or set_roles or merges)

    # No edits requested → interactive editor.
    if not has_edits and not remove_names:
        config = _build_config(args)
        _ae.run_anonymization_tui(output_dir, config=config)
        return

    new_key = old_key
    try:
        for spec in renames:
            old_id, new_id = _split_kv(spec, '--rename')
            new_key = _ae.rename_anon_id(new_key, old_id, new_id)
        for spec in rename_raws:
            old_raw, new_raw = _split_kv(spec, '--rename-raw')
            new_key = _ae.rename_raw_name(new_key, old_raw, new_raw)
        for spec in set_roles:
            anon_id, role = _split_kv(spec, '--set-role')
            new_key = _ae.change_role(new_key, anon_id, role)
        for spec in merges:
            raws_part, target = _split_kv(spec, '--merge')
            raw_list = [r.strip() for r in raws_part.split(',') if r.strip()]
            new_key = _ae.merge_speakers(new_key, raw_list, target)
    except (KeyError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(2)

    rescrub_opts = None
    if remove_names:
        rescrub_opts = {
            'use_transformer': not getattr(args, 'no_spacy', False),
            'confidence_threshold': getattr(args, 'confidence', None) or 0.6,
            'model_name': getattr(args, 'model', None) or 'obi/deid_roberta_i2b2',
        }

    prev = _ae.preview_key_update(output_dir, old_key, new_key)
    print("\nQRA EDIT-ANONYMIZATION")
    print(f"  Output dir       : {output_dir}")
    print(f"  Renamed IDs      : {prev['n_renamed_ids']}  {prev['relabel_map'] or ''}")
    print(f"  Segment IDs      : {prev['n_segment_ids']} to rewrite across {prev['n_sessions']} session(s)")
    print(f"  Overlays / ckpts : {prev['n_overlays']} / {prev['n_checkpoints']}")
    print(f"  Re-run scrub     : {'yes' if remove_names else 'no'}")

    if getattr(args, 'dry_run', False):
        print("\nDry run — no files written.")
        return

    if not getattr(args, 'yes', False):
        confirm = input(
            "\n  This will OVERWRITE frozen segments, overlays, checkpoints, validation\n"
            "  worksheets, and regenerate master + analysis. A backup is written first.\n"
            "  Type 'yes' to continue: "
        ).strip().lower()
        if confirm != 'yes':
            print("Aborted.")
            sys.exit(0)

    config = _build_config(args)
    stats = _ae.apply_key_update(
        output_dir, old_key, new_key,
        remove_names=remove_names,
        rescrub_opts=rescrub_opts,
        backup=not getattr(args, 'no_backup', False),
        config=config,
        regenerate=not getattr(args, 'no_regenerate', False),
        verbose=True,
    )
    print("\nDone.")
    print(f"  Backup            : {stats.get('backup')}")
    print(f"  Sessions rewritten: {stats['sessions_rewritten']}")
    print(f"  Overlay rows      : {stats['overlay_rows']}")
    print(f"  Checkpoint keys   : {stats['checkpoint_keys']}")
    print(f"  Testset txt files : {stats['testset_txt_files']}")
    print(f"  CV item rows      : {stats['cv_item_rows']}")
    print(f"  Testset meta files: {stats['testset_meta_files']}")
    if stats.get('regenerated'):
        print(f"  Regenerated master: {stats['regenerated']['master']} segments, "
              f"{stats['regenerated']['analysis_files']} analysis files")


def cmd_classify(args):
    """qra classify — run classifier(s) on frozen segments and write overlays."""
    from process import segments_io as _segments_io
    from process.orchestrator import (
        stage_classify_theme,
        stage_classify_purer,
        stage_classify_codebook,
        stage_cross_validation,
    )
    from process._freeze import FrozenArtifactError

    output_dir = args.output_dir
    what = getattr(args, 'what', 'all') or 'all'
    valid = {'vaamr', 'purer', 'codebook', 'cross-validation', 'all'}
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

    to_run = {what} if what != 'all' else {'vaamr', 'purer', 'codebook', 'cross-validation'}

    # --fresh: clear checkpoints + overlay for each targeted framework so the
    # classifier starts over instead of resuming from prior runs.
    if getattr(args, 'fresh', False):
        from process import reclassify_ops as _reclassify
        for w in sorted(to_run):
            r = _reclassify.reset_for_fresh(output_dir, w)
            print(f"  Fresh: cleared {r['checkpoints_removed']} checkpoint(s) + overlay for {w}")
        config.resume_from = None

    # Load frozen segments once (raw); apply overlays selectively per stage below.
    from process import classifications_io as _cio
    segments = _segments_io.load_segments_for_stage(output_dir, apply=())
    # Apply all existing overlays up-front so each stage sees the current on-disk state.
    by_id = {s.segment_id: s for s in segments}
    _cio.apply_overlays(output_dir, by_id, keys=('theme', 'purer', 'codebook', 'cv'))

    if 'vaamr' in to_run:
        print("  Running VAAMR classifier...")
        stage_classify_theme(config, framework, segments=segments, output_dir=output_dir)
        print(f"  theme overlay written to qra.db ({len(segments)} segments)")

    if 'purer' in to_run:
        print("  Running PURER classifier...")
        stage_classify_purer(config, segments=segments, output_dir=output_dir)
        print(f"  purer overlay written to qra.db")

    if 'codebook' in to_run:
        codebook = _load_codebook(getattr(args, 'codebook', None))
        print("  Running codebook classifier...")
        stage_classify_codebook(config, codebook, segments=segments, output_dir=output_dir)
        print(f"  codebook overlay written to qra.db")

    if 'cross-validation' in to_run:
        print("  Running cross-validation...")
        stage_cross_validation(config, framework, segments=segments, output_dir=output_dir)
        print(f"  cross-validation overlay written to qra.db")

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
    print(f"  master_segments.csv updated.")

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
        _cio.overlay_exists(output_dir, key)
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
    print(f"  master_segments.csv: {len(master_df)} segments")
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
# irr subcommand group (inter-rater reliability)
# =========================================================================

def _irr_run_full(output_dir, allow_drift=False):
    """Shared by `qra irr run` and `qra irr report`: analyze + report + figures."""
    from analysis import irr_analysis, irr_figures
    from analysis.reports import irr_report, irr_items
    from process import output_paths as _paths
    from process import irr_import

    try:
        results = irr_analysis.run_irr_analysis(
            output_dir, verbose=True, strict_drift=not allow_drift)
    except irr_import.TestsetDriftError as e:
        print("\nRefusing to run IRR — test-set content has drifted from the human-coded text:")
        print(str(e))
        print("\nThe machine labels would be scored against segments the humans never coded. "
              "Re-import the affected worksheet(s), or pass --allow-drift to score anyway "
              "(the report will flag the drift).")
        sys.exit(1)
    report_path = irr_report.generate_irr_report(results, output_dir)
    item_files = irr_items.write_irr_item_details(results, output_dir)
    figs = irr_figures.write_irr_figures(results, output_dir)
    print(f"Report:  {report_path}")
    print(f"Data:    {_paths.irr_validation_dir(output_dir)}")
    print(f"Per-item detail: {len(item_files)} test-set file(s)")
    print(f"Figures: {len(figs)} written")
    if not results.get('have_machine_labels'):
        print("Note: no frozen segments / machine labels — only Human↔Human computed.")
    return results


def cmd_irr_import(args):
    """qra irr import — import a human-coded IRR CSV into the project qra.db."""
    from process import irr_import
    if not os.path.isfile(args.csv):
        print(f"CSV not found: {args.csv}")
        sys.exit(1)
    irr_import.import_irr_csv(args.output_dir, args.csv, verbose=True)


def cmd_irr_run(args):
    """qra irr run — compute IRR (pull live LLM+GNN) + write report/figures/data."""
    _irr_run_full(args.output_dir, allow_drift=getattr(args, 'allow_drift', False))


def cmd_irr_report(args):
    """qra irr report — regenerate the IRR report + figures from a fresh analysis."""
    _irr_run_full(args.output_dir, allow_drift=getattr(args, 'allow_drift', False))


def cmd_irr_list(args):
    """qra irr list — list imported IRR test-sets."""
    from process import irr_import
    testsets = irr_import.list_imported_testsets(args.output_dir)
    if not testsets:
        print("No imported IRR test-sets found. Run `qra irr import` first.")
        return
    print(f"{'WS':>4}  {'Name':<14} {'Items':>6}  Raters")
    print('-' * 50)
    for t in testsets:
        print(f"{t['worksheet_n']:>4}  {t['name']:<14} {t['n_items']:>6}  "
              f"{', '.join(t['raters'])}")


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
    from process.assembly.content_validity import (
        refresh_cv_answer_key, list_content_validity_testsets,
    )

    entries = list_content_validity_testsets(args.output_dir)
    if not entries:
        print("No content-validity testsets found.")
        sys.exit(0)

    kind_by_name = {e['name']: e.get('kind', 'vaamr') for e in entries}

    if getattr(args, 'name', None) and not args.all:
        names = [args.name]
    else:
        names = sorted(kind_by_name)

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
        kind = kind_by_name.get(name, 'vaamr')
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

def _gnn_ballot_preflight(output_dir: str, *, force: bool = False) -> None:
    """Warn (or abort unless ``force``) when a project lacks multi-run LLM ballots.

    The GNN learns its consensus signal from per-segment rater ballots
    (``rater_votes``); a single-run LLM project trains on one-hot labels and the
    reliability gate κ becomes unreliable.  Never blocks training on a pre-flight
    error (degrades to a no-op).
    """
    try:
        import pandas as _pd
        from process import segments_io as _sio
        from process import classifications_io as _cio
        from gnn_layer.soft_labels import ballot_coverage
        segments = _sio.load_segments_for_stage(output_dir, apply=())
        by_id = {s.segment_id: s for s in segments}
        _cio.apply_overlays(output_dir, by_id, keys=('theme',))
        df = _pd.DataFrame([{'speaker': s.speaker, 'segment_id': s.segment_id,
                             'rater_votes': getattr(s, 'rater_votes', None)} for s in segments])
        cov = ballot_coverage(df)
    except Exception:
        return
    if cov['n_participant'] == 0 or cov['multirun_fraction'] >= 0.5:
        return
    pct = round(100 * cov['multirun_fraction'])
    print(
        "\nWARNING: this project's VAAMR labels look single-run\n"
        f"  (only {pct}% of participant segments carry multi-run ballots).\n"
        "  The GNN learns its consensus signal from per-segment rater ballots\n"
        "  (rater_votes); without >=2 runs it trains on one-hot labels only,\n"
        "  yielding a weak signal and an unreliable reliability gate.\n"
        "  Recommended: re-run VAAMR with n_runs >= 3 first, e.g.\n"
        "      qra classify --what vaamr --fresh\n"
    )
    if force:
        print("  (--force-ballots given; proceeding with degraded targets.)")
        return
    resp = input("  Proceed anyway with degraded targets? [y/N]: ").strip().lower()
    if resp not in ('y', 'yes'):
        print("Aborted. (Pass --force-ballots to skip this check.)")
        sys.exit(0)


def cmd_gnn_train(args):
    """qra gnn train — train the consensus-distillation CLASSIFIER + run the reliability gate.

    The GraphSAGE classifier is a SEPARATE, default-OFF concern (gnn_layer/classifier/): the
    Cohorts 1–2 pilot refuted its scaler role (H5; grouped-CV κ≈0.05–0.14, a probe ties/beats it).
    This command force-enables it (gnn_classifier_enabled=True) to train + gate it explicitly —
    e.g. to re-adjudicate at Cohorts 3–4 scale. The discovery + mechanism work-streams
    (discriminant validity / transition model / confound localization / communities) run by
    default at analyze-time regardless; the LLM consensus / probe remain the labels of record.
    """
    from analysis.runner import run_analysis
    from gnn_layer.classifier import validation as _val

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        print(f"Error: output directory not found: {output_dir}")
        sys.exit(1)

    _gnn_ballot_preflight(output_dir, force=getattr(args, 'force_ballots', False))

    print("\nQRA GNN TRAIN  (consensus-distillation classifier + reliability gate; default-OFF)")
    print(f"  Output: {output_dir}")
    run_analysis(output_dir, verbose=True, force_gnn=True, force_classifier=True)
    print()
    print(_val.format_gate_verdict(_val.read_gate_verdict(output_dir), output_dir))


def cmd_gnn_classify(args):
    """qra gnn classify — LLM-free scale-mode classification with the trained graph."""
    import pandas as _pd
    from process import segments_io as _sio
    from process import classifications_io as _cio
    from gnn_layer.runner import run_gnn_classify

    output_dir = args.output_dir
    sessions = _sio.list_segmented_sessions(output_dir)
    if not sessions:
        print(f"Error: no frozen segments in {output_dir}. Run `qra ingest` first.")
        sys.exit(1)

    config = _build_config(args)
    framework = _load_framework(getattr(args, 'framework', None))

    segments = _sio.load_segments_for_stage(output_dir, apply=())
    by_id = {s.segment_id: s for s in segments}
    _cio.apply_overlays(output_dir, by_id, keys=('theme', 'purer', 'codebook', 'cv', 'gnn'))
    df_all = _pd.DataFrame([{
        'segment_id': s.segment_id, 'text': s.text, 'speaker': s.speaker,
        'session_id': s.session_id,
        'start_time_ms': s.start_time_ms, 'end_time_ms': s.end_time_ms,
        'final_label': s.final_label, 'primary_stage': s.primary_stage,
        'purer_primary': getattr(s, 'purer_primary', None),
    } for s in segments])

    print("\nQRA GNN CLASSIFY  (graph-only, no LLM calls)")
    print(f"  Output: {output_dir}")
    if not config.gnn_layer.enabled:
        print("  Note: gnn_layer.enabled is False in config; using the existing checkpoint anyway.")
    res = run_gnn_classify(df_all, output_dir, framework=framework,
                           config=config.gnn_layer, verbose=True,
                           only_unlabeled=not getattr(args, 'all_segments', False))
    print(f"  status: {res['status']}; classified {res.get('n_classified', 0)} segment(s)")
    if not getattr(args, 'no_downstream', False):
        print("\nRunning downstream pipeline (assemble + analyze)...")
        cmd_assemble(args)
        cmd_analyze(args)


def cmd_gnn_status(args):
    """qra gnn status — print the GNN reliability-gate verdict (kappa vs LLM)."""
    from gnn_layer.classifier import validation as _val
    output_dir = args.output_dir
    verdict = _val.read_gate_verdict(output_dir)
    if getattr(args, 'json', False):
        print(json.dumps(verdict, indent=2))
        return
    print(_val.format_gate_verdict(verdict, output_dir))


def _load_master_df(output_dir):
    """Read the assembled master_segments.csv (where rater_votes is a JSON string)."""
    import pandas as _pd
    from process import output_paths as _paths
    csv_path = os.path.join(_paths.master_segments_dir(output_dir), 'master_segments.csv')
    if not os.path.isfile(csv_path):
        print(f"Error: {csv_path} not found. Run `qra assemble` first.")
        sys.exit(1)
    return _pd.read_csv(csv_path)


def _print_probe_verdict(v):
    """Human-readable probe gate summary (shared by `probe train` and `probe status`)."""
    if not v:
        print("No probe gate found. Run `qra probe train` first.")
        return
    def _f(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else 'n/a'
    hk, lk = v.get('probe_human_kappa'), v.get('probe_llm_kappa')
    hci, lci = v.get('probe_human_ci', [None, None]), v.get('probe_llm_ci', [None, None])
    print("PROBE SCALER — RELIABILITY GATE")
    print(f"  mode: {v.get('mode')}   raters: {', '.join(v.get('raters') or []) or '(A1n single probe)'}")
    print(f"  probe ↔ human : κ {_f(hk)} [{_f(hci[0])}, {_f(hci[1])}]  n={v.get('probe_human_n')}"
          f"   (floor {v.get('irr_human_band_floor')})")
    print(f"  probe ↔ LLM   : κ {_f(lk)} [{_f(lci[0])}, {_f(lci[1])}]  n={v.get('probe_llm_n')}")
    pcr = v.get('per_class_recall') or {}
    if pcr:
        print("  per-stage recall: " + ", ".join(f"{k} {_f(rv)}" for k, rv in pcr.items()))
    if v.get('rare_stage_notes'):
        print("  rare-stage: " + "; ".join(v['rare_stage_notes']))
    print(f"  → ready for LLM-free scaling: {'YES' if v.get('ready_for_scaling') else 'NO'}")
    print("    (the LLM consensus stays the label of record; the probe only FILLS unlabeled "
          "segments, tagged probe_consensus, never overriding the LLM.)")


def cmd_probe_train(args):
    """qra probe train — fit the per-rater ensemble scaler + run the participant-grouped gate."""
    from classification_tools import probe_classifier as _pc
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        print(f"Error: output directory not found: {output_dir}")
        sys.exit(1)
    config = _build_config(args)
    probe_cfg = getattr(config, 'probe', None) or _pc.ProbeConfig()
    df = _load_master_df(output_dir)
    print("\nQRA PROBE TRAIN  (LLM-free per-rater ensemble scaler + reliability gate)")
    print(f"  Output: {output_dir}")
    _pc.train_probe(df, output_dir, probe_cfg)
    verdict = _pc.evaluate_probe(df, output_dir, probe_cfg)
    print()
    _print_probe_verdict(verdict)


def cmd_probe_status(args):
    """qra probe status — print the probe reliability-gate verdict (probe↔human/LLM κ)."""
    from classification_tools import probe_classifier as _pc
    verdict = _pc.read_probe_gate(args.output_dir)
    if getattr(args, 'json', False):
        print(json.dumps(verdict, indent=2))
        return
    _print_probe_verdict(verdict)


def cmd_probe_classify(args):
    """qra probe classify — LLM-free label UNLABELED participant segments (gated + abstaining)."""
    from classification_tools import probe_classifier as _pc
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        print(f"Error: output directory not found: {output_dir}")
        sys.exit(1)
    config = _build_config(args)
    probe_cfg = getattr(config, 'probe', None) or _pc.ProbeConfig()
    df = _load_master_df(output_dir)
    if getattr(args, 'fresh', False):
        print("  --fresh: re-fitting the probe from scratch...")
        _pc.train_probe(df, output_dir, probe_cfg)
        _pc.evaluate_probe(df, output_dir, probe_cfg)
    if not _pc.probe_gate_ready(output_dir) and not getattr(args, 'force', False):
        print("\nProbe gate is NOT ready (probe↔human κ below the human band, or rare-stage "
              "collapse).\nRefusing to scale — the probe would add labels noisier than the LLM.")
        print("Re-run with --force to scale anyway (rows tagged probe_consensus, lower-confidence).")
        sys.exit(1)
    if _pc.load_probe_model(output_dir) is None:
        print("No persisted probe model. Run `qra probe train` (or pass --fresh) first.")
        sys.exit(1)
    print("\nQRA PROBE CLASSIFY  (LLM-free; fills unlabeled participant segments only)")
    n = _pc.classify_with_probe(df, output_dir, probe_cfg)
    print(f"  Probe labeled {n} previously-unlabeled participant segment(s) → probe_labels overlay.")
    if not getattr(args, 'no_downstream', False):
        print("\nRe-assembling master dataset (probe_consensus tier, below the LLM)...")
        cmd_assemble(args)


def cmd_migrate(args):
    """qra migrate — import a legacy JSONL project into qra.db (preview by default)."""
    from process import db as _db
    from process import legacy_migration as _lm

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        print(f"Error: output directory not found: {output_dir}")
        sys.exit(1)
    if _db.db_exists(output_dir):
        print(f"qra.db already present in {output_dir} — nothing to migrate.")
        return
    if not _lm.is_jsonl_project(output_dir):
        print(f"No legacy JSONL pipeline files found in {output_dir} — nothing to migrate.")
        return

    def _summary(c):
        ov = ', '.join(f"{k}:{n}" for k, n in sorted(c['overlays'].items())) or 'none'
        return (f"  segments:    {c['segments']} across {c['sessions']} session(s)\n"
                f"  overlays:    {ov}\n"
                f"  manifest:    {c['manifest_keys']} key(s)\n"
                f"  testsets:    {c['testset_worksheets']} worksheet(s)\n"
                f"  cv testsets: {c['cv_testsets']}")

    if not getattr(args, 'run', False):
        counts = _lm.preview_counts(output_dir)
        print(f"\nPreview — legacy JSONL project detected in {output_dir}:")
        print(_summary(counts))
        print("\nRun `qra migrate -o <dir> --run` to import it into qra.db")
        print("(non-destructive; originals are relocated to <dir>/_legacy_files/).")
        return

    print(f"\nMigrating legacy JSONL -> qra.db in {output_dir} ...")
    result = _lm.migrate_jsonl_to_sqlite(output_dir)
    print("Done. Imported:")
    print(_summary(result))
    print(f"Originals relocated to {os.path.join(output_dir, '_legacy_files')}/")


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

  # Re-classify a framework FROM SCRATCH (clears its checkpoints + overlay first):
  python qra.py classify --what vaamr --fresh -o ./data/output/
  # Re-segment every session from scratch:
  python qra.py ingest --fresh -o ./data/output/

  # GNN consensus layer (modular — add the GNN to an LLM-only project, then scale):
  python qra.py gnn train -o ./data/output/      # train graph + run the reliability gate
  python qra.py gnn status -o ./data/output/     # ready for LLM-free scaling? (kappa vs LLM)
  python qra.py gnn classify -o ./data/output/   # LLM-free label new/unlabeled segments

  # Import a legacy (pre-SQLite, JSONL-on-disk) project into qra.db
  python qra.py migrate -o ./data/output/        # preview what would be imported
  python qra.py migrate -o ./data/output/ --run  # perform the migration

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
    ingest_parser.add_argument('--fresh', '--from-scratch', dest='fresh', action='store_true',
                               help='Re-segment every session from scratch (alias for --reingest-all)')
    ingest_parser.add_argument('--no-text-anonymization', action='store_true',
                               help='Disable PHI name scrubbing during ingestion')
    ingest_parser.add_argument('--therapist-only', action='store_true',
                               help='Re-segment ONLY therapist (PURER) content, preserving '
                                    'participant VAAMR segments + frozen testsets')
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
        choices=['vaamr', 'purer', 'codebook', 'cross-validation', 'all'],
        help=(
            'Which classifier to run (default: all).\n'
            '  vaamr            — VAAMR participant-stage classification\n'
            '  purer            — PURER therapist-move classification\n'
            '  codebook         — VCE phenomenology codebook\n'
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
    classify_parser.add_argument(
        '--fresh', '--from-scratch', dest='fresh', action='store_true',
        help='Re-classify from scratch: clear the targeted framework\'s LLM '
             'checkpoints and overlay before running (scope follows --what). '
             'Without this, classification resumes from existing checkpoints.',
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
            'Reads master_segments.csv and theme_definitions.json from OUTPUT_DIR\n'
            'and produces per-session, per-participant, per-construct, and longitudinal\n'
            'reports in OUTPUT_DIR/06_reports/ (and graph-ready data in 03_analysis_data/).'
        ),
    )
    analyze_parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='Path to pipeline output directory containing master_segments.csv',
    )
    analyze_parser.add_argument('--config', '-c', default=None, help='Config JSON path')
    analyze_parser.add_argument('--framework', default=None, help='Framework preset or JSON path')
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

    # ---- testset (subcommand group) ----
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
    _ts_refresh.add_argument('--all', action='store_true',
                             help='Refresh all testsets (the default behavior; '
                                  'accepted for symmetry with `cv refresh --all`)')
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

    # ---- irr (inter-rater reliability subcommand group) ----
    irr_parser = subparsers.add_parser(
        'irr',
        help='Inter-rater reliability: import human codes, compute Human↔Human / '
             'Human↔LLM / Human↔GNN (bare `qra irr` launches the TUI)',
    )
    irr_sub = irr_parser.add_subparsers(dest='irr_command')

    _irr_import = irr_sub.add_parser('import', help='Import a human-coded IRR CSV into qra.db')
    _irr_import.add_argument('--output-dir', '-o', required=True)
    _irr_import.add_argument('--csv', required=True, help='Path to human_coded_testsets.csv')

    _irr_run = irr_sub.add_parser('run', help='Compute IRR (live LLM+GNN) + write report/figures')
    _irr_run.add_argument('--output-dir', '-o', required=True)
    _irr_run.add_argument('--allow-drift', action='store_true',
                          help='Score IRR even if test-set segment text has drifted from the '
                               'human-coded content (default: refuse). Drift is flagged in the report.')

    _irr_report = irr_sub.add_parser('report', help='Regenerate the IRR report + figures')
    _irr_report.add_argument('--output-dir', '-o', required=True)
    _irr_report.add_argument('--allow-drift', action='store_true',
                             help='Regenerate even if test-set segment text has drifted (default: refuse).')

    _irr_list = irr_sub.add_parser('list', help='List imported IRR test-sets')
    _irr_list.add_argument('--output-dir', '-o', required=True)

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

    # ---- edit-anonymization ----
    edit_anon_parser = subparsers.add_parser(
        'edit-anonymization',
        help='Edit the speaker anonymization key and cascade the change downstream',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            'Edit the speaker anonymization key (rename anonymized IDs, fix raw\n'
            'names, change roles, merge speakers) and propagate the change across\n'
            'EVERY downstream artifact: frozen segment fields + segment_ids +\n'
            '{token} text, classification overlays, checkpoints, validation\n'
            'worksheets, and regenerated master + analysis/reports.\n\n'
            'With no edit flags, an interactive editor (TUI) is launched.\n\n'
            'Examples:\n'
            '  qra edit-anonymization -o ./data/MMORE_Processed\n'
            '  qra edit-anonymization -o ./data/MMORE_Processed --rename therapist_1=therapist_9\n'
            '  qra edit-anonymization -o ./data/MMORE_Processed --remove-names --dry-run\n'
        ),
    )
    edit_anon_parser.add_argument('--output-dir', '-o', required=True,
                                  help='Project output directory')
    edit_anon_parser.add_argument('--config', '-c', default=None,
                                  help='Path to qra_config.json (auto-detected if omitted)')
    edit_anon_parser.add_argument('--rename', action='append', metavar='OLD=NEW',
                                  help='Rename an anonymized_id (repeatable)')
    edit_anon_parser.add_argument('--rename-raw', action='append', metavar='OLD=NEW',
                                  help='Rename an original/raw speaker label (repeatable)')
    edit_anon_parser.add_argument('--set-role', action='append', metavar='ID=ROLE',
                                  help='Set a speaker role: participant/therapist/staff (repeatable)')
    edit_anon_parser.add_argument('--merge', action='append', metavar='RAW1,RAW2=ANON_ID',
                                  help='Point several raw labels at one anonymized_id (repeatable)')
    edit_anon_parser.add_argument('--remove-names', action='store_true',
                                  help='Re-run the NLP de-id scrub over segment text after remapping')
    edit_anon_parser.add_argument('--no-spacy', action='store_true',
                                  help='With --remove-names: regex heuristics only (skip NLP engine)')
    edit_anon_parser.add_argument('--model', default=None,
                                  help='With --remove-names: HF de-id model (default: obi/deid_roberta_i2b2)')
    edit_anon_parser.add_argument('--confidence', type=float, default=None,
                                  help='With --remove-names: confidence threshold (default: 0.6)')
    edit_anon_parser.add_argument('--no-regenerate', action='store_true',
                                  help='Skip Phase B regeneration of master + analysis')
    edit_anon_parser.add_argument('--no-backup', action='store_true',
                                  help='Skip the timestamped backup written before any change')
    edit_anon_parser.add_argument('--dry-run', action='store_true',
                                  help='Print the relabel/segment-id maps and counts, write nothing')
    edit_anon_parser.add_argument('--yes', '-y', action='store_true',
                                  help='Skip interactive confirmation prompt')

    # ---- gnn (subcommand group) ----
    gnn_parser = subparsers.add_parser(
        'gnn',
        help='GNN consensus layer: train / classify (LLM-free) / status',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            'Run the GNN representation-and-consensus layer as modular steps:\n\n'
            '  gnn train     Train the graph, run the reliability gate, and write\n'
            '                GNN reports + the consensus overlay. Works on a project\n'
            '                that previously ran only LLM consensus (adds the GNN).\n'
            '  gnn classify  LLM-free scale-mode classification of new/unlabeled\n'
            '                segments with the already-trained graph (no LLM calls).\n'
            '  gnn status    Print the reliability-gate verdict (kappa vs LLM;\n'
            '                "ready for LLM-free scaling?").\n'
        ),
    )
    gnn_sub = gnn_parser.add_subparsers(dest='gnn_command')

    _gnn_train = gnn_sub.add_parser(
        'train', help='Train the graph + reliability gate + reports (force-enables the GNN)')
    _gnn_train.add_argument('--output-dir', '-o', required=True)
    _gnn_train.add_argument('--config', '-c', default=None)
    _gnn_train.add_argument('--force-ballots', action='store_true',
                            help='Skip the multi-run-ballot pre-flight check')

    _gnn_classify = gnn_sub.add_parser(
        'classify', help='LLM-free scale-mode classification with the trained graph')
    _gnn_classify.add_argument('--output-dir', '-o', required=True)
    _gnn_classify.add_argument('--config', '-c', default=None)
    _gnn_classify.add_argument('--framework', default=None)
    _gnn_classify.add_argument('--all-segments', action='store_true',
                               help='(Re)label every segment, not only those lacking an LLM label')
    _gnn_classify.add_argument('--no-downstream', action='store_true',
                               help='Skip the automatic assemble + analyze afterward')

    _gnn_status = gnn_sub.add_parser(
        'status', help='Print the reliability-gate verdict (kappa vs LLM; ready for scaling?)')
    _gnn_status.add_argument('--output-dir', '-o', required=True)
    _gnn_status.add_argument('--json', action='store_true', help='Emit the raw verdict JSON')

    # ---- probe (subcommand group) — the LLM-free, gated, abstention-aware scaler ----
    probe_parser = subparsers.add_parser(
        'probe',
        help='LLM-free VAAMR scaler: train / status / classify (per-rater ensemble probe)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            'Run the calibrated per-rater ensemble probe — QRA\'s LLM-free, gated,\n'
            'abstention-aware VAAMR scaler (methodology §8.6). The multi-run LLM consensus\n'
            'stays the label of record; the probe FILLS unlabeled participant segments only,\n'
            'tagged probe_consensus (ranked BELOW the LLM), and never overrides it.\n\n'
            '  probe train     Fit the probe on the LLM/human label of record + run the\n'
            '                  participant-grouped reliability gate (probe↔human / probe↔LLM κ).\n'
            '  probe status    Print the gate verdict ("ready for LLM-free scaling?").\n'
            '  probe classify  LLM-free label UNLABELED participant segments (abstaining where\n'
            '                  unsure). Refuses unless the gate is ready (or --force).\n'
        ),
    )
    probe_sub = probe_parser.add_subparsers(dest='probe_command')

    _probe_train = probe_sub.add_parser(
        'train', help='Fit the probe + run the reliability gate')
    _probe_train.add_argument('--output-dir', '-o', required=True)
    _probe_train.add_argument('--config', '-c', default=None)

    _probe_status = probe_sub.add_parser(
        'status', help='Print the probe reliability-gate verdict (probe↔human/LLM κ)')
    _probe_status.add_argument('--output-dir', '-o', required=True)
    _probe_status.add_argument('--json', action='store_true', help='Emit the raw verdict JSON')

    _probe_classify = probe_sub.add_parser(
        'classify', help='LLM-free label UNLABELED participant segments (gated + abstaining)')
    _probe_classify.add_argument('--output-dir', '-o', required=True)
    _probe_classify.add_argument('--config', '-c', default=None)
    _probe_classify.add_argument('--fresh', action='store_true',
                                 help='Re-fit the probe from scratch before classifying')
    _probe_classify.add_argument('--force', action='store_true',
                                 help='Scale even if the gate is not ready (rows tagged lower-confidence)')
    _probe_classify.add_argument('--no-downstream', action='store_true',
                                 help='Skip the automatic re-assemble afterward')

    # ---- migrate ----
    migrate_parser = subparsers.add_parser(
        'migrate',
        help='Import a legacy JSONL project into qra.db (preview by default)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            'Import a pre-SQLite (JSONL-on-disk) project into the per-project qra.db\n'
            'store. Preview by default; pass --run to perform it. Non-destructive:\n'
            'originals are relocated to <output-dir>/_legacy_files/.\n\n'
            'Auto-runs on the first `qra ingest`/`qra run` of a legacy project too;\n'
            'this command makes it explicit and previewable.'
        ),
    )
    migrate_parser.add_argument('--output-dir', '-o', required=True)
    migrate_parser.add_argument('--run', action='store_true',
                                help='Perform the migration (default: non-destructive preview)')

    return parser, testset_parser, cv_parser, gnn_parser


def main():
    parser, testset_parser, cv_parser, gnn_parser = _build_parser()
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
        elif args.command == 'irr':
            if getattr(args, 'irr_command', None) == 'import':
                cmd_irr_import(args)
            elif args.irr_command == 'run':
                cmd_irr_run(args)
            elif args.irr_command == 'report':
                cmd_irr_report(args)
            elif args.irr_command == 'list':
                cmd_irr_list(args)
            else:
                from process.irr_tui import run_irr_tui
                run_irr_tui()
        elif args.command == 'gnn':
            if getattr(args, 'gnn_command', None) == 'train':
                cmd_gnn_train(args)
            elif args.gnn_command == 'classify':
                cmd_gnn_classify(args)
            elif args.gnn_command == 'status':
                cmd_gnn_status(args)
            else:
                gnn_parser.print_help()
        elif args.command == 'probe':
            if getattr(args, 'probe_command', None) == 'train':
                cmd_probe_train(args)
            elif args.probe_command == 'status':
                cmd_probe_status(args)
            elif args.probe_command == 'classify':
                cmd_probe_classify(args)
            else:
                probe_parser.print_help()
        elif args.command == 'migrate':
            cmd_migrate(args)
        elif args.command == 'reclassify-run':
            cmd_reclassify_run(args)
        elif args.command == 'apply-anonymization':
            cmd_apply_anonymization(args)
        elif args.command == 'edit-anonymization':
            cmd_edit_anonymization(args)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
