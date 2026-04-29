#!/usr/bin/env python3
"""
qra.py
------
Unified CLI entry point for the QRA classification pipeline.

Subcommands:
    setup     Interactive wizard that saves a config JSON
    run       Execute pipeline (always auto mode)
    analyze   Post-hoc analysis on an existing output directory
    testsets  Generate validation test set worksheets from existing output
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
        choices=['openrouter', 'replicate', 'huggingface', 'ollama', 'lmstudio'],
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
    parser.add_argument('--replicate-api-token', default=None)

    # Framework & codebook
    parser.add_argument(
        '--framework', default=None,
        help='Theme framework: "vammr" (default) or path to custom JSON',
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


# =========================================================================
# Config building
# =========================================================================

def _load_framework(framework_arg):
    """Load a ThemeFramework from preset name or JSON path."""
    if framework_arg is None or framework_arg == 'vammr':
        from theme_framework.vammr import get_vammr_framework
        return get_vammr_framework()
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

    # Start from config file if provided
    if args.config:
        with open(args.config) as f:
            file_data = json.load(f)
        # Use from_json for proper nested reconstruction
        config = PipelineConfig.from_json(_flatten_wizard_config(file_data))
    else:
        config = PipelineConfig()

    # CLI overrides (only override if explicitly provided)
    if args.transcript_dir is not None:
        config.transcript_dir = args.transcript_dir
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.trial_id is not None:
        config.trial_id = args.trial_id
    if args.resume_from is not None:
        config.resume_from = args.resume_from

    # Speaker filter
    sf_mode = getattr(args, 'speaker_filter_mode', None)
    exclude_spk = getattr(args, 'exclude_speakers', None)
    if sf_mode is not None:
        config.speaker_filter.mode = sf_mode
    if exclude_spk is not None:
        config.speaker_filter.mode = 'exclude'
        config.speaker_filter.speakers = exclude_spk

    # Feature flags
    if args.no_theme_labeler:
        config.run_theme_labeler = False
    if args.run_codebook_classifier:
        config.run_codebook_classifier = True
    if args.no_codebook_classifier:
        config.run_codebook_classifier = False
    if getattr(args, 'verbose_segmentation', False):
        config.segmentation.verbose_segmentation = True

    # Backend & model
    tc = config.theme_classification
    if args.backend is not None:
        tc.backend = args.backend
    if args.model is not None:
        tc.model = args.model
    lmstudio_url = getattr(args, 'lmstudio_url', None)
    if lmstudio_url is not None:
        tc.lmstudio_base_url = lmstudio_url
    if args.models is not None:
        tc.models = args.models
    if args.n_runs is not None:
        tc.n_runs = args.n_runs
    if args.temperature is not None:
        tc.temperature = args.temperature

    # API keys: CLI > env > existing config
    if args.api_key is not None:
        tc.api_key = args.api_key
    elif not tc.api_key:
        tc.api_key = os.environ.get('OPENROUTER_API_KEY', '')

    if args.replicate_api_token is not None:
        tc.replicate_api_token = args.replicate_api_token
    elif not tc.replicate_api_token:
        tc.replicate_api_token = os.environ.get('REPLICATE_API_TOKEN', '')

    # Embedding settings
    emb = config.codebook_embedding
    if args.no_two_pass:
        emb.two_pass = False
    if getattr(args, 'embedding_model', None) is not None:
        emb.embedding_model = args.embedding_model
    if args.exemplar_import_path is not None:
        emb.exemplar_import_path = args.exemplar_import_path
    if args.criteria_weight is not None:
        emb.criteria_weight = args.criteria_weight
    if args.exemplar_weight is not None:
        emb.exemplar_weight = args.exemplar_weight
    if args.exemplar_confidence_threshold is not None:
        emb.exemplar_confidence_threshold = args.exemplar_confidence_threshold
    if args.max_exemplar_tokens is not None:
        emb.max_exemplar_tokens = args.max_exemplar_tokens

    # Confidence thresholds
    if args.high_confidence_threshold is not None:
        config.confidence_tiers.high_confidence = args.high_confidence_threshold
    if args.medium_confidence_threshold is not None:
        config.confidence_tiers.medium_min_confidence = args.medium_confidence_threshold

    if getattr(args, 'no_auto_analyze', False):
        config.auto_analyze = False

    return config


def _flatten_wizard_config(data: dict) -> dict:
    """Flatten wizard-format config (nested pipeline/theme_classification keys)
    into PipelineConfig.from_json-compatible dict."""
    result = {}

    # Lift pipeline-level keys to top level
    pipeline = data.get('pipeline', {})
    for key in ('transcript_dir', 'output_dir', 'trial_id',
                'run_theme_labeler', 'run_codebook_classifier'):
        if key in pipeline:
            result[key] = pipeline[key]

    # Pass through sub-config dicts directly
    for key in ('segmentation', 'speaker_filter', 'theme_classification', 'codebook_embedding',
                'codebook_llm', 'codebook_ensemble', 'validation', 'confidence_tiers',
                'test_sets'):
        if key in data:
            result[key] = data[key]

    # Copy top-level keys that are already flat
    for key in ('resume_from', 'autoresearch_dir'):
        if key in data:
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
        framework = _load_framework(framework_spec.get('custom_path') or framework_spec.get('preset', 'vammr'))

        codebook = None
        if config.run_codebook_classifier:
            codebook_spec = result['config_data'].get('codebook', {})
            codebook = _load_codebook(codebook_spec.get('custom_path') or codebook_spec.get('preset', 'phenomenology'))

        _execute_pipeline(config, framework, codebook)


def cmd_run(args):
    """Execute pipeline."""
    config = _build_config(args)

    # Load framework
    framework_arg = args.framework
    if framework_arg is None and args.config:
        with open(args.config) as f:
            file_data = json.load(f)
        fw = file_data.get('framework', {})
        framework_arg = fw.get('custom_path') or fw.get('preset', 'vammr')
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

    # HuggingFace model preloading
    if config.theme_classification.backend == 'huggingface':
        _preload_huggingface_models(config)

    _execute_pipeline(config, framework, codebook)


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

    result = run_analysis(output_dir, verbose=True)

    print(f"\nAnalysis complete.")
    print(f"  {result['n_segments']} segments | "
          f"{result['n_participants']} participants | "
          f"{result['n_sessions']} sessions")
    print(f"  Reports: {output_dir}/02_human_reports/ and {output_dir}/04_analysis_data/")
    print(f"  Files generated: {len(result['files_generated'])}")


def cmd_testsets(args):
    """Generate validation test set worksheets from existing pipeline output."""
    import glob
    from process.assembly import export_validation_testsets
    from process import output_paths as _paths

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        print(f"Error: output directory not found: {output_dir}")
        sys.exit(1)

    # Find master JSONL in new layout, then fall back to legacy root location
    _ms_dir = _paths.master_segments_dir(output_dir)
    jsonl_files = sorted(glob.glob(os.path.join(_ms_dir, 'master_segments*.jsonl')))
    if not jsonl_files:
        jsonl_files = sorted(glob.glob(os.path.join(output_dir, 'master_segments_*.jsonl')))
    if not jsonl_files:
        print(f"Error: no master_segments*.jsonl found in {output_dir}")
        print("Run the pipeline first: python qra.py run --config <config>")
        sys.exit(1)

    jsonl_path = jsonl_files[-1]
    print(f"\nLoading segments from: {os.path.relpath(jsonl_path)}")

    segments = _load_segments_from_jsonl(jsonl_path)
    participant_segs = [s for s in segments if s.speaker == 'participant']
    print(f"Loaded {len(segments)} segments ({len(participant_segs)} participant)")

    # Load framework
    framework_arg = args.framework
    if framework_arg is None and args.config:
        with open(args.config) as f:
            fw_data = json.load(f)
        fw = fw_data.get('framework', {})
        framework_arg = fw.get('custom_path') or fw.get('preset', 'vammr')
    framework = _load_framework(framework_arg)

    # Determine codebook status
    codebook_enabled = args.codebook_enabled
    if not codebook_enabled and args.config:
        with open(args.config) as f:
            cfg_data = json.load(f)
        codebook_enabled = cfg_data.get('pipeline', {}).get('run_codebook_classifier', False)

    # Generate
    written = export_validation_testsets(
        segments,
        framework,
        output_dir,
        n_sets=args.n_sets,
        fraction_per_set=args.fraction,
        random_seed=args.seed,
        codebook_enabled=codebook_enabled,
    )

    print(f"\nGenerated {len(written)} worksheet files:")
    for p in written:
        print(f"  {os.path.relpath(p, output_dir)}")


def _load_segments_from_jsonl(jsonl_path: str):
    """Reconstruct Segment objects from a master_segments JSONL file."""
    import pandas as pd
    from classification_tools.data_structures import Segment

    df = pd.read_json(jsonl_path, lines=True)
    segments = []

    def _int_or_none(val):
        try:
            return int(val) if val is not None and str(val) != 'nan' else None
        except (ValueError, TypeError):
            return None

    def _float_or_none(val):
        try:
            return float(val) if val is not None and str(val) != 'nan' else None
        except (ValueError, TypeError):
            return None

    def _list_or_none(val):
        if val is None:
            return None
        if isinstance(val, list):
            return val if val else None
        return None

    for _, row in df.iterrows():
        # rater_ids and rater_votes are stored as JSON strings
        rater_ids = None
        if isinstance(row.get('rater_ids'), str):
            try:
                rater_ids = json.loads(row['rater_ids'])
            except (json.JSONDecodeError, TypeError):
                pass

        rater_votes = None
        if isinstance(row.get('rater_votes'), str):
            try:
                rater_votes = json.loads(row['rater_votes'])
            except (json.JSONDecodeError, TypeError):
                pass

        seg = Segment(
            segment_id=str(row['segment_id']),
            trial_id=str(row.get('trial_id', '')),
            participant_id=str(row.get('participant_id', '')),
            session_id=str(row['session_id']),
            session_number=_int_or_none(row.get('session_number')) or 1,
            cohort_id=_int_or_none(row.get('cohort_id')),
            session_variant=str(row.get('session_variant', '') or ''),
            segment_index=_int_or_none(row.get('segment_index')) or 0,
            start_time_ms=_int_or_none(row.get('start_time_ms')) or 0,
            end_time_ms=_int_or_none(row.get('end_time_ms')) or 0,
            total_segments_in_session=_int_or_none(row.get('total_segments_in_session')) or 1,
            speaker=str(row.get('speaker', '')),
            text=str(row.get('text', '')),
            word_count=_int_or_none(row.get('word_count')) or 0,
            primary_stage=_int_or_none(row.get('primary_stage')),
            secondary_stage=_int_or_none(row.get('secondary_stage')),
            llm_confidence_primary=_float_or_none(row.get('llm_confidence_primary')),
            llm_confidence_secondary=_float_or_none(row.get('llm_confidence_secondary')),
            llm_justification=row.get('llm_justification'),
            llm_run_consistency=_int_or_none(row.get('llm_run_consistency')),
            rater_ids=rater_ids,
            rater_votes=rater_votes,
            agreement_level=str(row.get('agreement_level') or ''),
            agreement_fraction=_float_or_none(row.get('agreement_fraction')),
            needs_review=bool(row.get('needs_review', False)),
            consensus_vote=row.get('consensus_vote'),
            tie_broken_by_confidence=bool(row.get('tie_broken_by_confidence', False)),
            codebook_labels_embedding=_list_or_none(row.get('codebook_labels_embedding')),
            codebook_labels_llm=_list_or_none(row.get('codebook_labels_llm')),
            codebook_labels_ensemble=_list_or_none(row.get('codebook_labels_ensemble')),
            codebook_disagreements=_list_or_none(row.get('codebook_disagreements')),
            codebook_confidence=None,
            human_label=_int_or_none(row.get('human_label')),
            human_secondary_label=_int_or_none(row.get('human_secondary_label')),
            adjudicated_label=_int_or_none(row.get('adjudicated_label')),
            in_human_coded_subset=bool(row.get('in_human_coded_subset', False)),
            label_status=str(row.get('label_status', 'llm_only')),
            final_label=_int_or_none(row.get('final_label')),
            final_label_source=row.get('final_label_source'),
            label_confidence_tier=row.get('label_confidence_tier'),
            speakers_in_segment=None,
            session_file=str(row.get('session_file', '')),
        )
        segments.append(seg)

    return segments


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


def _preload_huggingface_models(config):
    """Download and preload HuggingFace models."""
    from classification_tools.model_loader import ensure_models_ready, load_model

    models = config.theme_classification.models
    if not models:
        models = [config.theme_classification.model]

    print("\nPreparing HuggingFace models...")
    ensure_models_ready(download_if_missing=True)

    for model_id in models:
        try:
            print(f"  Loading {model_id}...")
            load_model(model_id)
            print(f"  Loaded successfully")
        except Exception as e:
            print(f"  Failed to load {model_id}: {e}")
            sys.exit(1)


def _prompt_yes_no_simple(label: str, default: bool = True) -> bool:
    default_str = 'Y/n' if default else 'y/N'
    raw = input(f"\n{label} [{default_str}]: ").strip().lower()
    if not raw:
        return default
    return raw in ('y', 'yes')


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        prog='qra',
        description='QRA: Qualitative Research Algorithm pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive setup wizard
  python qra.py setup

  # Run pipeline with defaults
  python qra.py run --backend openrouter --model openai/gpt-4o

  # Run with saved config
  python qra.py run --config ./qra_config.json

  # HuggingFace multi-model pipeline
  python qra.py run --backend huggingface

  # Run pipeline and auto-generate analysis reports afterward
  python qra.py run --backend lmstudio --auto-analyze

  # Analyze existing pipeline output (standalone post-hoc)
  python qra.py analyze --output-dir ./data/output/

  # Generate validation test set worksheets from existing output
  python qra.py testsets --output-dir ./data/output/
  python qra.py testsets --output-dir ./data/output/ --n-sets 3 --fraction 0.15
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # ---- setup ----
    subparsers.add_parser('setup', help='Interactive configuration wizard')

    # ---- run ----
    run_parser = subparsers.add_parser(
        'run', help='Execute the classification pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_args(run_parser)

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

    # ---- testsets ----
    testsets_parser = subparsers.add_parser(
        'testsets',
        help='Generate validation test set worksheets from existing pipeline output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            'Loads master_segments_*.jsonl from OUTPUT_DIR and generates cross-session\n'
            'human coding and AI classification worksheets for inter-rater reliability.'
        ),
    )
    testsets_parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='Path to pipeline output directory (must contain master_segments_*.jsonl)',
    )
    testsets_parser.add_argument(
        '--config', '-c',
        default=None,
        help='Path to config JSON (used to load framework and codebook settings)',
    )
    testsets_parser.add_argument(
        '--framework', default=None,
        help='Theme framework: "vammr" (default) or path to custom JSON',
    )
    testsets_parser.add_argument(
        '--n-sets', type=int, default=2,
        help='Number of test sets to generate (default: 2)',
    )
    testsets_parser.add_argument(
        '--fraction', type=float, default=0.10,
        help='Fraction of participant segments per set (default: 0.10)',
    )
    testsets_parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)',
    )
    testsets_parser.add_argument(
        '--codebook-enabled', action='store_true',
        help='Include codebook classification labels in AI worksheet',
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == 'setup':
            cmd_setup(args)
        elif args.command == 'run':
            cmd_run(args)
        elif args.command == 'analyze':
            cmd_analyze(args)
        elif args.command == 'testsets':
            cmd_testsets(args)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
