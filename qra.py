#!/usr/bin/env python3
"""
qra.py
------
Unified CLI entry point for the QRA classification pipeline.

Subcommands:
    setup    Interactive wizard that saves a config JSON
    run      Execute pipeline (auto/interactive/review modes)
    guided   Execute with step-by-step educational narration
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
    """Add arguments shared by `run` and `guided` subcommands."""
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
        choices=['ollama', 'openrouter', 'replicate', 'huggingface'],
        help='LLM backend (default: ollama)',
    )
    parser.add_argument(
        '--model', default=None,
        help='Model name (default: mixtral:8x7b for Ollama)',
    )
    parser.add_argument(
        '--models', nargs='+', default=None,
        help='Two model IDs for cross-referencing (Ollama only)',
    )
    parser.add_argument('--ollama-host', default=None, help='Ollama host (default: 0.0.0.0)')
    parser.add_argument('--ollama-port', type=int, default=None, help='Ollama port (default: 11434)')
    # Cloud backend credentials (not needed for Ollama)
    parser.add_argument('--api-key', default=None, help='OpenRouter API key')
    parser.add_argument('--replicate-api-token', default=None)

    # Framework & codebook
    parser.add_argument(
        '--framework', default=None,
        help='Theme framework: "vamr" (default) or path to custom JSON',
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

    # Embedding classifier settings
    parser.add_argument('--no-two-pass', action='store_true')
    parser.add_argument('--exemplar-import-path', default=None)
    parser.add_argument('--criteria-weight', type=float, default=None)
    parser.add_argument('--exemplar-weight', type=float, default=None)
    parser.add_argument('--exemplar-confidence-threshold', type=float, default=None)
    parser.add_argument('--max-exemplar-tokens', type=int, default=None)

    # Checkpoint
    parser.add_argument('--resume-from', default=None)


# =========================================================================
# Config building
# =========================================================================

def _load_framework(framework_arg):
    """Load a ThemeFramework from preset name or JSON path."""
    if framework_arg is None or framework_arg == 'vamr':
        from constructs.vamr import get_vamr_framework
        return get_vamr_framework()

    # Custom JSON path
    with open(framework_arg) as f:
        fw_data = json.load(f)
    from constructs.theme_schema import ThemeFramework, ThemeDefinition
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
    from process.config import (
        PipelineConfig, SegmentationConfig, ValidationConfig, ConfidenceTierConfig,
    )
    from constructs.config import ThemeClassificationConfig
    from codebook.config import EmbeddingClassifierConfig

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
    if hasattr(args, 'mode') and args.mode is not None:
        config.run_mode = args.mode
    if args.resume_from is not None:
        config.resume_from = args.resume_from

    # Feature flags
    if args.no_theme_labeler:
        config.run_theme_labeler = False
    if args.run_codebook_classifier:
        config.run_codebook_classifier = True
    if args.no_codebook_classifier:
        config.run_codebook_classifier = False

    # Backend & model
    tc = config.theme_classification
    if args.backend is not None:
        tc.backend = args.backend
    if args.model is not None:
        tc.model = args.model
    if args.models is not None:
        tc.models = args.models
    if args.n_runs is not None:
        tc.n_runs = args.n_runs
    if args.temperature is not None:
        tc.temperature = args.temperature

    # API keys: only needed for cloud backends (openrouter / replicate)
    if args.api_key is not None:
        tc.api_key = args.api_key
    elif not tc.api_key:
        tc.api_key = os.environ.get('OPENROUTER_API_KEY', '')

    if args.replicate_api_token is not None:
        tc.replicate_api_token = args.replicate_api_token
    elif not tc.replicate_api_token:
        tc.replicate_api_token = os.environ.get('REPLICATE_API_TOKEN', '')

    # Ollama host/port overrides
    if hasattr(args, 'ollama_host') and args.ollama_host is not None:
        tc.ollama_host = args.ollama_host
        config.codebook_embedding.ollama_host = args.ollama_host
    if hasattr(args, 'ollama_port') and args.ollama_port is not None:
        tc.ollama_port = args.ollama_port
        config.codebook_embedding.ollama_port = args.ollama_port

    # Embedding settings
    emb = config.codebook_embedding
    if args.no_two_pass:
        emb.two_pass = False
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

    return config


def _flatten_wizard_config(data: dict) -> dict:
    """Flatten wizard-format config (nested pipeline/theme_classification keys)
    into PipelineConfig.from_json-compatible dict."""
    result = {}

    # Lift pipeline-level keys to top level
    pipeline = data.get('pipeline', {})
    for key in ('transcript_dir', 'output_dir', 'trial_id', 'run_mode',
                'run_theme_labeler', 'run_codebook_classifier'):
        if key in pipeline:
            result[key] = pipeline[key]

    # Pass through sub-config dicts directly
    for key in ('segmentation', 'theme_classification', 'codebook_embedding',
                'codebook_llm', 'codebook_ensemble', 'validation', 'confidence_tiers'):
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

    if _prompt_yes_no_simple("Run pipeline now?", False):
        config = build_config_from_wizard_data(result['config_data'])
        framework_spec = result['config_data'].get('framework', {})
        framework = _load_framework(framework_spec.get('custom_path') or framework_spec.get('preset', 'vamr'))

        codebook = None
        if config.run_codebook_classifier:
            codebook_spec = result['config_data'].get('codebook', {})
            codebook = _load_codebook(codebook_spec.get('custom_path') or codebook_spec.get('preset', 'phenomenology'))

        _execute_pipeline(config, framework, codebook)


def cmd_run(args):
    """Execute pipeline (autonomous or with human validation)."""
    config = _build_config(args)

    # Load framework
    framework_arg = args.framework
    if framework_arg is None and args.config:
        with open(args.config) as f:
            file_data = json.load(f)
        fw = file_data.get('framework', {})
        framework_arg = fw.get('custom_path') or fw.get('preset', 'vamr')
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

    # HuggingFace preloading (only when explicitly selected)
    if config.theme_classification.backend == 'huggingface':
        _preload_huggingface_models(config)
    elif config.theme_classification.backend == 'ollama':
        _check_ollama_available(config)

    _execute_pipeline(config, framework, codebook)


def cmd_guided(args):
    """Execute pipeline with step-by-step educational narration."""
    config = _build_config(args)
    config.run_mode = 'interactive'  # guided always allows validation

    # Load framework
    framework_arg = args.framework
    if framework_arg is None and args.config:
        with open(args.config) as f:
            file_data = json.load(f)
        fw = file_data.get('framework', {})
        framework_arg = fw.get('custom_path') or fw.get('preset', 'vamr')
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

    # Use GuidedModeObserver
    from process.pipeline_hooks import GuidedModeObserver
    observer = GuidedModeObserver(config=config, framework=framework)

    # Always create validator for guided mode
    from process.human_validator import HumanValidator
    validator = HumanValidator(framework, codebook, skip_confirmation=False)

    _execute_pipeline(config, framework, codebook, observer=observer, validator=validator)


# =========================================================================
# Pipeline execution
# =========================================================================

def _execute_pipeline(config, framework, codebook=None, observer=None, validator=None):
    """Run the pipeline with the given config, framework, and optional hooks."""
    from process.orchestrator import run_full_pipeline
    from process.pipeline_hooks import SilentObserver

    if observer is None:
        observer = SilentObserver()

    # Create validator based on run mode if not provided
    if validator is None and config.run_mode in ('interactive', 'review'):
        from process.human_validator import HumanValidator
        validator = HumanValidator(
            framework, codebook,
            skip_confirmation=(config.run_mode == 'auto'),
        )

    # Print header
    print("\n" + "=" * 70)
    print("QRA CLASSIFICATION PIPELINE")
    print("=" * 70)
    print(f"  Mode:      {config.run_mode}")
    print(f"  Framework: {framework.name} ({framework.num_themes} themes)")
    print(f"  Model:     {config.theme_classification.model}")
    print(f"  Backend:   {config.theme_classification.backend}")
    if config.run_codebook_classifier:
        print(f"  Codebook:  enabled")
    print()

    master_df = run_full_pipeline(
        config, framework,
        codebook=codebook,
        observer=observer,
        validator=validator,
    )

    # Summary
    print(f"\nTotal segments: {len(master_df)}")
    if 'primary_stage' in master_df.columns:
        labeled = master_df[master_df['primary_stage'].notna()]
        print(f"Successfully labeled: {len(labeled)}")

    return master_df


def _check_ollama_available(config):
    """Verify Ollama is reachable and the configured model is available."""
    import requests
    tc = config.theme_classification
    base_url = f"http://{tc.ollama_host}:{tc.ollama_port}"
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        data = resp.json()
        available = [m['name'] for m in data.get('models', [])]
        model = tc.model
        # Ollama tags may include digest suffix; do prefix match
        model_base = model.split(':')[0]
        found = any(m.startswith(model_base) for m in available)
        if not found:
            print(f"\n  WARNING: Model '{model}' not found in Ollama.")
            print(f"  Available models: {available or '(none)'}")
            print(f"  Pull it with:  ollama pull {model}\n")
        else:
            print(f"  Ollama ready  ({base_url})  model: {model}")
    except Exception as e:
        print(f"\n  WARNING: Cannot reach Ollama at {base_url}: {e}")
        print(f"  Make sure Ollama is running:  ollama serve\n")


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
Setup (one-time):
  ollama pull mixtral:8x7b          # primary model (~26 GB, Q4)
  ollama pull mistral:7b-instruct   # secondary model for embedding classifier (~4 GB)

Examples:
  # Interactive setup wizard
  python qra.py setup

  # Run with local Ollama (default – mixtral:8x7b)
  python qra.py run

  # Run with a different local model
  python qra.py run --model mistral:7b-instruct

  # Run with two models for cross-referencing
  python qra.py run --models mixtral:8x7b mistral:7b-instruct

  # Run with saved config
  python qra.py run --config ./qra_config.json

  # Run with human validation
  python qra.py run --mode interactive

  # Guided mode with educational narration
  python qra.py guided

  # Cloud fallback (requires API key)
  python qra.py run --backend openrouter --model openai/gpt-4o --api-key $OPENROUTER_API_KEY
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
    run_parser.add_argument(
        '--mode', default='auto',
        choices=['auto', 'interactive', 'review'],
        help='Run mode (default: auto)',
    )
    _add_common_args(run_parser)

    # ---- guided ----
    guided_parser = subparsers.add_parser(
        'guided', help='Execute with step-by-step educational narration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_args(guided_parser)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == 'setup':
            cmd_setup(args)
        elif args.command == 'run':
            cmd_run(args)
        elif args.command == 'guided':
            cmd_guided(args)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
