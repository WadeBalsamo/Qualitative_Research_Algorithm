"""
setup_wizard.py
---------------
Interactive 9-step configuration wizard for the QRA pipeline.

Walks the user through all pipeline settings and saves a config JSON file
that can later be loaded with ``qra run --config``.
"""

import json
import os
from typing import Optional

from constructs.theme_schema import ThemeFramework
from constructs.vamr import get_vamr_framework
from constructs.config import ThemeClassificationConfig
from codebook.config import EmbeddingClassifierConfig, LLMCodebookConfig, EnsembleConfig
from .config import (
    PipelineConfig,
    SegmentationConfig,
    ValidationConfig,
    ConfidenceTierConfig,
)


def _prompt(label: str, default: str = '') -> str:
    """Prompt user for input with an optional default."""
    if default:
        raw = input(f"  {label} [{default}]: ").strip()
        return raw if raw else default
    return input(f"  {label}: ").strip()


def _prompt_float(label: str, default: float) -> float:
    raw = _prompt(label, str(default))
    try:
        return float(raw)
    except ValueError:
        print(f"    Invalid number, using default: {default}")
        return default


def _prompt_int(label: str, default: int) -> int:
    raw = _prompt(label, str(default))
    try:
        return int(raw)
    except ValueError:
        print(f"    Invalid number, using default: {default}")
        return default


def _prompt_choice(label: str, choices: list, default: str = '') -> str:
    choices_str = '/'.join(choices)
    raw = _prompt(f"{label} ({choices_str})", default).lower()
    if raw in choices:
        return raw
    print(f"    Invalid choice, using default: {default}")
    return default


def _prompt_yes_no(label: str, default: bool = True) -> bool:
    default_str = 'Y/n' if default else 'y/N'
    raw = input(f"  {label} [{default_str}]: ").strip().lower()
    if not raw:
        return default
    return raw in ('y', 'yes')


class SetupWizard:
    """Interactive configuration wizard for the QRA pipeline."""

    def __init__(self):
        self.config_data = {}
        self.framework: Optional[ThemeFramework] = None
        self.custom_exemplars = {}

    def run(self) -> dict:
        """Run all 9 wizard steps and return config dict."""
        print("\n" + "=" * 60)
        print("  QRA Pipeline Setup Wizard")
        print("=" * 60)
        print()

        self._step_1_paths()
        self._step_2_backend()
        self._step_3_framework()
        self._step_4_exemplars()
        self._step_5_codebook()
        self._step_6_classification()
        self._step_7_confidence()
        self._step_8_run_mode()
        config_path = self._step_9_save()

        return {
            'config_path': config_path,
            'config_data': self.config_data,
        }

    # -----------------------------------------------------------------
    # Step 1: Input/Output paths
    # -----------------------------------------------------------------
    def _step_1_paths(self):
        print("--- Step 1/9: Input/Output Paths ---")
        self.config_data['pipeline'] = {
            'transcript_dir': _prompt("Transcript directory", './data/input/diarized_sessions/'),
            'output_dir': _prompt("Output directory", './data/output/'),
            'trial_id': _prompt("Trial ID", 'standard'),
        }
        print()

    # -----------------------------------------------------------------
    # Step 2: Backend & model
    # -----------------------------------------------------------------
    def _step_2_backend(self):
        print("--- Step 2/9: Backend & Model ---")
        backend = _prompt_choice("Backend", ['openrouter', 'replicate', 'huggingface', 'ollama'], 'openrouter')

        model = _prompt("Model ID", 'openai/gpt-4o')

        # API key resolution (from env, not saved)
        if backend == 'openrouter':
            env_key = os.environ.get('OPENROUTER_API_KEY', '')
            if not env_key:
                print("    Note: Set OPENROUTER_API_KEY environment variable before running.")
        elif backend == 'replicate':
            env_key = os.environ.get('REPLICATE_API_TOKEN', '')
            if not env_key:
                print("    Note: Set REPLICATE_API_TOKEN environment variable before running.")

        # Multi-model option
        models = []
        if _prompt_yes_no("Use multiple models for cross-referencing?", False):
            print("    Enter model IDs one per line (blank line to finish):")
            models.append(model)  # Primary model is always first
            while True:
                m = input("    > ").strip()
                if not m:
                    break
                if m not in models:
                    models.append(m)

        self.config_data['theme_classification'] = {
            'backend': backend,
            'model': model,
        }
        if models:
            self.config_data['theme_classification']['models'] = models
        print()

    # -----------------------------------------------------------------
    # Step 3: Framework selection
    # -----------------------------------------------------------------
    def _step_3_framework(self):
        print("--- Step 3/9: Theme Framework ---")
        choice = _prompt_choice("Framework", ['vamr', 'custom'], 'vamr')

        if choice == 'vamr':
            self.framework = get_vamr_framework()
            self.config_data['framework'] = {'preset': 'vamr'}
        else:
            path = _prompt("Path to custom framework JSON")
            self.config_data['framework'] = {'custom_path': path}
            # Try to load the custom framework for use in step 4
            try:
                with open(path) as f:
                    fw_data = json.load(f)
                from constructs.theme_schema import ThemeDefinition
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
                self.framework = ThemeFramework(
                    name=fw_data.get('framework', 'custom'),
                    version=fw_data.get('version', '1.0'),
                    description=fw_data.get('description', ''),
                    themes=themes,
                )
            except Exception as e:
                print(f"    Warning: Could not load framework from {path}: {e}")
                print("    Skipping exemplar customization in step 4.")

        if self.framework:
            print(f"    Framework: {self.framework.name} ({self.framework.num_themes} themes)")
        print()

    # -----------------------------------------------------------------
    # Step 4: Exemplar utterances
    # -----------------------------------------------------------------
    def _step_4_exemplars(self):
        print("--- Step 4/9: Exemplar Utterances ---")
        if not self.framework:
            print("    No framework loaded; skipping exemplar customization.")
            print()
            return

        if not _prompt_yes_no("Customize exemplar utterances for themes?", False):
            print("    Using default exemplars.")
            print()
            return

        self.custom_exemplars = {}
        for theme in self.framework.themes:
            print(f"\n  Theme {theme.theme_id}: {theme.short_name}")
            print(f"    Definition: {theme.definition[:80]}...")
            if theme.exemplar_utterances:
                print(f"    Current exemplars ({len(theme.exemplar_utterances)}):")
                for ex in theme.exemplar_utterances[:3]:
                    print(f"      - {ex[:60]}...")

            print("    Enter custom exemplar utterances (one per line, blank to finish):")
            custom = []
            while True:
                line = input("      > ").strip()
                if not line:
                    break
                custom.append(line)

            if custom:
                self.custom_exemplars[str(theme.theme_id)] = custom
                print(f"    Added {len(custom)} custom exemplar(s)")

        if self.custom_exemplars:
            self.config_data.setdefault('framework', {})['custom_exemplars'] = self.custom_exemplars
        print()

    # -----------------------------------------------------------------
    # Step 5: Codebook selection
    # -----------------------------------------------------------------
    def _step_5_codebook(self):
        print("--- Step 5/9: Codebook Classification ---")
        enable = _prompt_yes_no("Enable codebook classification?", False)
        self.config_data['pipeline']['run_codebook_classifier'] = enable

        if not enable:
            print()
            return

        codebook_choice = _prompt_choice("Codebook", ['phenomenology', 'custom'], 'phenomenology')
        if codebook_choice == 'phenomenology':
            self.config_data['codebook'] = {'preset': 'phenomenology'}
        else:
            path = _prompt("Path to custom codebook JSON")
            self.config_data['codebook'] = {'custom_path': path}

        two_pass = _prompt_yes_no("Enable two-pass embedding classification?", True)
        self.config_data['codebook_embedding'] = {'two_pass': two_pass}

        exemplar_path = _prompt("Exemplar import path (blank for none)", '')
        if exemplar_path:
            self.config_data['codebook_embedding']['exemplar_import_path'] = exemplar_path

        print()

    # -----------------------------------------------------------------
    # Step 6: Classification parameters
    # -----------------------------------------------------------------
    def _step_6_classification(self):
        print("--- Step 6/9: Classification Parameters ---")
        n_runs = _prompt_int("Number of runs per segment", 3)
        temperature = _prompt_float("Temperature", 0.0)

        self.config_data['theme_classification']['n_runs'] = n_runs
        self.config_data['theme_classification']['temperature'] = temperature
        print()

    # -----------------------------------------------------------------
    # Step 7: Confidence thresholds
    # -----------------------------------------------------------------
    def _step_7_confidence(self):
        print("--- Step 7/9: Confidence Thresholds ---")
        self.config_data['confidence_tiers'] = {
            'high_confidence': _prompt_float("High confidence threshold", 0.8),
            'medium_min_confidence': _prompt_float("Medium confidence threshold", 0.6),
        }
        print()

    # -----------------------------------------------------------------
    # Step 8: Run mode
    # -----------------------------------------------------------------
    def _step_8_run_mode(self):
        print("--- Step 8/9: Run Mode ---")
        print("    auto        : Fully automated (no human intervention)")
        print("    interactive : Prompt for validation of uncertain results")
        print("    review      : Batch validation at end")
        mode = _prompt_choice("Run mode", ['auto', 'interactive', 'review'], 'auto')
        self.config_data['pipeline']['run_mode'] = mode
        print()

    # -----------------------------------------------------------------
    # Step 9: Save & run
    # -----------------------------------------------------------------
    def _step_9_save(self) -> str:
        print("--- Step 9/9: Save Configuration ---")
        default_path = os.path.join(
            self.config_data['pipeline'].get('output_dir', './data/output/'),
            'qra_config.json',
        )
        save_path = _prompt("Save config to", default_path)

        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.config_data, f, indent=2)
        print(f"    Configuration saved to: {save_path}")
        print()

        return save_path


def build_config_from_wizard_data(data: dict) -> PipelineConfig:
    """Convert wizard output dict into a PipelineConfig instance."""
    pipeline = data.get('pipeline', {})
    tc = data.get('theme_classification', {})
    cb_emb = data.get('codebook_embedding', {})
    ct = data.get('confidence_tiers', {})

    # Resolve API credentials from environment
    backend = tc.get('backend', 'openrouter')
    api_key = ''
    replicate_token = ''
    if backend == 'openrouter':
        api_key = os.environ.get('OPENROUTER_API_KEY', '')
    elif backend == 'replicate':
        replicate_token = os.environ.get('REPLICATE_API_TOKEN', '')

    config = PipelineConfig(
        transcript_dir=pipeline.get('transcript_dir', './data/input/diarized_sessions/'),
        trial_id=pipeline.get('trial_id', 'standard'),
        output_dir=pipeline.get('output_dir', './data/output/'),
        run_mode=pipeline.get('run_mode', 'auto'),
        run_theme_labeler=pipeline.get('run_theme_labeler', True),
        run_codebook_classifier=pipeline.get('run_codebook_classifier', False),
        segmentation=SegmentationConfig(),
        theme_classification=ThemeClassificationConfig(
            backend=backend,
            model=tc.get('model', 'openai/gpt-4o'),
            models=tc.get('models', []),
            n_runs=tc.get('n_runs', 3),
            temperature=tc.get('temperature', 0.0),
            api_key=api_key,
            replicate_api_token=replicate_token,
        ),
        codebook_embedding=EmbeddingClassifierConfig(
            two_pass=cb_emb.get('two_pass', True),
            exemplar_import_path=cb_emb.get('exemplar_import_path'),
        ),
        validation=ValidationConfig(),
        confidence_tiers=ConfidenceTierConfig(
            high_confidence=ct.get('high_confidence', 0.8),
            medium_min_confidence=ct.get('medium_min_confidence', 0.6),
        ),
    )

    return config
