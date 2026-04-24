"""
setup_wizard.py
---------------
Interactive 9-step configuration wizard for the QRA pipeline.

Walks the user through all pipeline settings and saves a config JSON file
that can later be loaded with ``qra run --config``.
"""

import json
import os
from typing import Optional, List

from constructs.theme_schema import ThemeFramework
from constructs.vamr import get_vamr_framework
from constructs.config import ThemeClassificationConfig
from codebook.config import EmbeddingClassifierConfig
from .config import (
    PipelineConfig,
    SegmentationConfig,
    ValidationConfig,
    TestSetConfig,
    ConfidenceTierConfig,
    TherapistCueConfig,
)
from .transcript_ingestion import scan_speakers


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


def _validate_speaker_anonymization_key(path: str) -> tuple[bool, Optional[str]]:
    """Validate speaker anonymization key file format.

    Returns (is_valid, error_message).
    Valid format: {speaker_name: {role: str, anonymized_id: str}, ...}
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return False, f"File not found: {path}"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"

    if not isinstance(data, dict):
        return False, "Root must be a JSON object"

    for speaker_name, entry in data.items():
        if not isinstance(entry, dict):
            return False, f"Entry for '{speaker_name}' is not a dict"

        if set(entry.keys()) != {'role', 'anonymized_id'}:
            return False, f"Entry for '{speaker_name}' must have exactly 'role' and 'anonymized_id' keys"

        role = entry.get('role')
        if role not in ('therapist', 'participant'):
            return False, f"Entry for '{speaker_name}': role must be 'therapist' or 'participant', got '{role}'"

        anon_id = entry.get('anonymized_id')
        if not isinstance(anon_id, str) or not anon_id.strip():
            return False, f"Entry for '{speaker_name}': anonymized_id must be a non-empty string"

    return True, None


class SetupWizard:
    """Interactive configuration wizard for the QRA pipeline."""

    def __init__(self):
        self.config_data = {}
        self.framework: Optional[ThemeFramework] = None
        self.custom_exemplars = {}

    def run(self) -> dict:
        """Run all wizard steps and return config dict."""
        print("\n" + "=" * 60)
        print("  QRA Pipeline Setup Wizard")
        print("=" * 60)
        print()

        self._step_1_paths()
        self._step_1b_speaker_key()
        self._step_2_speaker_filter()
        self._step_3_segmentation()
        self._step_4_backend()
        self._step_5_framework()
        self._step_6_exemplars()
        self._step_7_codebook()
        self._step_8_classification()
        self._step_9_confidence()
        self._step_10_testsets()
        self._step_11_analysis()
        self._step_11b_therapist_cues()
        config_path = self._step_12_save()

        return {
            'config_path': config_path,
            'config_data': self.config_data,
        }

    # -----------------------------------------------------------------
    # Step 1: Input/Output paths
    # -----------------------------------------------------------------
    def _step_1_paths(self):
        print("--- Step 1/12: Input/Output Paths ---")
        self.config_data['pipeline'] = {
            'transcript_dir': _prompt("Transcript directory", './data/input/'),
            'output_dir': _prompt("Output directory", './data/output/'),
            'trial_id': _prompt("Trial ID", 'standard'),
        }
        print()

    # -----------------------------------------------------------------
    # Step 1b: Speaker anonymization key import
    # -----------------------------------------------------------------
    def _step_1b_speaker_key(self):
        print("--- Step 1b/12: Speaker Anonymization Key ---")
        print("    Optionally import a pre-existing speaker ID mapping to keep")
        print("    participant IDs consistent across runs (e.g., Participant_MM001).")
        print("    New speakers not in the key will be assigned unknownparticipant_{N}.")
        print()

        transcript_dir = self.config_data.get('pipeline', {}).get('transcript_dir', './data/input/')
        default_key_path = os.path.join(transcript_dir, 'speaker_anonymization_key.json')

        # Check for default location
        key_path = None
        if os.path.isfile(default_key_path):
            print(f"    Found speaker key at: {default_key_path}")
            if _prompt_yes_no("Use this key?", default=True):
                is_valid, err = _validate_speaker_anonymization_key(default_key_path)
                if is_valid:
                    key_path = os.path.abspath(default_key_path)
                    print(f"    ✓ Key loaded successfully")
                else:
                    print(f"    ✗ Key validation failed: {err}")
                    print()

        # Offer custom path if no valid key yet
        if not key_path:
            if _prompt_yes_no("Import a key from another path?", default=False):
                while True:
                    custom_path = _prompt("Enter path to speaker_anonymization_key.json", "")
                    if not custom_path:
                        print("    Skipping key import.")
                        break
                    custom_path = os.path.expanduser(custom_path)
                    is_valid, err = _validate_speaker_anonymization_key(custom_path)
                    if is_valid:
                        key_path = os.path.abspath(custom_path)
                        print(f"    ✓ Key loaded successfully")
                        break
                    else:
                        print(f"    ✗ Validation failed: {err}")
                        print()

        # Store in config
        if key_path:
            self.config_data['pipeline']['speaker_anonymization_key_path'] = key_path
            print()
        else:
            print("    Speaker IDs will be auto-generated as participant_N.")
            print()

    # -----------------------------------------------------------------
    # Step 2: Speaker role identification
    # -----------------------------------------------------------------

    # Default speakers to pre-select as therapy facilitators (Move-MORE study)
    _DEFAULT_THERAPISTS = [ 'Move-MORE Study','Instructor', 'Anand', 'Lani', 'Wade (Study Coordinator)', 'Rebecca Heron', 'Wade Balsamo (Study Coordinator)', 'Michelle Berg']

    def _step_2_speaker_filter(self):
        print("--- Step 2/12: Speaker Role Identification ---")
        print("    Identify which speakers are therapists/facilitators and which")
        print("    are participants. Therapist dialogue is preserved as read-only")
        print("    conversational context for adjacent participant segments.")
        print()

        # Scan transcripts to discover speakers
        transcript_dir = self.config_data.get('pipeline', {}).get('transcript_dir', './data/input/')
        discovered: dict = {}
        if os.path.isdir(transcript_dir):
            print(f"    Scanning transcripts in {transcript_dir} ...")
            try:
                discovered = scan_speakers(transcript_dir)
            except Exception as e:
                print(f"    Warning: Could not scan transcripts: {e}")

        if discovered:
            print(f"    Found {len(discovered)} speakers:")
            for i, (name, count) in enumerate(discovered.items(), 1):
                default_tag = " [therapist]" if name in self._DEFAULT_THERAPISTS else " [participant]"
                print(f"      {i:2d}. {name} ({count} utterances){default_tag}")
            print()
        else:
            print("    No transcripts found to scan — enter speaker names manually.")
            print()

        # Build default therapist selection
        if discovered:
            defaults = [n for n in self._DEFAULT_THERAPISTS if n in discovered]
        else:
            defaults = list(self._DEFAULT_THERAPISTS)

        if defaults:
            print("    Default therapist/facilitator speakers:")
            for name in defaults:
                print(f"      - {name}")
            use_defaults = _prompt_yes_no("Use these defaults?", True)
            if use_defaults:
                therapists = list(defaults)
            else:
                therapists = self._prompt_therapist_selection(discovered)
        else:
            therapists = self._prompt_therapist_selection(discovered)

        if not therapists:
            print("    No therapists identified — all speakers treated as participants.")

        # Option to exclude therapist utterances from classification
        exclude_from_classification = True
        if therapists:
            print()
            print("    Therapist utterances can be excluded from classification")
            print("    while still being used as conversational context.")
            exclude_from_classification = _prompt_yes_no(
                "Exclude therapist utterances from classification?", True
            )

        if therapists and exclude_from_classification:
            print(f"    Therapists (excluded from classification): {therapists}")
            self.config_data['speaker_filter'] = {
                'mode': 'exclude',
                'speakers': therapists,
            }
        elif therapists:
            print(f"    Therapists (included in classification): {therapists}")
            self.config_data['speaker_filter'] = {
                'mode': 'none',
                'speakers': therapists,
            }
        else:
            self.config_data['speaker_filter'] = {'mode': 'none', 'speakers': []}
        print()

    def _prompt_therapist_selection(self, discovered: dict) -> List[str]:
        """Let the user select therapist speakers from discovered list or enter manually."""
        speakers: List[str] = []

        if discovered:
            speaker_names = list(discovered.keys())
            print("    Enter speaker numbers for THERAPISTS (comma-separated),")
            print("    or type names manually. Blank line when done:")

            while True:
                raw = input("      > ").strip()
                if not raw:
                    break
                # Try parsing as comma-separated numbers
                if all(part.strip().isdigit() for part in raw.split(',')):
                    for part in raw.split(','):
                        idx = int(part.strip()) - 1
                        if 0 <= idx < len(speaker_names):
                            name = speaker_names[idx]
                            if name not in speakers:
                                speakers.append(name)
                                print(f"        + {name} (therapist)")
                        else:
                            print(f"        Invalid number: {part.strip()}")
                else:
                    if raw not in speakers:
                        speakers.append(raw)
                        print(f"        + {raw} (therapist)")
        else:
            print("    Enter therapist/facilitator speaker labels (one per line, blank when done):")
            while True:
                name = input("      > ").strip()
                if not name:
                    break
                if name not in speakers:
                    speakers.append(name)

        return speakers

    # -----------------------------------------------------------------
    # Step 3: Advanced segmentation parameters
    # -----------------------------------------------------------------
    def _step_3_segmentation(self):
        print("--- Step 3/12: Segmentation Parameters ---")

        if not _prompt_yes_no("Configure advanced segmentation options?", False):
            # Use defaults
            self.config_data['segmentation'] = {
                'max_gap_seconds': 15.0,
                'min_words_per_sentence': 20,
                'max_segment_duration_seconds': 60.0,
                'min_segment_words_conversational': 60,
                'max_segment_words_conversational': 500,
                'use_adaptive_threshold': True,
                'min_prominence': 0.05,
                'use_topic_clustering': True,
                'use_llm_refinement': True,
                'llm_refinement_mode': 'full',
            }
            print("    Using defaults: single-speaker segments, 15s max gap,")
            print("    20-word min per sentence (shorter folded into neighbors),")
            print("    5min max segment duration, 60-500 word segments,")
            print("    adaptive threshold + LLM refinement enabled.")
            print()
            return

        max_gap = _prompt_float("Max time gap (seconds) between utterances to group", 15.0)
        min_words_sent = _prompt_int("Min words per sentence (shorter are folded into neighbors)", 20)
        max_duration = _prompt_float("Max segment duration (seconds)", 60.0)
        min_words = _prompt_int("Min words per segment", 60)
        max_words = _prompt_int("Max words per segment", 500)

        # Adaptive threshold
        use_adaptive = _prompt_yes_no(
            "Use adaptive similarity threshold (local minima detection)?", True
        )
        min_prominence = 0.05
        if use_adaptive:
            min_prominence = _prompt_float("Min prominence for similarity dips", 0.05)

        # Topic clustering
        use_clustering = _prompt_yes_no(
            "Use topic clustering for additional boundary detection?", True
        )

        # LLM refinement
        use_llm_refine = _prompt_yes_no(
            "Use LLM-assisted segmentation refinement (boundary review + context selection)?", True
        )
        llm_refine_mode = 'full'
        if use_llm_refine:
            print("    Refinement modes:")
            print("      boundary_review     : Re-evaluate ambiguous boundaries")
            print("      context_expansion   : Expand segments with surrounding dialogue")
            print("      coherence_check     : Split oversized segments at natural boundaries")
            print("      full                : All three passes")
            llm_refine_mode = _prompt_choice(
                "Refinement mode",
                ['boundary_review', 'context_expansion', 'coherence_check', 'full'],
                'full',
            )

        self.config_data['segmentation'] = {
            'max_gap_seconds': max_gap,
            'min_words_per_sentence': min_words_sent,
            'max_segment_duration_seconds': max_duration,
            'min_segment_words_conversational': min_words,
            'max_segment_words_conversational': max_words,
            'use_adaptive_threshold': use_adaptive,
            'min_prominence': min_prominence,
            'use_topic_clustering': use_clustering,
            'use_llm_refinement': use_llm_refine,
            'llm_refinement_mode': llm_refine_mode,
        }
        print()

    # -----------------------------------------------------------------
    # Step 4: Backend & model
    # -----------------------------------------------------------------
    def _step_4_backend(self):
        print("--- Step 4/12: Backend & Model ---")
        backend = _prompt_choice(
            "Backend",
            ['openrouter', 'replicate', 'huggingface', 'ollama', 'lmstudio'],
            'lmstudio',
        )

        # LM Studio: prompt for server URL and model name
        if backend == 'lmstudio':
            lmstudio_url = _prompt(
                "LM Studio server URL", 'http://10.0.0.58:1234/v1'
            )
            model = _prompt("Primary model (used for segmentation, classification, and all LLM calls)", 'nvidia/nemotron-3-super')
            print(f"    LM Studio backend: {lmstudio_url}")
            self.config_data['theme_classification'] = {
                'backend': 'lmstudio',
                'model': model,
                'lmstudio_base_url': lmstudio_url,
            }
            print()
            return

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
    # Step 5: Framework selection
    # -----------------------------------------------------------------
    def _step_5_framework(self):
        print("--- Step 5/12: Theme Framework ---")
        choice = _prompt_choice("Framework", ['vamr', 'custom'], 'vamr')

        if choice == 'vamr':
            self.framework = get_vamr_framework()
            self.config_data['framework'] = {'preset': 'vamr'}
        else:
            path = _prompt("Path to custom framework JSON")
            self.config_data['framework'] = {'custom_path': path}
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
                print("    Skipping exemplar customization.")

        if self.framework:
            print(f"    Framework: {self.framework.name} ({self.framework.num_themes} themes)")
        print()

    # -----------------------------------------------------------------
    # Step 6: Exemplar utterances
    # -----------------------------------------------------------------
    def _step_6_exemplars(self):
        print("--- Step 6/12: Exemplar Utterances ---")
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
    # Step 7: Codebook selection
    # -----------------------------------------------------------------
    def _step_7_codebook(self):
        print("--- Step 7/12: Codebook Classification ---")
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

        print()
        print("    Codebook classification uses two complementary methods:")
        print("    1. Embedding similarity  — sentence-transformer model (configurable below)")
        print("       Segments are encoded as queries; codebook entries as passages")
        print("       (asymmetric encoding for better construct retrieval).")
        print("    2. LLM zero-shot prompt  — uses your configured primary model")
        print("    Both results are reconciled by an ensemble step.")
        print()

        print("    Embedding model options:")
        print("      Qwen/Qwen3-Embedding-8B  — 4096-dim, best quality, ~16 GB download (default)")
        print("      all-MiniLM-L6-v2         — 384-dim, lightweight, 90 MB, no download needed")
        print()
        embedding_model = _prompt("Embedding model", 'Qwen/Qwen3-Embedding-8B')
        two_pass = _prompt_yes_no("Enable two-pass embedding classification?", True)
        self.config_data['codebook_embedding'] = {
            'embedding_model': embedding_model,
            'two_pass': two_pass,
        }

        exemplar_path = _prompt("Exemplar import path (blank for none)", '')
        if exemplar_path:
            self.config_data['codebook_embedding']['exemplar_import_path'] = exemplar_path

        print()

    # Default LM Studio models for per-run interrater assignment
    _DEFAULT_LMSTUDIO_RUN_MODELS = [
        'google/gemma-4-31b',
        'qwen/qwen3-next-80b',
    ]

    # -----------------------------------------------------------------
    # Step 8: Classification parameters
    # -----------------------------------------------------------------
    def _step_8_classification(self):
        print("--- Step 8/12: Classification Parameters ---")
        n_runs = _prompt_int("Number of classification runs per segment", 3)
        temperature = _prompt_float("Temperature", 0.1)

        self.config_data['theme_classification']['n_runs'] = n_runs
        self.config_data['theme_classification']['temperature'] = temperature

        # When n_runs > 1, offer adversarial checker models for interrater reliability.
        # Run 1 is always the primary model; we only ask for the additional checkers.
        if n_runs >= 2:
            self._configure_per_run_models(n_runs)

        print()

    def _configure_per_run_models(self, n_runs: int):
        """Configure adversarial checker models for interrater reliability.

        Run 1 is always the primary model (set in Step 4).  The user is only
        asked to supply the n_runs-1 additional checker models.  The majority-vote
        consistency score then reflects genuine cross-model agreement rather than
        stochastic variation within one model.
        """
        primary_model = self.config_data.get('theme_classification', {}).get('model', '')
        n_checkers = n_runs - 1

        # Build checker defaults from the LM Studio preset list, excluding primary
        checker_defaults: List[str] = [
            m for m in self._DEFAULT_LMSTUDIO_RUN_MODELS if m != primary_model
        ]
        while len(checker_defaults) < n_checkers:
            checker_defaults.append(checker_defaults[-1] if checker_defaults else primary_model)

        print()
        print(f"    Interrater reliability: {n_runs} runs total")
        print(f"      Run 1: {primary_model}  (primary model)")
        if n_checkers > 0:
            print(f"    Enter {n_checkers} additional checker model(s).")
            print("    Each acts as an independent rater; majority-vote consistency reflects")
            print("    genuine cross-model agreement, not within-model stochasticity.")
        print()

        per_run_models: List[str] = [primary_model]
        for i in range(n_checkers):
            default = checker_defaults[i] if i < len(checker_defaults) else ''
            model = _prompt(f"Checker model {i + 1}", default)
            per_run_models.append(model)

        if len(set(per_run_models)) < n_runs:
            print("    Warning: duplicate models — runs sharing a model will not provide")
            print("    independent ratings. Consider using distinct models.")

        self.config_data['theme_classification']['per_run_models'] = per_run_models
        print(f"    Configured {n_runs} rater models:")
        for i, m in enumerate(per_run_models):
            label = " (primary)" if i == 0 else f" (checker {i})"
            print(f"      Run {i + 1}: {m}{label}")

    # -----------------------------------------------------------------
    # Step 9: Confidence thresholds
    # -----------------------------------------------------------------
    def _step_9_confidence(self):
        print("--- Step 9/12: Confidence Thresholds ---")
        self.config_data['confidence_tiers'] = {
            'high_confidence': _prompt_float("High confidence threshold", 0.8),
            'medium_min_confidence': _prompt_float("Medium confidence threshold", 0.6),
        }
        print()


    # -----------------------------------------------------------------
    # Step 10: Validation Test Sets
    # -----------------------------------------------------------------
    def _step_10_testsets(self):
        print("--- Step 10/12: Validation Test Sets ---")
        print("    Cross-session test sets let human raters independently code a random")
        print("    sample of participant segments drawn proportionally from all cohorts,")
        print("    for inter-rater reliability comparison against AI classifications.")
        print()
        enable = _prompt_yes_no("Generate validation test sets?", True)
        self.config_data['test_sets'] = {'enabled': enable}
        if not enable:
            print()
            return
        n_sets = _prompt_int("Number of test sets", 2)
        fraction = _prompt_float("Fraction of segments per set (e.g. 0.10 = 10%)", 0.10)
        if n_sets * fraction > 1.0:
            fraction = round(1.0 / n_sets, 2)
            print(f"    Adjusted fraction to {fraction} so sets don't overlap.")
        self.config_data['test_sets'].update({
            'n_sets': n_sets,
            'fraction_per_set': fraction,
            'random_seed': 42,
        })
        print(f"    Will produce {n_sets} test sets × ~{fraction:.0%} of participant segments, stratified by cohort.")
        print()

    # -----------------------------------------------------------------
    # Step 11: Post-pipeline Analysis
    # -----------------------------------------------------------------
    def _step_11_analysis(self):
        print("--- Step 11/12: Post-Pipeline Analysis ---")
        print("    After the pipeline completes, the analysis module can generate:")
        print("    - Per-participant longitudinal reports (VA-MR progression)")
        print("    - Per-session summaries with prototypical exemplars")
        print("    - Per-construct (stage + codebook) analyses")
        print("    - Graph-ready CSVs for R/Python visualization")
        print("    - Feasibility and validity assessment")
        print()
        auto_analyze = _prompt_yes_no(
            "Automatically run analysis after pipeline completes?", True
        )
        self.config_data['pipeline']['auto_analyze'] = auto_analyze
        if auto_analyze:
            print("    Analysis will run automatically at the end of the pipeline.")
        else:
            print("    Analysis can be run manually: python qra.py analyze --output-dir ./data/output/")
        print()

        print()

    # -----------------------------------------------------------------
    # Step 11b: Therapist cue summarization
    # -----------------------------------------------------------------
    def _step_11b_therapist_cues(self):
        print("--- Step 11b/12: Therapist Cue Summarization ---")
        print("    When enabled, therapist dialogue between two participant segments")
        print("    is surfaced as a CUE in state_transition_explanation.txt, and")
        print("    cue_response.txt is generated with averaged cues by transition type.")
        print()
        enable = _prompt_yes_no("Enable therapist cue summarization in transition analysis?", True)
        if not enable:
            self.config_data['therapist_cues'] = {'enabled': False}
            print()
            return

        max_per_cue = _prompt_int(
            "Max words per inline cue (longer cues will be LLM-summarized)", 250
        )
        max_agg = _prompt_int(
            "Max words per averaged block in cue_response.txt", 1000
        )
        self.config_data['therapist_cues'] = {
            'enabled': True,
            'max_length_per_cue': max_per_cue,
            'max_length_of_average_cue_responses': max_agg,
        }
        print()

    # -----------------------------------------------------------------
    # Step 12: Save & run
    # -----------------------------------------------------------------
    def _step_12_save(self) -> str:
        print("--- Step 12/12: Save Configuration ---")
        from . import output_paths as _paths
        default_path = os.path.join(
            _paths.meta_dir(self.config_data['pipeline'].get('output_dir', './data/output/')),
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
    from .config import SpeakerFilterConfig

    pipeline = data.get('pipeline', {})
    tc = data.get('theme_classification', {})
    cb_emb = data.get('codebook_embedding', {})
    ct = data.get('confidence_tiers', {})
    sf = data.get('speaker_filter', {})
    seg = data.get('segmentation', {})

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
        run_theme_labeler=pipeline.get('run_theme_labeler', True),
        run_codebook_classifier=pipeline.get('run_codebook_classifier', False),
        speaker_anonymization_key_path=pipeline.get('speaker_anonymization_key_path'),
        auto_analyze=pipeline.get('auto_analyze', True),
        segmentation=SegmentationConfig(
            min_segment_words_conversational=seg.get('min_segment_words_conversational', 60),
            max_segment_words_conversational=seg.get('max_segment_words_conversational', 300),
            max_gap_seconds=seg.get('max_gap_seconds', 15.0),
            min_words_per_sentence=seg.get('min_words_per_sentence', 20),
            max_segment_duration_seconds=seg.get('max_segment_duration_seconds', 60.0),
            use_adaptive_threshold=seg.get('use_adaptive_threshold', True),
            min_prominence=seg.get('min_prominence', 0.05),
            broad_window_size=seg.get('broad_window_size', 7),
            use_topic_clustering=seg.get('use_topic_clustering', False),
            use_llm_refinement=seg.get('use_llm_refinement', False),
            llm_refinement_mode=seg.get('llm_refinement_mode', 'boundary_review'),
            llm_ambiguity_threshold=seg.get('llm_ambiguity_threshold', 0.15),
            llm_batch_size=seg.get('llm_batch_size', 5),
        ),
        speaker_filter=SpeakerFilterConfig(
            mode=sf.get('mode', 'none'),
            speakers=sf.get('speakers', []),
        ),
        theme_classification=ThemeClassificationConfig(
            backend=backend,
            model=tc.get('model', 'openai/gpt-4o'),
            models=tc.get('models', []),
            per_run_models=tc.get('per_run_models', []),
            n_runs=tc.get('n_runs', 3),
            temperature=tc.get('temperature', 0.0),
            api_key=api_key,
            replicate_api_token=replicate_token,
            lmstudio_base_url=tc.get('lmstudio_base_url', 'http://127.0.0.1:1234/'),
        ),
        codebook_embedding=EmbeddingClassifierConfig(
            two_pass=cb_emb.get('two_pass', True),
            embedding_model=cb_emb.get('embedding_model', 'Qwen/Qwen3-Embedding-8B'),
            exemplar_import_path=cb_emb.get('exemplar_import_path'),
        ),
        validation=ValidationConfig(),
        test_sets=TestSetConfig(
            enabled=data.get('test_sets', {}).get('enabled', True),
            n_sets=data.get('test_sets', {}).get('n_sets', 2),
            fraction_per_set=data.get('test_sets', {}).get('fraction_per_set', 0.10),
            random_seed=data.get('test_sets', {}).get('random_seed', 42),
        ),
        confidence_tiers=ConfidenceTierConfig(
            high_confidence=ct.get('high_confidence', 0.8),
            medium_min_confidence=ct.get('medium_min_confidence', 0.6),
        ),
        therapist_cues=TherapistCueConfig(
            **{k: v for k, v in data.get('therapist_cues', {}).items()
               if k in ('enabled', 'max_length_per_cue', 'max_length_of_average_cue_responses')}
        ),
    )

    return config
