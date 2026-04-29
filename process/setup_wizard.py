"""
setup_wizard.py
---------------
Interactive configuration wizard for the QRA pipeline.

Offers three entry points:
  1. Small / Test  — lightweight models, MiniLM embeddings, fast iteration
  2. Production    — large models, Qwen3-8B embeddings, research-grade output
  3. Custom        — full step-by-step walkthrough with parameter explanations

The resulting config JSON can be loaded with ``qra run --config``.
"""

import json
import os
from typing import Optional, List

from theme_framework.theme_schema import ThemeFramework
from theme_framework.vammr import get_vammr_framework
from theme_framework.config import ThemeClassificationConfig
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


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

_PRESET_SMALL = {
    'label': 'Small / Test',
    'description': (
        'Lightweight models for rapid iteration and development testing.\n'
        '  Segmentation embedding : all-MiniLM-L6-v2  (384-dim, 90 MB, no download)\n'
        '  Classification models  : 3-run interrater ensemble\n'
        '    Run 1: nvidia/nemotron-3-nano-4b  (primary)\n'
        '    Run 2: google/gemma-4-e2b\n'
        '    Run 3: qwen/qwen3-8b\n'
        '  Framework              : VAMMR\n'
        '  Codebook classification: disabled'
    ),
    'segmentation_embedding_model': 'all-MiniLM-L6-v2',
    'primary_model': 'nvidia/nemotron-3-nano-4b',
    'per_run_models': [
        'nvidia/nemotron-3-nano-4b',
        'google/gemma-4-e2b',
        'qwen/qwen3-8b',
    ],
    'n_runs': 3,
    'temperature': 0.1,
    'framework': 'vammr',
    'run_codebook_classifier': False,
}

_PRESET_PRODUCTION = {
    'label': 'Production',
    'description': (
        'High-accuracy large models for research-grade classification.\n'
        '  Segmentation embedding : Qwen/Qwen3-Embedding-8B  (4096-dim, ~16 GB)\n'
        '  Classification models  : 3-run interrater ensemble\n'
        '    Run 1: qwen/qwen3-next-80b  (primary)\n'
        '    Run 2: google/gemma-4-31b\n'
        '    Run 3: nvidia/nemotron-3-super\n'
        '  Framework              : VAMMR\n'
        '  Codebook classification: disabled'
    ),
    'segmentation_embedding_model': 'Qwen/Qwen3-Embedding-8B',
    'primary_model': 'qwen/qwen3-next-80b',
    'per_run_models': [
        'qwen/qwen3-next-80b',
        'google/gemma-4-31b',
        'nvidia/nemotron-3-super',
    ],
    'n_runs': 3,
    'temperature': 0.1,
    'framework': 'vammr',
    'run_codebook_classifier': False,
}

_EMBEDDING_MODEL_OPTIONS = {
    'minilm': 'all-MiniLM-L6-v2',
    'qwen8b': 'Qwen/Qwen3-Embedding-8B',
}

_EMBEDDING_MODEL_DESCRIPTIONS = (
    '  all-MiniLM-L6-v2        — 384-dim, 90 MB, no download needed, fast\n'
    '                            Good for testing and lower-resource machines.\n'
    '  Qwen/Qwen3-Embedding-8B — 4096-dim, ~16 GB download, state-of-the-art\n'
    '                            Recommended for production research runs.'
)


# ---------------------------------------------------------------------------
# Helper prompts
# ---------------------------------------------------------------------------

def _prompt(label: str, default: str = '') -> str:
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
        self._preset_mode: str = 'custom'  # 'small', 'production', or 'custom'

    def run(self) -> dict:
        """Run all wizard steps and return config dict."""
        print("\n" + "=" * 60)
        print("  QRA Pipeline Setup Wizard")
        print("=" * 60)
        print()

        self._step_0_preset()

        self._step_1_paths()
        self._step_1b_speaker_key()
        self._step_2_speaker_filter()

        if self._preset_mode == 'custom':
            self._step_3_segmentation()
            self._step_4_backend()
            self._step_5_framework()
            self._step_6_exemplars()
            self._step_7_codebook()
            self._step_8_classification()
            self._step_9_confidence()
        else:
            self._step_preset_backend()
            self._apply_preset_settings()

        self._step_10_testsets()
        self._step_11_analysis()
        self._step_11b_therapist_cues()
        config_path = self._step_12_save()

        return {
            'config_path': config_path,
            'config_data': self.config_data,
        }

    # -----------------------------------------------------------------
    # Step 0: Preset selection
    # -----------------------------------------------------------------
    def _step_0_preset(self):
        print("--- Step 0: Configuration Mode ---")
        print()
        print("  Choose a configuration preset or walk through full custom setup:")
        print()
        print("  [1] Small / Test")
        for line in _PRESET_SMALL['description'].splitlines():
            print(f"      {line}")
        print()
        print("  [2] Production")
        for line in _PRESET_PRODUCTION['description'].splitlines():
            print(f"      {line}")
        print()
        print("  [3] Custom")
        print("      Full step-by-step walkthrough with explanations for every")
        print("      hyperparameter. Enables complete control over all settings.")
        print()

        while True:
            choice = input("  Select mode [1/2/3]: ").strip()
            if choice == '1':
                self._preset_mode = 'small'
                print(f"  Selected: Small / Test preset")
                break
            elif choice == '2':
                self._preset_mode = 'production'
                print(f"  Selected: Production preset")
                break
            elif choice == '3':
                self._preset_mode = 'custom'
                print("  Selected: Custom setup")
                break
            else:
                print("  Please enter 1, 2, or 3.")
        print()

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

        if key_path:
            self.config_data['pipeline']['speaker_anonymization_key_path'] = key_path
            print()
        else:
            print("    Speaker IDs will be auto-generated as participant_N.")
            print()

    # -----------------------------------------------------------------
    # Step 2: Speaker role identification
    # -----------------------------------------------------------------

    _DEFAULT_THERAPISTS = [
        'Move-MORE Study', 'Instructor', 'Anand', 'Lani',
        'Wade (Study Coordinator)', 'Rebecca Heron',
        'Wade Balsamo (Study Coordinator)', 'Michelle Berg',
    ]

    def _step_2_speaker_filter(self):
        print("--- Step 2/12: Speaker Role Identification ---")
        print("    Identify which speakers are therapists/facilitators and which")
        print("    are participants. Therapist dialogue is preserved as read-only")
        print("    conversational context for adjacent participant segments.")
        print()

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
        speakers: List[str] = []

        if discovered:
            speaker_names = list(discovered.keys())
            print("    Enter speaker numbers for THERAPISTS (comma-separated),")
            print("    or type names manually. Blank line when done:")

            while True:
                raw = input("      > ").strip()
                if not raw:
                    break
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
    # Preset: LM Studio URL + apply preset settings
    # -----------------------------------------------------------------

    def _step_preset_backend(self):
        """For preset modes: just ask for the LM Studio server URL."""
        preset = _PRESET_SMALL if self._preset_mode == 'small' else _PRESET_PRODUCTION
        print(f"--- Backend: LM Studio Server URL ---")
        print("    All model calls are routed through your local LM Studio instance.")
        print(f"    Models that will be used:")
        for i, m in enumerate(preset['per_run_models']):
            label = ' (primary)' if i == 0 else f' (checker {i})'
            print(f"      Run {i + 1}: {m}{label}")
        print()

        lmstudio_url = _prompt("LM Studio server URL", 'http://10.0.0.58:1234/v1')
        self.config_data['theme_classification'] = {
            'backend': 'lmstudio',
            'model': preset['primary_model'],
            'lmstudio_base_url': lmstudio_url,
            'n_runs': preset['n_runs'],
            'temperature': preset['temperature'],
            'per_run_models': preset['per_run_models'],
        }
        print()

    def _apply_preset_settings(self):
        """Apply all non-interactive preset defaults and print a summary."""
        preset = _PRESET_SMALL if self._preset_mode == 'small' else _PRESET_PRODUCTION

        # Segmentation with preset embedding model
        self.config_data['segmentation'] = {
            'embedding_model': preset['segmentation_embedding_model'],
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

        # Framework
        self.config_data['framework'] = {'preset': 'vammr'}
        self.framework = get_vammr_framework()

        # Feature flags
        self.config_data['pipeline']['run_codebook_classifier'] = False

        # Confidence tiers (defaults)
        self.config_data['confidence_tiers'] = {
            'high_confidence': 0.8,
            'medium_min_confidence': 0.6,
        }

        print(f"--- Applied {preset['label']} Preset ---")
        print(f"    Segmentation embedding : {preset['segmentation_embedding_model']}")
        print(f"    Framework              : VAMMR ({self.framework.num_themes} themes)")
        print(f"    Classification runs    : {preset['n_runs']}")
        for i, m in enumerate(preset['per_run_models']):
            label = ' (primary)' if i == 0 else f' (checker {i})'
            print(f"      Run {i + 1}: {m}{label}")
        print(f"    Codebook classification: disabled")
        print(f"    Confidence thresholds  : high ≥ 0.8, medium ≥ 0.6")
        print()

    # -----------------------------------------------------------------
    # Step 3: Segmentation parameters (custom mode)
    # -----------------------------------------------------------------

    def _step_3_segmentation(self):
        print("--- Step 3/12: Segmentation Parameters ---")
        print()
        print("    SEGMENTATION EMBEDDING MODEL")
        print("    The embedding model converts transcript text into dense vectors.")
        print("    These vectors are compared with cosine similarity to detect topic")
        print("    shifts between utterances, forming the basis for segment boundaries.")
        print()
        print(_EMBEDDING_MODEL_DESCRIPTIONS)
        print()

        emb_choice = _prompt_choice(
            "Segmentation embedding model",
            ['minilm', 'qwen8b'],
            'qwen8b',
        )
        segmentation_embedding_model = _EMBEDDING_MODEL_OPTIONS[emb_choice]
        print(f"    Using: {segmentation_embedding_model}")
        print()

        if not _prompt_yes_no("Configure advanced segmentation options?", False):
            self.config_data['segmentation'] = {
                'embedding_model': segmentation_embedding_model,
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

        print()
        print("    MAX GAP SECONDS")
        print("    Utterances from the same speaker within this time gap are grouped")
        print("    together before semantic analysis. Larger values merge more of a")
        print("    speaker's consecutive speech into a single candidate segment.")
        max_gap = _prompt_float("Max time gap (seconds) between utterances to group", 15.0)

        print()
        print("    MIN WORDS PER SENTENCE")
        print("    Sentences shorter than this threshold are folded into the")
        print("    preceding or following sentence rather than standing alone.")
        print("    Prevents very short phrases (e.g., 'Mm-hmm') from fragmenting segments.")
        min_words_sent = _prompt_int("Min words per sentence (shorter are folded into neighbors)", 20)

        print()
        print("    MAX SEGMENT DURATION")
        print("    Hard ceiling on how long (in seconds of audio) a single segment")
        print("    can span. Segments exceeding this are split at the nearest")
        print("    natural boundary to keep analysis units manageable.")
        max_duration = _prompt_float("Max segment duration (seconds)", 60.0)

        print()
        print("    SEGMENT WORD BOUNDS")
        print("    Min/max word counts for the conversational segmentation pass.")
        print("    Segments below min words are merged with adjacent segments.")
        print("    Segments above max words are split at topic-shift boundaries.")
        min_words = _prompt_int("Min words per segment", 60)
        max_words = _prompt_int("Max words per segment", 500)

        print()
        print("    ADAPTIVE SIMILARITY THRESHOLD")
        print("    Rather than a fixed cosine-similarity cutoff, the adaptive mode")
        print("    detects local minima in the similarity curve — places where the")
        print("    conversation is least coherent — and places boundaries there.")
        print("    Min prominence controls how pronounced a dip must be to count.")
        use_adaptive = _prompt_yes_no(
            "Use adaptive similarity threshold (local minima detection)?", True
        )
        min_prominence = 0.05
        if use_adaptive:
            print()
            print("    MIN PROMINENCE")
            print("    Controls sensitivity of the adaptive boundary detector. Lower values")
            print("    create more boundaries (finer segmentation); higher values require")
            print("    steeper topic shifts before a boundary is placed.")
            min_prominence = _prompt_float("Min prominence for similarity dips", 0.05)

        print()
        print("    TOPIC CLUSTERING")
        print("    After similarity-based segmentation, agglomerative clustering groups")
        print("    segments by topic similarity, strengthening boundary confidence at")
        print("    points where clusters transition. Adds a second pass of boundary evidence.")
        use_clustering = _prompt_yes_no(
            "Use topic clustering for additional boundary detection?", True
        )

        print()
        print("    LLM SEGMENTATION REFINEMENT")
        print("    Passes candidate boundaries and segment pairs to the primary LLM")
        print("    for review. The LLM can confirm, reject, or relocate boundaries")
        print("    based on conversational coherence cues that embedding similarity misses.")
        use_llm_refine = _prompt_yes_no(
            "Use LLM-assisted segmentation refinement?", True
        )
        llm_refine_mode = 'full'
        if use_llm_refine:
            print()
            print("    REFINEMENT MODES")
            print("      boundary_review   : Re-evaluate ambiguous boundaries only")
            print("      context_expansion : Expand segments with surrounding dialogue")
            print("      coherence_check   : Split oversized segments at natural breaks")
            print("      full              : All three passes (recommended)")
            llm_refine_mode = _prompt_choice(
                "Refinement mode",
                ['boundary_review', 'context_expansion', 'coherence_check', 'full'],
                'full',
            )

        self.config_data['segmentation'] = {
            'embedding_model': segmentation_embedding_model,
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
    # Step 4: Backend & model (custom mode)
    # -----------------------------------------------------------------

    def _step_4_backend(self):
        print("--- Step 4/12: Backend & Model ---")
        print()
        print("    PRIMARY MODEL")
        print("    The primary model is the main LLM used for all classification and")
        print("    LLM-assisted segmentation refinement. For multi-run interrater")
        print("    reliability (configured in Step 8), this model serves as Run 1.")
        print()

        backend = _prompt_choice(
            "Backend",
            ['openrouter', 'replicate', 'huggingface', 'ollama', 'lmstudio'],
            'lmstudio',
        )

        if backend == 'lmstudio':
            lmstudio_url = _prompt(
                "LM Studio server URL", 'http://10.0.0.58:1234/v1'
            )
            model = _prompt(
                "Primary model (used for segmentation refinement and classification run 1)",
                'nvidia/nemotron-3-super',
            )
            print(f"    LM Studio backend: {lmstudio_url}")
            self.config_data['theme_classification'] = {
                'backend': 'lmstudio',
                'model': model,
                'lmstudio_base_url': lmstudio_url,
            }
            print()
            return

        model = _prompt("Model ID", 'openai/gpt-4o')

        if backend == 'openrouter':
            env_key = os.environ.get('OPENROUTER_API_KEY', '')
            if not env_key:
                print("    Note: Set OPENROUTER_API_KEY environment variable before running.")
        elif backend == 'replicate':
            env_key = os.environ.get('REPLICATE_API_TOKEN', '')
            if not env_key:
                print("    Note: Set REPLICATE_API_TOKEN environment variable before running.")

        models = []
        if _prompt_yes_no("Use multiple models for cross-referencing?", False):
            print("    Enter model IDs one per line (blank line to finish):")
            models.append(model)
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
    # Step 5: Framework selection (custom mode)
    # -----------------------------------------------------------------

    def _step_5_framework(self):
        print("--- Step 5/12: Theme Framework ---")
        print()
        print("    THEME FRAMEWORK")
        print("    Defines the coding scheme applied to each participant segment.")
        print("    VAMMR (Values, Actions, Motivational states, Meaning-making,")
        print("    and Regulatory processes) is the validated framework for this study.")
        print("    A custom framework JSON can be used for other qualitative coding systems.")
        print()
        choice = _prompt_choice("Framework", ['vammr', 'custom'], 'vammr')

        if choice == 'vammr':
            self.framework = get_vammr_framework()
            self.config_data['framework'] = {'preset': 'vammr'}
        else:
            path = _prompt("Path to custom framework JSON")
            self.config_data['framework'] = {'custom_path': path}
            try:
                with open(path) as f:
                    fw_data = json.load(f)
                from theme_framework.theme_schema import ThemeDefinition
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
    # Step 6: Exemplar utterances (custom mode)
    # -----------------------------------------------------------------

    def _step_6_exemplars(self):
        print("--- Step 6/12: Exemplar Utterances ---")
        if not self.framework:
            print("    No framework loaded; skipping exemplar customization.")
            print()
            return

        print()
        print("    EXEMPLAR UTTERANCES")
        print("    Each theme in the framework can include example utterances that")
        print("    ground the LLM's understanding of that theme. Custom exemplars")
        print("    drawn from your own study population can improve classification")
        print("    accuracy by reducing abstraction drift.")
        print()
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
    # Step 7: Codebook classification (custom mode)
    # -----------------------------------------------------------------

    def _step_7_codebook(self):
        print("--- Step 7/12: Codebook Classification ---")
        print()
        print("    CODEBOOK CLASSIFICATION")
        print("    A secondary classification pass that applies a phenomenological codebook")
        print("    alongside the theme framework. Uses two complementary methods:")
        print("      1. Embedding similarity — segments and codebook entries are encoded")
        print("         as vectors and matched by cosine distance (asymmetric encoding:")
        print("         segments as queries, codebook entries as passages).")
        print("      2. LLM zero-shot prompt — the primary model classifies each segment")
        print("         against codebook codes with chain-of-thought reasoning.")
        print("    Both results are reconciled by an ensemble step.")
        print()
        enable = _prompt_yes_no("Enable codebook classification?", False)
        self.config_data['pipeline']['run_codebook_classifier'] = enable

        print()
        print("    SEGMENTATION EMBEDDING MODEL (also used for codebook if enabled)")
        print("    Even without codebook classification, the embedding model is used")
        print("    during segmentation to compute semantic similarity between utterances.")
        print("    This is the same model configured in Step 3.")
        print()
        print("    CODEBOOK EMBEDDING MODEL")
        if enable:
            print("    When codebook classification is enabled, you may use a separate")
            print("    embedding model for codebook retrieval (or the same one as segmentation).")
            print()
            print(_EMBEDDING_MODEL_DESCRIPTIONS)
            print()
            codebook_choice = _prompt_choice(
                "Codebook", ['phenomenology', 'custom'], 'phenomenology'
            )
            if codebook_choice == 'phenomenology':
                self.config_data['codebook'] = {'preset': 'phenomenology'}
            else:
                path = _prompt("Path to custom codebook JSON")
                self.config_data['codebook'] = {'custom_path': path}

            print()
            print("    CODEBOOK EMBEDDING MODEL")
            print("    This model encodes codebook entries and participant segments into")
            print("    vectors for retrieval. Qwen3-8B is strongly recommended here")
            print("    because codebook entries use technical phenomenological language")
            print("    that benefits from a richer embedding space.")
            print()
            emb_choice = _prompt_choice(
                "Codebook embedding model",
                ['minilm', 'qwen8b'],
                'qwen8b',
            )
            embedding_model = _EMBEDDING_MODEL_OPTIONS[emb_choice]

            print()
            print("    TWO-PASS EMBEDDING CLASSIFICATION")
            print("    Pass 1 retrieves the top-K most similar codebook entries for each")
            print("    segment. Pass 2 re-ranks those candidates with a more targeted query.")
            print("    Two-pass generally improves precision at moderate extra cost.")
            two_pass = _prompt_yes_no("Enable two-pass embedding classification?", True)

            self.config_data['codebook_embedding'] = {
                'embedding_model': embedding_model,
                'two_pass': two_pass,
            }

            exemplar_path = _prompt("Exemplar import path for codebook (blank for none)", '')
            if exemplar_path:
                self.config_data['codebook_embedding']['exemplar_import_path'] = exemplar_path
        else:
            print("    Codebook classification is disabled — no codebook embedding model needed.")
            print("    The segmentation embedding model (configured in Step 3) is still active.")

        print()

    # -----------------------------------------------------------------
    # Step 8: Classification parameters (custom mode)
    # -----------------------------------------------------------------

    _DEFAULT_LMSTUDIO_RUN_MODELS = [
        'google/gemma-4-31b',
        'qwen/qwen3-next-80b',
    ]

    def _step_8_classification(self):
        print("--- Step 8/12: Classification Parameters ---")
        print()
        print("    NUMBER OF CLASSIFICATION RUNS (N_RUNS)")
        print("    Each run is an independent classification of every segment by a")
        print("    separate LLM (interrater reliability). The final theme label is")
        print("    determined by majority vote across all runs. A consistency score")
        print("    reflects genuine cross-model agreement rather than stochastic")
        print("    variation within a single model.")
        print("    Recommended: 3 runs (one primary + two checkers).")
        print()
        n_runs = _prompt_int("Number of classification runs per segment", 3)

        print()
        print("    TEMPERATURE")
        print("    Controls output randomness for each model. Lower values (0.0–0.1)")
        print("    produce more deterministic, reproducible classifications. Higher")
        print("    values (0.5+) introduce more variation — not recommended for")
        print("    qualitative research where consistency matters.")
        temperature = _prompt_float("Temperature", 0.1)

        self.config_data['theme_classification']['n_runs'] = n_runs
        self.config_data['theme_classification']['temperature'] = temperature

        if n_runs >= 2:
            self._configure_per_run_models(n_runs)

        print()
        if _prompt_yes_no("Configure advanced classification settings?", False):
            self._step_8b_advanced_classification()

    def _step_8b_advanced_classification(self):
        """Advanced: zero-shot toggle, exemplar count caps, evidence thresholds."""
        tc = self.config_data.setdefault('theme_classification', {})

        print()
        print("    ZERO-SHOT MODE")
        print("    When enabled, no example utterances are shown in the classification")
        print("    prompt — only the theme definitions, features, and key distinctions.")
        print("    Default: off (all exemplars, subtle, and adversarial examples shown).")
        zero_shot = _prompt_yes_no("Use zero-shot prompting (no examples in prompt)?", False)
        tc['zero_shot_prompt'] = zero_shot

        if not zero_shot:
            print()
            print("    EXEMPLAR COUNT LIMITS")
            print("    Controls how many example utterances of each type are included per theme.")
            print("    Leave blank to include all available (default behavior).")
            n_ex_raw = _prompt("Max exemplars per theme (blank = all)", '')
            tc['prompt_n_exemplars'] = int(n_ex_raw) if n_ex_raw.strip().isdigit() else None

            include_subtle = _prompt_yes_no("Include subtle (edge-case) utterances?", True)
            tc['prompt_include_subtle'] = include_subtle
            if include_subtle:
                n_sub_raw = _prompt("Max subtle examples per theme (blank = all)", '')
                tc['prompt_n_subtle'] = int(n_sub_raw) if n_sub_raw.strip().isdigit() else None

            include_adv = _prompt_yes_no("Include adversarial (boundary watch-out) utterances?", True)
            tc['prompt_include_adversarial'] = include_adv
            if include_adv:
                n_adv_raw = _prompt("Max adversarial examples per theme (blank = all)", '')
                tc['prompt_n_adversarial'] = int(n_adv_raw) if n_adv_raw.strip().isdigit() else None

        print()
        print("    SECONDARY EVIDENCE THRESHOLDS")
        print("    Controls how secondary stages are assigned via evidence pooling across raters.")
        print("    Weight (0.0–1.0): how much a secondary or dissenting primary vote contributes.")
        print("      Default 0.6 = a dissenting rater's strong vote counts as meaningful evidence.")
        print("    Threshold: minimum pooled evidence score needed to assign a secondary label.")
        print("      Default 0.5 = at least one rater's moderately-confident mention required.")
        sec_weight = _prompt_float("Evidence weight for secondary/dissenting votes", 0.6)
        sec_thresh = _prompt_float("Minimum evidence threshold for secondary label", 0.5)
        tc['evidence_secondary_weight'] = sec_weight
        tc['evidence_presence_threshold'] = sec_thresh
        print()

    def _configure_per_run_models(self, n_runs: int):
        """Configure adversarial checker models for interrater reliability."""
        primary_model = self.config_data.get('theme_classification', {}).get('model', '')
        n_checkers = n_runs - 1

        checker_defaults: List[str] = [
            m for m in self._DEFAULT_LMSTUDIO_RUN_MODELS if m != primary_model
        ]
        while len(checker_defaults) < n_checkers:
            checker_defaults.append(checker_defaults[-1] if checker_defaults else primary_model)

        print()
        print("    CHECKER MODELS (ADDITIONAL RATERS)")
        print("    Each checker model independently classifies every segment.")
        print("    Using architecturally distinct models (different families and sizes)")
        print("    maximizes the independence of each rater's perspective, giving the")
        print("    majority-vote consistency score more interpretive weight.")
        print()
        print(f"    Interrater reliability: {n_runs} runs total")
        print(f"      Run 1: {primary_model}  (primary — set in Step 4)")
        if n_checkers > 0:
            print(f"    Enter {n_checkers} additional checker model(s).")
            print("    Press Enter to accept the suggested default for each checker.")
        print()

        per_run_models: List[str] = [primary_model]
        for i in range(n_checkers):
            default = checker_defaults[i] if i < len(checker_defaults) else ''
            model = _prompt(f"Checker model {i + 1}", default)
            per_run_models.append(model)

        if len(set(per_run_models)) < n_runs:
            print("    Warning: duplicate models — runs sharing a model will not provide")
            print("    independent ratings. Consider using distinct model families.")

        self.config_data['theme_classification']['per_run_models'] = per_run_models
        print(f"    Configured {n_runs} rater models:")
        for i, m in enumerate(per_run_models):
            label = " (primary)" if i == 0 else f" (checker {i})"
            print(f"      Run {i + 1}: {m}{label}")

    # -----------------------------------------------------------------
    # Step 9: Confidence thresholds (custom mode)
    # -----------------------------------------------------------------

    def _step_9_confidence(self):
        print("--- Step 9/12: Confidence Thresholds ---")
        print()
        print("    CONFIDENCE THRESHOLDS")
        print("    After majority-vote classification, each segment receives a")
        print("    confidence score (0.0–1.0) based on model agreement and LLM")
        print("    certainty signals. Segments are assigned to tiers:")
        print("      High   — score ≥ high threshold: strong label, included in")
        print("               primary analyses without human review flag.")
        print("      Medium — score ≥ medium threshold: adequate confidence,")
        print("               may be flagged for spot-check review.")
        print("      Low    — below medium: recommended for human review.")
        print()
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
            "Max words per averaged block in cue_response.txt", 500
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
            embedding_model=seg.get('embedding_model', 'Qwen/Qwen3-Embedding-8B'),
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
            zero_shot_prompt=tc.get('zero_shot_prompt', False),
            prompt_n_exemplars=tc.get('prompt_n_exemplars'),
            prompt_include_subtle=tc.get('prompt_include_subtle', True),
            prompt_n_subtle=tc.get('prompt_n_subtle'),
            prompt_include_adversarial=tc.get('prompt_include_adversarial', True),
            prompt_n_adversarial=tc.get('prompt_n_adversarial'),
            evidence_secondary_weight=tc.get('evidence_secondary_weight', 0.6),
            evidence_presence_threshold=tc.get('evidence_presence_threshold', 0.5),
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
