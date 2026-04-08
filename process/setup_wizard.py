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
    ConfidenceTierConfig,
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
        self._step_2_speaker_filter()
        self._step_3_segmentation()
        self._step_4_backend()
        self._step_5_framework()
        self._step_6_exemplars()
        self._step_7_codebook()
        self._step_8_classification()
        self._step_9_confidence()
        self._step_10_run_mode()
        config_path = self._step_11_save()

        return {
            'config_path': config_path,
            'config_data': self.config_data,
        }

    # -----------------------------------------------------------------
    # Step 1: Input/Output paths
    # -----------------------------------------------------------------
    def _step_1_paths(self):
        print("--- Step 1/11: Input/Output Paths ---")
        self.config_data['pipeline'] = {
            'transcript_dir': _prompt("Transcript directory", './data/input/'),
            'output_dir': _prompt("Output directory", './data/output/'),
            'trial_id': _prompt("Trial ID", 'standard'),
        }
        print()

    # -----------------------------------------------------------------
    # Step 2: Speaker filtering
    # -----------------------------------------------------------------

    # Default speakers to pre-select for exclusion (Move-MORE study)
    _DEFAULT_EXCLUDED = ['Wade (Study Coordinator)', 'Move-MORE Study', 'Anand', 'Lani']

    def _step_2_speaker_filter(self):
        print("--- Step 2/11: Speaker Filtering ---")
        print("    Control which speakers' sentences are included in classification.")
        print("    Filtering is applied at the *sentence* level BEFORE segmentation,")
        print("    so excluded speakers' content never enters segments.")
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
                default_tag = " [default exclude]" if name in self._DEFAULT_EXCLUDED else ""
                print(f"      {i:2d}. {name} ({count} utterances){default_tag}")
            print()
        else:
            print("    No transcripts found to scan — enter speaker names manually.")
            print()

        print("    Filter modes:")
        print("      none    : classify all speakers' segments")
        print("      exclude : remove listed speakers' sentences before segmentation")
        print("      isolate : keep ONLY listed speakers' sentences")
        print()

        mode = _prompt_choice("Speaker filter mode", ['none', 'exclude', 'isolate'], 'exclude')

        if mode == 'none':
            self.config_data['speaker_filter'] = {'mode': 'none', 'speakers': []}
            print()
            return

        # Build default selection
        if discovered:
            if mode == 'exclude':
                defaults = [n for n in self._DEFAULT_EXCLUDED if n in discovered]
            else:
                # For isolate, default to all speakers NOT in the exclusion list
                defaults = [n for n in discovered if n not in self._DEFAULT_EXCLUDED]
        else:
            defaults = list(self._DEFAULT_EXCLUDED) if mode == 'exclude' else []

        if defaults:
            verb = "exclude" if mode == 'exclude' else "isolate"
            print(f"    Default speakers to {verb}:")
            for name in defaults:
                print(f"      - {name}")
            use_defaults = _prompt_yes_no(f"Use these defaults?", True)
            if use_defaults:
                speakers = list(defaults)
            else:
                speakers = self._prompt_speaker_selection(discovered, mode)
        else:
            speakers = self._prompt_speaker_selection(discovered, mode)

        if not speakers:
            print("    No speakers selected — falling back to default exclusion list.")
            speakers = list(self._DEFAULT_EXCLUDED)

        verb = "Excluding" if mode == 'exclude' else "Isolating"
        print(f"    {verb}: {speakers}")
        self.config_data['speaker_filter'] = {'mode': mode, 'speakers': speakers}
        print()

    def _prompt_speaker_selection(self, discovered: dict, mode: str) -> List[str]:
        """Let the user select speakers from the discovered list or enter manually."""
        speakers: List[str] = []

        if discovered:
            speaker_names = list(discovered.keys())
            verb = "EXCLUDE" if mode == 'exclude' else "ISOLATE"
            print(f"    Enter speaker numbers to {verb} (comma-separated), or type names manually.")
            print(f"    Blank line when done:")

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
                                print(f"        + {name}")
                        else:
                            print(f"        Invalid number: {part.strip()}")
                else:
                    # Treat as a speaker name
                    if raw not in speakers:
                        speakers.append(raw)
                        print(f"        + {raw}")
        else:
            verb = "EXCLUDE" if mode == 'exclude' else "ISOLATE"
            print(f"    Enter speaker labels to {verb} (one per line, blank when done):")
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
        print("--- Step 3/11: Segmentation Parameters ---")

        if not _prompt_yes_no("Configure advanced segmentation options?", False):
            # Use defaults
            self.config_data['segmentation'] = {
                'use_conversational_segmenter': True,
                'max_gap_seconds': 30.0,
                'min_words_per_sentence': 10,
                'max_segment_duration_seconds': 300.0,
                'min_segment_words_conversational': 60,
                'max_segment_words_conversational': 400,
                'use_adaptive_threshold': True,
                'min_prominence': 0.05,
                'use_topic_clustering': False,
                'use_llm_refinement': False,
                'llm_refinement_mode': 'boundary_review',
            }
            print("    Using defaults: no cross-speaker grouping, 30s max gap,")
            print("    10-word min per sentence, 5min max segment duration,")
            print("    60-400 word segments, adaptive threshold enabled.")
            print()
            return

        use_conv = _prompt_yes_no(
            "Use conversational segmenter (groups by topic across speakers)?", True
        )

        max_gap = _prompt_float("Max time gap (seconds) between utterances to group", 30.0)
        min_words_sent = _prompt_int("Min words per sentence (shorter sentences are dropped)", 10)
        max_duration = _prompt_float("Max segment duration (seconds)", 30.0)

        if use_conv:
            min_words = _prompt_int("Min words per segment (conversational)", 60)
            max_words = _prompt_int("Max words per segment (conversational)", 400)
        else:
            min_words = _prompt_int("Min words per segment", 30)
            max_words = _prompt_int("Max words per segment", 200)

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
            "Use LLM-based boundary refinement?", True
        )
        llm_refine_mode = 'boundary_review'
        if use_llm_refine:
            print("    Refinement modes:")
            print("      boundary_review     : Re-evaluate ambiguous boundaries")
            print("      cross_speaker_merge : Merge cross-speaker conversational units")
            print("      full                : Both passes")
            llm_refine_mode = _prompt_choice(
                "Refinement mode",
                ['boundary_review', 'cross_speaker_merge', 'full'],
                'full',
            )

        self.config_data['segmentation'] = {
            'use_conversational_segmenter': use_conv,
            'max_gap_seconds': max_gap,
            'min_words_per_sentence': min_words_sent,
            'max_segment_duration_seconds': max_duration,
            'min_segment_words_conversational' if use_conv else 'min_segment_words': min_words,
            'max_segment_words_conversational' if use_conv else 'max_segment_words': max_words,
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
        print("--- Step 4/11: Backend & Model ---")
        backend = _prompt_choice(
            "Backend",
            ['openrouter', 'replicate', 'huggingface', 'ollama', 'lmstudio'],
            'lmstudio',
        )

        # LM Studio: prompt for server URL and model name
        if backend == 'lmstudio':
            lmstudio_url = _prompt(
                "LM Studio server URL", 'http://127.0.0.1:1234/v1'
            )
            model = _prompt("Model ID (as shown in LM Studio)", 'nvidia/nemotron-3-nano-4b')
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
        print("--- Step 5/11: Theme Framework ---")
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
        print("--- Step 6/11: Exemplar Utterances ---")
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
        print("--- Step 7/11: Codebook Classification ---")
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
    # Step 8: Classification parameters
    # -----------------------------------------------------------------
    def _step_8_classification(self):
        print("--- Step 8/11: Classification Parameters ---")
        n_runs = _prompt_int("Number of runs per segment", 1)
        temperature = _prompt_float("Temperature", 0.0)

        self.config_data['theme_classification']['n_runs'] = n_runs
        self.config_data['theme_classification']['temperature'] = temperature
        print()

    # -----------------------------------------------------------------
    # Step 9: Confidence thresholds
    # -----------------------------------------------------------------
    def _step_9_confidence(self):
        print("--- Step 9/11: Confidence Thresholds ---")
        self.config_data['confidence_tiers'] = {
            'high_confidence': _prompt_float("High confidence threshold", 0.8),
            'medium_min_confidence': _prompt_float("Medium confidence threshold", 0.6),
        }
        print()

    # -----------------------------------------------------------------
    # Step 10: Run mode
    # -----------------------------------------------------------------
    def _step_10_run_mode(self):
        print("--- Step 10/11: Run Mode ---")
        print("    auto        : Fully automated (no human intervention)")
        print("    interactive : Prompt for validation of uncertain results")
        print("    review      : Batch validation at end")
        mode = _prompt_choice("Run mode", ['auto', 'interactive', 'review'], 'auto')
        self.config_data['pipeline']['run_mode'] = mode
        print()

    # -----------------------------------------------------------------
    # Step 11: Save & run
    # -----------------------------------------------------------------
    def _step_11_save(self) -> str:
        print("--- Step 11/11: Save Configuration ---")
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
        run_mode=pipeline.get('run_mode', 'auto'),
        run_theme_labeler=pipeline.get('run_theme_labeler', True),
        run_codebook_classifier=pipeline.get('run_codebook_classifier', False),
        segmentation=SegmentationConfig(
            use_conversational_segmenter=seg.get('use_conversational_segmenter', True),
            min_segment_words=seg.get('min_segment_words', 30),
            max_segment_words=seg.get('max_segment_words', 200),
            min_segment_words_conversational=seg.get('min_segment_words_conversational', 60),
            max_segment_words_conversational=seg.get('max_segment_words_conversational', 400),
            max_gap_seconds=seg.get('max_gap_seconds', 30.0),
            min_words_per_sentence=seg.get('min_words_per_sentence', 10),
            max_segment_duration_seconds=seg.get('max_segment_duration_seconds', 300.0),
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
            n_runs=tc.get('n_runs', 3),
            temperature=tc.get('temperature', 0.0),
            api_key=api_key,
            replicate_api_token=replicate_token,
            lmstudio_base_url=tc.get('lmstudio_base_url', 'http://127.0.0.1:1234/'),
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
