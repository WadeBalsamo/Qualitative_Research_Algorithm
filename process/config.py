"""
config.py
---------
Top-level pipeline configuration.

Aggregates sub-configs from classification_tools, constructs, and codebook.
"""

import os
from dataclasses import dataclass, field, asdict, fields
from typing import Dict, List, Optional

from theme_framework.config import ThemeClassificationConfig
from codebook.config import (
    EmbeddingClassifierConfig,
    LLMCodebookConfig,
    EnsembleConfig,
)

# Keys that hold secrets and must never be written to config JSON
_SECRET_KEYS = frozenset({'api_key', 'replicate_api_token'})


@dataclass
class SegmentationConfig:
    """Parameters for transcript segmentation."""
    embedding_model: str = 'Qwen/Qwen3-Embedding-8B'
    silence_threshold_ms: int = 1500
    semantic_shift_percentile: int = 25
    min_segment_words_conversational: int = 60
    max_segment_words_conversational: int = 500
    # Advanced grouping parameters
    max_gap_seconds: float = 15.0              # Max time gap (seconds) between utterances to group
    min_words_per_sentence: int = 20           # Sentences below this are folded into adjacent same-speaker sentence
    max_segment_duration_seconds: float = 60.0  # Max duration (seconds) of a single segment
    # Adaptive threshold / dual-window / clustering
    use_adaptive_threshold: bool = True        # Use local-minima detection instead of static percentile
    min_prominence: float = 0.05               # Minimum prominence for adaptive threshold peaks
    broad_window_size: int = 7                 # Window size for broad similarity curve
    use_topic_clustering: bool = True          # Use AgglomerativeClustering for topic boundaries (strengthens boundary confidence)
    # LLM segmentation refinement (runs by default when LLM backend is configured)
    use_llm_refinement: bool = True
    llm_refinement_mode: str = 'full'          # 'boundary_review', 'context_expansion', 'coherence_check', 'full'
    llm_ambiguity_threshold: float = 0.15      # Similarity-threshold proximity for "ambiguous" boundaries
    llm_batch_size: int = 5                    # Boundaries/pairs per LLM call
    # Verbose process logging
    verbose_segmentation: bool = True         # Write process_log.txt with every LLM I/O and step


@dataclass
class SpeakerFilterConfig:
    """Controls which speakers' segments are sent to the classifier.

    mode:
        'none'    — classify all segments (default)
        'exclude' — drop segments whose *sole* speaker is in ``speakers``
                    (multi-speaker segments are always kept)
    speakers:
        List of speaker label strings exactly as they appear in the transcript
        (e.g. ['Move-MORE Study', 'Wade (Study Coordinator)']).
    """
    mode: str = 'none'
    speakers: List[str] = field(default_factory=list)


@dataclass
class ValidationConfig:
    """Parameters for human validation set construction."""
    n_per_class: int = 50
    min_kappa: float = 0.70
    min_agreement: float = 0.75


@dataclass
class TestSetConfig:
    """Parameters for cross-session validation test set generation."""
    enabled: bool = True
    n_sets: int = 2
    fraction_per_set: float = 0.10
    random_seed: int = 42


@dataclass
class ConfidenceTierConfig:
    """Configurable thresholds for confidence tier assignment."""
    high_consistency: int = 3
    high_confidence: float = 0.8
    medium_min_consistency: int = 2
    medium_min_confidence: float = 0.6


@dataclass
class TherapistCueConfig:
    """Parameters for surfacing therapist dialogue cues at stage transitions."""
    enabled: bool = True
    max_length_per_cue: int = 250          # words; if cue exceeds this, LLM-summarize
    max_length_of_average_cue_responses: int = 500  # cap per averaged block in cue_response.txt


@dataclass
class PurerCueConfig:
    """Settings for cue-unit PURER classification (Stage 3c).

    PURER is classified at the *cue-block* level — one label per therapist
    response between two consecutive participant turns — rather than classifying
    every individual therapist segment in isolation.

    By default, PURER uses a single model for classification to ensure robustness,
    with no multi-run validation required. This simplifies implementation and 
    avoids the complexity of inter-rater reliability checks that aren't needed
    for this use case.

    skip_lesson_content : bool
        When True, skip PURER classification on cue blocks whose therapist
        text exceeds ``max_lesson_words``.  These are long didactic stretches
        (psychoeducation, guided meditation scripts) that are therapist-to-group
        monologues rather than direct responses to a specific participant turn.

    max_lesson_words : int
        Word threshold for lesson-content detection.  A cue block with more
        words than this is assumed to be a lesson segment rather than an
        interactive cue.  Relevant only when ``skip_lesson_content=True``.

    therapist_max_gap_seconds : float
        Gap threshold used when aggregating therapist sentences into cue-level
        blocks.  Overrides the participant segmentation ``max_gap_seconds``.
        Set higher (default 120 s) to avoid splitting within guided meditation
        pauses or psychoeducation delivery.

    max_context_words : int
        Word budget for the conversational context preamble included with each
        PURER cue classification prompt.  Uses the same ``_build_context_block``
        logic and ``CONTEXT_PREAMBLE`` format as participant VAAMR classification,
        but with a wider window driven by ``purer_classification.context_window_segments``
        (default 6, vs 2 for VAAMR).  Increase this if therapist cue text is long
        and preceding exchanges are being truncated too aggressively.
    """
    skip_lesson_content: bool = True
    max_lesson_words: int = 400
    therapist_max_gap_seconds: float = 120.0
    max_context_words: int = 1000


@dataclass
class SessionSummariesConfig:
    """LLM summaries of therapist language per session, stored as JSON + txt."""
    enabled: bool = True
    max_words_per_session: int = 500


@dataclass
class ParticipantSummariesConfig:
    """LLM summaries of each participant's own language per session, shown in participant reports."""
    enabled: bool = True
    max_words_per_session: int = 300


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    # Input
    transcript_dir: str = './data/input/'
    trial_id: str = 'standard'

    # Output
    output_dir: str = './data/output/'

    # Feature flags
    run_theme_labeler: bool = True
    run_codebook_classifier: bool = False
    run_purer_labeler: bool = True

    # Sub-configs
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    speaker_filter: SpeakerFilterConfig = field(default_factory=SpeakerFilterConfig)
    theme_classification: ThemeClassificationConfig = field(default_factory=ThemeClassificationConfig)
    purer_classification: ThemeClassificationConfig = field(
        default_factory=lambda: ThemeClassificationConfig(context_window_segments=6)
    )
    codebook_embedding: EmbeddingClassifierConfig = field(default_factory=EmbeddingClassifierConfig)
    codebook_llm: LLMCodebookConfig = field(default_factory=LLMCodebookConfig)
    codebook_ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    test_sets: TestSetConfig = field(default_factory=TestSetConfig)
    confidence_tiers: ConfidenceTierConfig = field(default_factory=ConfidenceTierConfig)
    therapist_cues: TherapistCueConfig = field(default_factory=TherapistCueConfig)
    purer_cue: PurerCueConfig = field(default_factory=PurerCueConfig)
    session_summaries: SessionSummariesConfig = field(default_factory=SessionSummariesConfig)
    participant_summaries: ParticipantSummariesConfig = field(default_factory=ParticipantSummariesConfig)

    # Resume from checkpoint
    resume_from: Optional[str] = None

    # Speaker anonymization key import (for seeding participant IDs)
    speaker_anonymization_key_path: Optional[str] = None

    # Post-pipeline results analysis
    auto_analyze: bool = True

    # Downstream
    autoresearch_dir: str = '../autoresearch/'

    def __post_init__(self):
        if self.speaker_anonymization_key_path is None:
            candidate = os.path.join(self.transcript_dir, 'speaker_anonymization_key.json')
            if os.path.exists(candidate):
                self.speaker_anonymization_key_path = candidate

    def to_json(self) -> Dict:
        """Serialize config to a JSON-safe dict, blanking secret fields."""
        data = asdict(self)
        _blank_secrets(data)
        return data

    @classmethod
    def from_json(cls, data: Dict) -> 'PipelineConfig':
        """Reconstruct PipelineConfig from a dict (e.g., loaded JSON).

        Nested sub-config dicts are converted back to their dataclass types.
        Secret fields (api_key, replicate_api_token) are resolved from
        environment variables if blank in the data.
        """
        import os

        sub_config_map = {
            'segmentation': SegmentationConfig,
            'speaker_filter': SpeakerFilterConfig,
            'theme_classification': ThemeClassificationConfig,
            'purer_classification': ThemeClassificationConfig,
            'codebook_embedding': EmbeddingClassifierConfig,
            'codebook_llm': LLMCodebookConfig,
            'codebook_ensemble': EnsembleConfig,
            'validation': ValidationConfig,
            'test_sets': TestSetConfig,
            'confidence_tiers': ConfidenceTierConfig,
            'therapist_cues': TherapistCueConfig,
            'purer_cue': PurerCueConfig,
            'session_summaries': SessionSummariesConfig,
            'participant_summaries': ParticipantSummariesConfig,
        }

        kwargs = {}
        valid_field_names = {f.name for f in fields(cls)}

        for key, value in data.items():
            if key not in valid_field_names:
                continue
            if key in sub_config_map and isinstance(value, dict):
                dc_cls = sub_config_map[key]
                dc_fields = {f.name for f in fields(dc_cls)}
                filtered = {k: v for k, v in value.items() if k in dc_fields}
                kwargs[key] = dc_cls(**filtered)
            else:
                kwargs[key] = value

        config = cls(**kwargs)

        # Resolve secrets from environment if blank
        tc = config.theme_classification
        if not tc.api_key:
            tc.api_key = os.environ.get('OPENROUTER_API_KEY', '')
        if not tc.replicate_api_token:
            tc.replicate_api_token = os.environ.get('REPLICATE_API_TOKEN', '')

        return config


def _blank_secrets(d: dict):
    """Recursively blank secret keys in a nested dict."""
    for key in list(d.keys()):
        if key in _SECRET_KEYS:
            d[key] = ''
        elif isinstance(d[key], dict):
            _blank_secrets(d[key])
