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
from gnn_layer.config import GnnLayerConfig

# Keys that hold secrets and must never be written to config JSON
_SECRET_KEYS = frozenset({'api_key'})


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
class TestSetSpec:
    """Parameters for one kind of cross-session validation test set."""
    enabled: bool = False
    name: str = ""
    n_sets: int = 1
    fraction_per_set: float = 0.10
    random_seed: int = 42


@dataclass
class TestSetsConfig:
    """Multi-kind test set configuration (VAAMR, PURER, codebook)."""
    vaamr: TestSetSpec = field(
        default_factory=lambda: TestSetSpec(enabled=True, name='vaamr_testset')
    )
    purer: TestSetSpec = field(
        default_factory=lambda: TestSetSpec(enabled=False, name='purer_testset')
    )
    codebook: TestSetSpec = field(
        default_factory=lambda: TestSetSpec(enabled=False, name='codebook_testset')
    )

    def any_enabled(self) -> bool:
        return self.vaamr.enabled or self.purer.enabled or self.codebook.enabled


@dataclass
class ContentValiditySpec:
    """Parameters for one content-validity testset variant."""
    enabled: bool = False
    name: str = ""


@dataclass
class ContentValidityConfig:
    """Multi-kind content-validity configuration."""
    vaamr: ContentValiditySpec = field(
        default_factory=lambda: ContentValiditySpec(enabled=True, name='cv_vaamr_v1')
    )
    purer: ContentValiditySpec = field(
        default_factory=lambda: ContentValiditySpec(enabled=False, name='cv_purer_v1')
    )

    def any_enabled(self) -> bool:
        return self.vaamr.enabled or self.purer.enabled


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

    max_cue_words : int
        Word budget per sub-cue when splitting an over-long cue block.  Cue
        blocks whose total therapist word count exceeds this value are split
        along therapist turn boundaries into contiguous sub-cues, each staying
        within the budget, before being sent to the LLM.  A single therapist
        turn that alone exceeds the budget is sent as a singleton sub-cue.
        Splitting produces finer-grained PURER labels (one per sub-cue) rather
        than dropping or monolithically sending the block.
    """
    skip_lesson_content: bool = True
    max_lesson_words: int = 400
    therapist_max_gap_seconds: float = 120.0
    max_context_words: int = 1000
    max_cue_words: int = 300   # words; cue blocks longer than this are split into contiguous sub-cues (along turn boundaries) before LLM classification, instead of being sent as one over-long prompt


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
class SuperpositionConfig:
    """Surface VAAMR stage-mixture (superposition) in analysis reports.

    Always-on at analysis time and source-agnostic: mixtures come from the GNN
    when it has run, else are reconstructed from LLM ballots, else from
    secondary_stage. Drives the liminality (entropy) reporting and the
    mechanistic Δprogression analysis. Additive — never alters hard labels.
    """
    enabled: bool = True
    # 'auto' = GNN → ballots → secondary priority; or force one of gnn|ballots|secondary.
    mixture_source: str = 'auto'
    liminal_entropy_threshold: float = 0.6   # normalized entropy ≥ this ⇒ liminal
    liminal_gap_threshold: float = 0.25      # top1−top2 mixture gap < this ⇒ liminal
    active_stage_threshold: float = 0.15     # stage counts as "active" at ≥ this probability
    run_mechanism_analysis: bool = True      # FROM→CUE→TO continuous Δprogression analysis


@dataclass
class EfficacyConfig:
    """Progression-summary settings (descriptive single-arm dossier).

    Describes internal VAAMR language progression and, when an external clinical-
    outcomes CSV is provided, correlates progression against it for CONVERGENT
    VALIDITY (not efficacy — single-arm, observational). Keeps the 'efficacy' name
    for config back-compat; the report is 06_reports/01_outcomes/progression_summary.txt.
    """
    enabled: bool = True
    # Participant-keyed CSV; absolute, or relative to the output dir. Default 02_meta/outcomes.csv.
    outcomes_path: Optional[str] = None
    adaptive_stages: List[int] = field(default_factory=lambda: [2, 3, 4])
    maladaptive_stages: List[int] = field(default_factory=lambda: [0, 1])
    barrier_from: int = 1       # Avoidance
    barrier_to: int = 2         # Attention-Regulation


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    # Input
    transcript_dir: str = './data/input/'
    trial_id: str = 'standard'

    # Output
    output_dir: str = './data/output/'

    # Framework selection per dialogue side
    participant_framework: str = 'vaamr'
    therapist_framework: Optional[str] = 'purer'  # None = no therapist-side classification

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
    test_sets: TestSetsConfig = field(default_factory=TestSetsConfig)
    content_validity: ContentValidityConfig = field(default_factory=ContentValidityConfig)
    confidence_tiers: ConfidenceTierConfig = field(default_factory=ConfidenceTierConfig)
    therapist_cues: TherapistCueConfig = field(default_factory=TherapistCueConfig)
    purer_cue: PurerCueConfig = field(default_factory=PurerCueConfig)
    session_summaries: SessionSummariesConfig = field(default_factory=SessionSummariesConfig)
    participant_summaries: ParticipantSummariesConfig = field(default_factory=ParticipantSummariesConfig)
    # GNN representation-and-discovery layer (analysis-time; OFF by default)
    gnn_layer: GnnLayerConfig = field(default_factory=GnnLayerConfig)
    # Superposition surfacing + mechanistic analysis (analysis-time; ON by default)
    superposition: SuperpositionConfig = field(default_factory=SuperpositionConfig)
    # Program-efficacy dossier (analysis-time; ON by default)
    efficacy: EfficacyConfig = field(default_factory=EfficacyConfig)

    # Resume from checkpoint
    resume_from: Optional[str] = None

    # Speaker anonymization key import (for seeding participant IDs)
    speaker_anonymization_key_path: Optional[str] = None

    # PHI text scrubbing — replace human names in segment text before freezing
    anonymize_transcript_text: bool = True
    anonymize_text_model: str = 'obi/deid_roberta_i2b2'
    anonymize_text_confidence_threshold: float = 0.6

    # Post-pipeline results analysis
    auto_analyze: bool = True

    # Set by the incremental add-data walkthrough when the user indicates new transcripts belong to a previously-unseen cohort.
    # Applied to segmentation metadata for any new sessions processed during that invocation.
    incremental_new_cohort_id: Optional[str] = None

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

    def to_dict(self) -> Dict:
        return self.to_json()

    @classmethod
    def from_dict(cls, data: Dict) -> 'PipelineConfig':
        return cls.from_json(data)

    @classmethod
    def from_json(cls, data: Dict) -> 'PipelineConfig':
        """Reconstruct PipelineConfig from a dict (e.g., loaded JSON).

        Nested sub-config dicts are converted back to their dataclass types.
        Secret fields (api_key) are resolved from
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
            'confidence_tiers': ConfidenceTierConfig,
            'therapist_cues': TherapistCueConfig,
            'purer_cue': PurerCueConfig,
            'session_summaries': SessionSummariesConfig,
            'participant_summaries': ParticipantSummariesConfig,
            'gnn_layer': GnnLayerConfig,
            'superposition': SuperpositionConfig,
            'efficacy': EfficacyConfig,
        }

        kwargs = {}
        valid_field_names = {f.name for f in fields(cls)}

        # Flatten legacy working-branch format: top-level PipelineConfig fields
        # may be nested under a "pipeline" key.
        if 'pipeline' in data and isinstance(data['pipeline'], dict):
            merged = dict(data)
            merged.update(data['pipeline'])
            del merged['pipeline']
            data = merged

        for key, value in data.items():
            if key not in valid_field_names:
                continue
            if key == 'test_sets' and isinstance(value, dict):
                kwargs['test_sets'] = _parse_test_sets_config(value)
            elif key == 'content_validity' and isinstance(value, dict):
                kwargs['content_validity'] = _parse_content_validity_config(value)
            elif key in sub_config_map and isinstance(value, dict):
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

        return config


def _parse_test_set_spec(d: dict) -> TestSetSpec:
    spec_fields = {f.name for f in fields(TestSetSpec)}
    return TestSetSpec(**{k: v for k, v in d.items() if k in spec_fields})


def _parse_test_sets_config(d: dict) -> TestSetsConfig:
    """Parse test_sets dict — handles both old flat format and new multi-kind format."""
    # Old Phase 1 format: {'enabled': bool, 'n_sets': int, 'fraction_per_set': float, 'random_seed': int}
    if 'enabled' in d or 'n_sets' in d:
        vaamr = TestSetSpec(
            enabled=d.get('enabled', True),
            name='vaamr_testset',
            n_sets=d.get('n_sets', 1),
            fraction_per_set=d.get('fraction_per_set', 0.10),
            random_seed=d.get('random_seed', 42),
        )
        return TestSetsConfig(vaamr=vaamr)
    # New format: {'vaamr': {...}, 'purer': {...}, 'codebook': {...}}
    return TestSetsConfig(
        vaamr=_parse_test_set_spec(d.get('vaamr', {})) if d.get('vaamr') else TestSetSpec(enabled=True, name='vaamr_testset'),
        purer=_parse_test_set_spec(d.get('purer', {})) if d.get('purer') else TestSetSpec(name='purer_testset'),
        codebook=_parse_test_set_spec(d.get('codebook', {})) if d.get('codebook') else TestSetSpec(name='codebook_testset'),
    )


def _parse_content_validity_spec(d: dict) -> ContentValiditySpec:
    spec_fields = {f.name for f in fields(ContentValiditySpec)}
    return ContentValiditySpec(**{k: v for k, v in d.items() if k in spec_fields})


def _parse_content_validity_config(d: dict) -> ContentValidityConfig:
    return ContentValidityConfig(
        vaamr=_parse_content_validity_spec(d.get('vaamr', {})) if d.get('vaamr') else ContentValiditySpec(enabled=True, name='cv_vaamr_v1'),
        purer=_parse_content_validity_spec(d.get('purer', {})) if d.get('purer') else ContentValiditySpec(name='cv_purer_v1'),
    )


def _blank_secrets(d: dict):
    """Recursively blank secret keys in a nested dict."""
    for key in list(d.keys()):
        if key in _SECRET_KEYS:
            d[key] = ''
        elif isinstance(d[key], dict):
            _blank_secrets(d[key])
