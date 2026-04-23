"""
config.py
---------
Top-level pipeline configuration.

Aggregates sub-configs from classification_tools, constructs, and codebook.
"""

from dataclasses import dataclass, field, asdict, fields
from typing import Dict, List, Optional

from constructs.config import ThemeClassificationConfig
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
    min_segment_words: int = 30
    max_segment_words: int = 200
    silence_threshold_ms: int = 1500
    semantic_shift_percentile: int = 25
    # Conversational segmenter (groups all speakers together by topic)
    use_conversational_segmenter: bool = True
    min_segment_words_conversational: int = 60
    max_segment_words_conversational: int = 400
    # Advanced grouping parameters
    max_gap_seconds: float = 30.0              # Max time gap (seconds) between utterances to group
    min_words_per_sentence: int = 10           # Sentences below this are folded into adjacent same-speaker sentence
    max_segment_duration_seconds: float = 300.0  # Max duration (seconds) of a single segment
    # Adaptive threshold / dual-window / clustering
    use_adaptive_threshold: bool = True        # Use local-minima detection instead of static percentile
    min_prominence: float = 0.05               # Minimum prominence for adaptive threshold peaks
    broad_window_size: int = 7                 # Window size for broad similarity curve
    use_topic_clustering: bool = True          # Use AgglomerativeClustering for topic boundaries (strengthens boundary confidence)
    # LLM segmentation refinement (runs by default when LLM backend is configured)
    use_llm_refinement: bool = True
    llm_refinement_mode: str = 'full'  # 'boundary_review', 'context_expansion', 'coherence_check', 'full'
    llm_ambiguity_threshold: float = 0.15      # Similarity-threshold proximity for "ambiguous" boundaries
    llm_batch_size: int = 5                    # Boundaries/pairs per LLM call
    # Verbose process logging
    verbose_segmentation: bool = True         # Write process_log.txt with every LLM I/O and step


@dataclass
class SpeakerFilterConfig:
    """Controls which speakers' segments are sent to the classifier.

    mode:
        'none'    — classify all segments (default with conversational segmenter)
        'exclude' — drop segments whose *sole* speaker is in ``speakers``
                    (multi-speaker segments are always kept)
        'isolate' — keep only segments that include at least one speaker
                    from ``speakers``
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
class ConfidenceTierConfig:
    """Configurable thresholds for confidence tier assignment."""
    high_consistency: int = 3
    high_confidence: float = 0.8
    medium_min_consistency: int = 2
    medium_min_confidence: float = 0.6


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    # Input
    transcript_dir: str = './data/input/diarized_sessions/'
    trial_id: str = 'standard'

    # Output
    output_dir: str = './data/output/'

    # Run mode
    run_mode: str = 'auto'  # 'auto', 'interactive', or 'review'

    # Feature flags
    run_theme_labeler: bool = True
    run_codebook_classifier: bool = False

    # Sub-configs
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    speaker_filter: SpeakerFilterConfig = field(default_factory=SpeakerFilterConfig)
    theme_classification: ThemeClassificationConfig = field(default_factory=ThemeClassificationConfig)
    codebook_embedding: EmbeddingClassifierConfig = field(default_factory=EmbeddingClassifierConfig)
    codebook_llm: LLMCodebookConfig = field(default_factory=LLMCodebookConfig)
    codebook_ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    confidence_tiers: ConfidenceTierConfig = field(default_factory=ConfidenceTierConfig)

    # Resume from checkpoint
    resume_from: Optional[str] = None

    # Speaker anonymization key import (for seeding participant IDs)
    speaker_anonymization_key_path: Optional[str] = None

    # Post-pipeline results analysis
    auto_analyze: bool = False

    # Downstream
    autoresearch_dir: str = '../autoresearch/'

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
            'codebook_embedding': EmbeddingClassifierConfig,
            'codebook_llm': LLMCodebookConfig,
            'codebook_ensemble': EnsembleConfig,
            'validation': ValidationConfig,
            'confidence_tiers': ConfidenceTierConfig,
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
