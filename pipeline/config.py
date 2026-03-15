"""
config.py
---------
Top-level pipeline configuration.

Aggregates sub-configs from shared, theme_labeler, and codebook_classifier.
"""

from dataclasses import dataclass, field
from typing import Optional

from theme_labeler.config import ThemeClassificationConfig
from codebook_classifier.config import (
    EmbeddingClassifierConfig,
    LLMCodebookConfig,
    EnsembleConfig,
)


@dataclass
class SegmentationConfig:
    """Parameters for transcript segmentation."""
    embedding_model: str = 'all-MiniLM-L6-v2'
    min_segment_words: int = 30
    max_segment_words: int = 200
    silence_threshold_ms: int = 1500
    semantic_shift_percentile: int = 25


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

    # Feature flags
    run_theme_labeler: bool = True
    run_codebook_classifier: bool = False

    # Sub-configs
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    theme_classification: ThemeClassificationConfig = field(default_factory=ThemeClassificationConfig)
    codebook_embedding: EmbeddingClassifierConfig = field(default_factory=EmbeddingClassifierConfig)
    codebook_llm: LLMCodebookConfig = field(default_factory=LLMCodebookConfig)
    codebook_ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    confidence_tiers: ConfidenceTierConfig = field(default_factory=ConfidenceTierConfig)

    # Resume from checkpoint
    resume_from: Optional[str] = None

    # Downstream
    autoresearch_dir: str = '../autoresearch/'
