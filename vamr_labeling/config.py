"""
config.py
---------
Configuration for the VA-MR zero-shot labeling pipeline.

Replaces the previous mentalbert_sentence_aqua config with pipeline-specific
settings for transcript ingestion, LLM classification, and dataset assembly.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SegmentationConfig:
    """Parameters for transcript segmentation (Stage 1)."""
    embedding_model: str = 'all-MiniLM-L6-v2'
    min_segment_words: int = 30
    max_segment_words: int = 200
    silence_threshold_ms: int = 1500
    semantic_shift_percentile: int = 25


@dataclass
class ClassificationConfig:
    """Parameters for zero-shot LLM classification (Stage 3)."""
    model: str = 'openai/gpt-4o'
    temperature: float = 0.0
    n_runs: int = 3
    randomize_codebook: bool = True
    api_key: str = field(default_factory=lambda: os.environ.get('OPENROUTER_API_KEY', ''))


@dataclass
class ValidationConfig:
    """Parameters for human validation (Stage 5)."""
    n_per_stage: int = 50
    min_kappa: float = 0.70
    min_agreement: float = 0.75


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    # Input
    transcript_dir: str = './data/input/diarized_sessions/'
    trial_id: str = 'standard_MORE'

    # Output
    output_dir: str = './data/output/'

    # Sub-configs
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # Downstream
    autoresearch_dir: str = '../autoresearch/'


# VA-MR stage labels (shared constant)
NUM_CLASSES = 4
STAGE_LABELS = ['Vigilance', 'Avoidance', 'Metacognition', 'Reappraisal']
