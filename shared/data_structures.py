"""
data_structures.py
------------------
Core data structures for the qualitative research classification pipeline.

The Segment dataclass is the atomic unit shared across both the
theme_labeler (single-label stage classification) and codebook_classifier
(multi-label phenomenology coding) modules.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class Segment:
    """
    A single transcript segment -- the atomic unit of classification.

    Identity, temporal, speaker/text, label, validation, and final-label
    fields follow the master segment dataset schema.

    The ``primary_stage`` / ``secondary_stage`` fields hold theme labels
    (e.g. VA-MR stage IDs).  The ``codebook_labels_*`` fields hold
    multi-label codebook assignments.
    """
    # Identity fields
    segment_id: str = ""
    trial_id: str = ""
    participant_id: str = ""
    session_id: str = ""
    session_number: int = 0

    # Temporal fields
    segment_index: int = 0
    start_time_ms: int = 0
    end_time_ms: int = 0
    total_segments_in_session: int = 0

    # Speaker and text fields
    speaker: str = ""
    text: str = ""
    word_count: int = 0

    # Theme label fields (populated by theme_labeler zero-shot classification)
    primary_stage: Optional[int] = None
    secondary_stage: Optional[int] = None
    llm_confidence_primary: Optional[float] = None
    llm_confidence_secondary: Optional[float] = None
    llm_justification: Optional[str] = None
    llm_run_consistency: Optional[int] = None

    # Multi-model cross-referencing fields (populated when using multiple models)
    model_agreement: Optional[str] = None  # 'unanimous', 'majority', 'split', or 'none'
    model_predictions: Optional[Dict] = None  # Per-model predictions
    total_models: Optional[int] = None  # Number of models used

    # Codebook label fields (populated by codebook_classifier)
    codebook_labels_embedding: Optional[List[str]] = None
    codebook_labels_llm: Optional[List[str]] = None
    codebook_labels_ensemble: Optional[List[str]] = None
    codebook_disagreements: Optional[List[str]] = None
    codebook_confidence: Optional[Dict[str, float]] = None

    # Validation fields (populated after human coding comparison)
    human_label: Optional[int] = None
    human_secondary_label: Optional[int] = None
    adjudicated_label: Optional[int] = None
    in_human_coded_subset: bool = False
    label_status: str = "llm_only"

    # Final training label (populated by dataset assembly)
    final_label: Optional[int] = None
    final_label_source: Optional[str] = None
    label_confidence_tier: Optional[str] = None


@dataclass
class SpeakerRun:
    """Contiguous block of speech by one speaker."""
    speaker: str
    sentences: list
