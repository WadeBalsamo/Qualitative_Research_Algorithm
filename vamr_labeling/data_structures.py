"""
data_structures.py
------------------
Core data structures for the VA-MR labeling pipeline.

The Segment dataclass parallels srl_constructs.py's construct definition
pattern: a structured data object with all fields needed downstream.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Segment:
    """
    A single transcript segment -- the atomic unit of VA-MR classification.

    Identity, temporal, speaker/text, label, validation, and final-label
    fields follow the master segment dataset schema specified in the task.
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

    # Label fields (populated by zero-shot classification)
    primary_stage: Optional[int] = None
    secondary_stage: Optional[int] = None
    llm_confidence_primary: Optional[float] = None
    llm_confidence_secondary: Optional[float] = None
    llm_justification: Optional[str] = None
    llm_run_consistency: Optional[int] = None

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
