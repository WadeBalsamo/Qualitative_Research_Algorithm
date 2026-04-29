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
    cohort_id: Optional[int] = None
    session_variant: str = ''   # '' for normal sessions, 'a'/'b' for split sessions (e.g. c1s4a)

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

    # Interrater-reliability fields (populated by unified vote_single_label)
    # rater_ids     : list of rater identifiers, one per slot
    # rater_votes   : list of per-rater ballot dicts (see majority_vote.py)
    #                 {rater, vote, stage, confidence, secondary_stage,
    #                  secondary_confidence, justification}
    # agreement_level    : 'unanimous' | 'majority' | 'split' | 'none'
    # agreement_fraction : n_agree / n_raters
    # needs_review       : True when split or all raters errored
    # consensus_vote     : int stage | 'ABSTAIN' | None
    rater_ids: Optional[List[str]] = None
    rater_votes: Optional[List[Dict]] = None
    agreement_level: Optional[str] = None
    agreement_fraction: Optional[float] = None
    needs_review: bool = False
    consensus_vote: Optional[object] = None
    tie_broken_by_confidence: bool = False
    secondary_agreement_level: Optional[str] = None       # 'unanimous'|'majority'|'partial'|'split'|None
    secondary_agreement_fraction: Optional[float] = None  # n_agreeing_with_secondary / n_agreeing

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

    # Conversational segmenter fields
    speakers_in_segment: Optional[List[str]] = None  # All speakers who contributed
    session_file: str = ""  # Source file path (for per-transcript reporting)
