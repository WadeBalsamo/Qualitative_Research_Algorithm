"""
shared
------
Common data structures, validation utilities, and LLM API clients
used by both the codebook_classifier and theme_labeler modules.
"""

__version__ = "0.2.0"

from .data_structures import Segment, SpeakerRun
from .validation import create_balanced_evaluation_set
from .llm_client import LLMClient, LLMClientConfig, extract_json
from .classification_loop import filter_participant_segments, classify_segments
from .majority_vote import single_label_majority_vote, multi_label_majority_vote
