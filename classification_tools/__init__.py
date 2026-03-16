"""
classification_tools
--------------------
Common data structures, validation utilities, LLM API clients,
and classification logic used across the pipeline.
"""

__version__ = "0.2.0"

from .data_structures import Segment, SpeakerRun
from .validation import create_balanced_evaluation_set
from .llm_client import LLMClient, LLMClientConfig, extract_json
from .classification_loop import filter_participant_segments, classify_segments
from .majority_vote import single_label_majority_vote, multi_label_majority_vote
from .llm_classifier import (
    classify_segments_zero_shot,
    create_content_validity_test_set,
    LLMCodebookClassifier,
)
from .response_parser import parse_all_results
