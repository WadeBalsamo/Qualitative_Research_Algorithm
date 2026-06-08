"""
classification_tools
--------------------
The pipeline's classifiers, plus the shared infrastructure they all build on.
The *definitions* these classifiers assign (VAAMR/PURER frameworks, the VCE
codebook) live separately in the ``constructs`` package.

Shared infrastructure (this package's top level):
    data_structures (Segment) · llm_client · classification_loop ·
    majority_vote · reliability · response_parser · validation ·
    zeroshot_reporting

Classifier subpackages — note the label cardinality of each:
    theme_llm/            PRIMARY path. SINGLE-LABEL zero-shot LLM theme
                          classification for any ThemeFramework (VAAMR
                          participant stages, PURER therapist moves) + PURER
                          cue units. (Also hosts LLMCodebookClassifier, the
                          LLM arm of the codebook — see below.)
    codebook_multilabel/  MULTI-LABEL codebook (VCE) classifier: embedding
                          similarity + ensemble reconciliation. The multi-label
                          counterpart to theme_llm's single-label path.
    probe/                SINGLE-LABEL, LLM-free VAAMR scaler (per-rater probe
                          ensemble; gated, abstains; ranked below the LLM).

The GNN consensus-distillation classifier (DEFAULT OFF) is a fourth member of
this family but lives in ``gnn_layer/classifier/`` because it shares the
gnn_layer embedding/graph substrate.
"""

__version__ = "0.2.0"

from .data_structures import Segment
from .validation import create_balanced_evaluation_set
from .llm_client import LLMClient, LLMClientConfig, extract_json
from .classification_loop import filter_participant_segments, classify_segments
from .majority_vote import vote_single_label, vote_multi_label, ABSTAIN
from .theme_llm.llm_classifier import (
    classify_segments_zero_shot,
    classify_purer_cue_units,
    _build_context_block,
    create_content_validity_test_set,
    LLMCodebookClassifier,
)
from .response_parser import parse_all_results
