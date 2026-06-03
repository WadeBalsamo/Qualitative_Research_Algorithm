"""
gnn_layer/coupling.py
---------------------
Capability E — inductive participant<->therapist coupling (SCAFFOLD).

Learns the relationship between therapist-language representation (PURER + microskill
+ chain position) and SUBSEQUENT participant VAAMR movement — the methodology's
neurophenomenological mutual-constraints core (Sec 7.6). Extracts latent factors from
the coupling-weighted therapist embeddings and, optionally, interprets them against a
small INLINE Common-Factors / Intervention-Concepts reference lexicon (defined here,
NOT imported from any external dataset). This is the principled home for the dropped
CF/IC constructs: alliance-like factors are *rediscovered* from data, then named
against the lexicon — never imposed as labels.

sklearn imported lazily.
"""

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from .config import GnnLayerConfig


# Inline interpretive lexicon (authored here; used only to NAME discovered factors,
# never to classify). Common Factors (Bordin alliance) + Intervention Concepts.
CF_IC_REFERENCE = {
    'bond': "warmth, empathy, trust, relational connection between therapist and participant",
    'goal_alignment': "shared agreement on the goals/aims of the therapeutic work",
    'task_agreement': "shared agreement on the tasks/methods used to pursue the goals",
    'empathy_acceptance_regard': "empathic, accepting, non-judgmental positive regard",
    'collaboration_partnership': "collaborative, partnership-oriented, autonomy-supportive stance",
}


def fit_coupling_model(
    cue_embeddings: "np.ndarray",
    from_stages: "np.ndarray",
    to_stages: "np.ndarray",
    config: "GnnLayerConfig",
):
    """Model subsequent participant movement (to_stage) from therapist cue embedding.

    TODO(scaffold): from_stage-conditioned regression of stage change on cue embedding.
    """
    raise NotImplementedError("gnn_layer.coupling.fit_coupling_model: scaffold")


def extract_latent_factors(
    cue_embeddings: "np.ndarray",
    coupling_weights: Optional["np.ndarray"],
    config: "GnnLayerConfig",
) -> dict:
    """Decompose coupling-weighted therapist embeddings into latent factors (NMF/PCA).

    TODO(scaffold): sklearn NMF/PCA with config.n_latent_factors; return loadings +
    top exemplar cues per factor.
    """
    raise NotImplementedError("gnn_layer.coupling.extract_latent_factors: scaffold")


def interpret_factors(factors: dict, config: "GnnLayerConfig") -> dict:
    """Name each latent factor against CF_IC_REFERENCE by embedding similarity.

    Returns {factor_id: {nearest_cf_ic, similarity, exemplars}} — interpretation only.

    TODO(scaffold): embed factor exemplars + CF_IC_REFERENCE descriptions; cosine match.
    """
    raise NotImplementedError("gnn_layer.coupling.interpret_factors: scaffold")
