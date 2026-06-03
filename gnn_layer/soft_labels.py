"""
gnn_layer/soft_labels.py
------------------------
Reconstruct soft VAAMR supervision targets from multi-run ballots (SCAFFOLD).

Capability A. The pipeline already stores, per participant segment, the per-rater
ballots in ``rater_votes`` (schema documented at
classification_tools/data_structures.py:60-65):
    {rater, vote, stage, confidence, secondary_stage, secondary_confidence, justification}

This module turns those ballots into a normalized 5-vector mixture over VAAMR stages
(the superposition signal that majority-vote throws away) and a scalar progression
coordinate E[stage] = sum_k k * p_k. Where a segment is in the human-coded subset,
its human label upgrades / replaces the weak target.

Pure-Python + numpy; no torch needed here.
"""

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

N_VAAMR_STAGES = 5  # 0=Vigilance .. 4=Reappraisal


def ballots_to_mixture(rater_votes: Optional[List[Dict]],
                       n_stages: int = N_VAAMR_STAGES,
                       secondary_weight: float = 0.5) -> "np.ndarray":
    """Convert a segment's ``rater_votes`` list into a normalized stage mixture.

    Counts each rater's primary stage (weight 1.0) and optionally its secondary
    stage (weight ``secondary_weight``), weighting by confidence when present, then
    L1-normalizes to a probability vector of length ``n_stages``. Returns a uniform
    vector if no usable ballots are present (caller may prefer a final_label one-hot).

    TODO(scaffold): implement counting/normalization.
    """
    raise NotImplementedError("gnn_layer.soft_labels.ballots_to_mixture: scaffold")


def mixture_to_progression(mixture: "np.ndarray") -> float:
    """Expected stage value sum_k k * p_k — the continuous progression coordinate.

    TODO(scaffold): implement dot(arange(n), mixture).
    """
    raise NotImplementedError("gnn_layer.soft_labels.mixture_to_progression: scaffold")


def build_soft_targets(df_all, label_mode: str = 'weak') -> Dict[str, "np.ndarray"]:
    """Return {segment_id: mixture_vector} for participant segments.

    label_mode:
      'weak'  — from rater_votes (fallback: one-hot of final_label)
      'human' — only segments with in_human_coded_subset; one-hot of human_label
      'self_supervised' — empty dict (no label targets; link-prediction only)

    TODO(scaffold): parse df_all['rater_votes'] (JSON string per analysis/loader),
    call ballots_to_mixture per row, honor label_mode.
    """
    raise NotImplementedError("gnn_layer.soft_labels.build_soft_targets: scaffold")
