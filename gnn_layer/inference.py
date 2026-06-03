"""
gnn_layer/inference.py
----------------------
Per-segment GNN outputs (SCAFFOLD).

Produces, for every segment, the learned embedding plus (participant) the VAAMR
mixture vector and progression coordinate, and (therapist) microskill logits. These
are written to ``03_analysis_data/gnn/`` keyed by segment_id — NOT folded into
master_segments (the GNN runs after assembly; re-assembly is out of scope).
"""

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from .graph_builder import HeteroGraph
    from .config import GnnLayerConfig


def infer_segment_positions(model, graph: "HeteroGraph", config: "GnnLayerConfig") -> dict:
    """Run a forward pass and return per-segment outputs.

    Returns a dict:
      {'segment_id': [...],
       'vaamr_mixture': np.ndarray[N,5],
       'progression_coord': np.ndarray[N],
       'gnn_embedding': np.ndarray[N,H],
       'microskill_logits': np.ndarray[M,8]}  (therapist subset)

    TODO(scaffold): model.eval(); forward(graph); softmax soft_vaamr; E[stage];
    collect embeddings.
    """
    raise NotImplementedError("gnn_layer.inference.infer_segment_positions: scaffold")


def cue_block_embeddings(
    df_all,
    segment_gnn_embeddings: Dict[str, "np.ndarray"],
) -> dict:
    """Mean-pool therapist-segment GNN embeddings into one vector per cue block.

    Reuses analysis.purer_analysis.compute_cue_block_purer_profiles to define the
    cue blocks (FROM->CUE->TO), then averages the GNN embeddings of the therapist
    segments inside each block. Returns block-level arrays for Capability B.

    TODO(scaffold): call compute_cue_block_purer_profiles(df_all); map block
    therapist segments -> embeddings -> mean.
    """
    raise NotImplementedError("gnn_layer.inference.cue_block_embeddings: scaffold")
