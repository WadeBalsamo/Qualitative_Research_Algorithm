"""
gnn_layer/graph_builder.py
--------------------------
Build the heterogeneous QRA graph from the assembled master DataFrame (SCAFFOLD).

Node types:
  participant_segment, therapist_segment,
  vaamr_stage (5), purer_move (5), vce_code (N), microskill (8)   [anchors]

Edge types:
  - temporal_chain:  segment -> next segment within a session (encodes FROM->CUE->TO
                     directly in the graph). Reuse the timestamp-window logic proven
                     in analysis/purer_analysis.compute_cue_block_purer_profiles.
  - anchor_label:    segment -> assigned construct anchor, weighted by confidence /
                     agreement_fraction.
  - knn_similarity:  segment <-> k nearest segments in Qwen3 space (sklearn
                     NearestNeighbors) — the inductive attachment path for unseen
                     segments / new cohorts.
  - cross_framework: vaamr_stage <-> vce_code and purer_move <-> microskill, weighted
                     by empirical lift computed from df_all (ablatable, Capability D).

Returns a lightweight ``HeteroGraph`` of torch tensors — NO torch-geometric
dependency. torch is imported lazily inside the build function.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch
    from .config import GnnLayerConfig


@dataclass
class HeteroGraph:
    """Container of typed node features and typed edge indices (torch tensors).

    node_features: {node_type: FloatTensor[n_type, D]}
    node_ids:      {node_type: List[str]}  (segment_id or construct id, row-aligned)
    edges:         {(src_type, relation, dst_type): LongTensor[2, E]}
    edge_weights:  {(src_type, relation, dst_type): FloatTensor[E]}
    meta:          free-form dict (n per type, embedding dim, etc.)
    """
    node_features: Dict[str, "torch.Tensor"] = field(default_factory=dict)
    node_ids: Dict[str, List[str]] = field(default_factory=dict)
    edges: Dict[Tuple[str, str, str], "torch.Tensor"] = field(default_factory=dict)
    edge_weights: Dict[Tuple[str, str, str], "torch.Tensor"] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def build_graph(
    df_all,
    segment_embeddings: Dict[str, "np.ndarray"],
    config: "GnnLayerConfig",
    framework: Optional[dict] = None,
) -> HeteroGraph:
    """Assemble the full heterogeneous graph from the corpus.

    TODO(scaffold): construct node tensors from segment_embeddings + anchor
    embeddings; build the four edge families described in the module docstring;
    honor config.include_*_nodes and config.knn_k / cross_framework_min_lift.
    """
    raise NotImplementedError("gnn_layer.graph_builder.build_graph: scaffold")


def attach_new_segments(
    graph: HeteroGraph,
    new_embeddings: Dict[str, "np.ndarray"],
    config: "GnnLayerConfig",
) -> HeteroGraph:
    """Inductively attach unseen segments to a frozen graph via kNN edges.

    New segment nodes connect to the existing anchor/labeled nodes only (never to
    each other), so predictions stay order-invariant and reproducible.

    TODO(scaffold): kNN from new_embeddings into graph segment nodes; append nodes
    and edges; return the extended graph.
    """
    raise NotImplementedError("gnn_layer.graph_builder.attach_new_segments: scaffold")


def compute_cross_framework_lift(df_all, min_lift: float = 1.5) -> Dict[Tuple[str, str], float]:
    """Empirical (vaamr_stage, vce_code) and (purer_move, microskill) lift for edges.

    TODO(scaffold): reuse the lift formula P(b|a)/P(b) over df_all label columns.
    """
    raise NotImplementedError(
        "gnn_layer.graph_builder.compute_cross_framework_lift: scaffold"
    )
