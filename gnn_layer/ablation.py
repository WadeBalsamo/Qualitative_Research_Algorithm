"""
gnn_layer/ablation.py
---------------------
Capability D — principled ablation & sub-typing (SCAFFOLD).

Turns prose-level design questions into empirical ones, as a byproduct of the single
graph:

  - is_vce_superfluous(): re-train with VCE nodes/edges ablated; report the change in
    superposition / transition-prediction quality. A negligible delta is evidence VCE
    is superfluous; a meaningful drop is evidence it carries signal.
  - purer / microskill ablations: do these distinctions add signal beyond emergent
    motifs?
  - within_stage / within_move sub-types: cluster segment embeddings inside each VAAMR
    stage and each PURER move to surface emergent sub-types from the data (not imposed).
"""

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .graph_builder import HeteroGraph
    from .config import GnnLayerConfig


def run_ablation(
    df_all,
    base_graph: "HeteroGraph",
    config: "GnnLayerConfig",
    ablate: str,
) -> dict:
    """Re-train with one construct family removed; return metric deltas vs the base.

    ``ablate`` in {'vce', 'purer', 'microskill'}.

    TODO(scaffold): rebuild graph without the named node/edge family, train, diff
    metrics against the full model.
    """
    raise NotImplementedError("gnn_layer.ablation.run_ablation: scaffold")


def discover_subtypes(
    segment_embeddings: Dict[str, "list"],
    df_all,
    by: str = 'vaamr_stage',
    k_per_group: int = 3,
) -> dict:
    """Cluster segment embeddings within each stage/move to surface emergent sub-types.

    ``by`` in {'vaamr_stage', 'purer_move'}.

    TODO(scaffold): group rows; KMeans within group; return cluster exemplars/sizes.
    """
    raise NotImplementedError("gnn_layer.ablation.discover_subtypes: scaffold")
