"""
gnn_layer/model.py
------------------
Pure-PyTorch GraphSAGE + multi-task heads for the QRA GNN layer (SCAFFOLD).

NO torch-geometric dependency: mean-aggregation message passing is implemented with
stock torch ops (``torch.zeros(...).index_add_(0, dst, src[edge_src])`` followed by
degree normalization = scatter-mean). This avoids the torch-scatter/torch-sparse
compile fragility against torch 2.11 while preserving the inductive GraphSAGE
behaviour borrowed from CFiCS.

Heads (applied to segment-node embeddings):
  - soft_vaamr            : Linear -> 5 logits; trained with KL to the ballot mixture
  - progression           : Linear -> 1 scalar; regress E[stage]
  - vce_multilabel        : Linear -> |VCE| logits (BCE), participant nodes
  - purer                 : Linear -> 5 logits, therapist nodes
  - microskill_multilabel : Linear -> 8 logits (BCE), therapist nodes

Auxiliary losses:
  - supervised contrastive (InfoNCE; rank-aware variant for the ordered VAAMR arc)
  - self-supervised temporal-chain link prediction (label-free; powers B/C/E)

torch is imported lazily at construction so this module imports without torch loaded.
"""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from .graph_builder import HeteroGraph
    from .config import GnnLayerConfig


def _import_torch():
    import torch  # local import keeps package import light
    return torch


def scatter_mean(src, index, dim_size):
    """Mean-aggregate ``src`` rows into ``dim_size`` buckets given destination ``index``.

    Equivalent to torch_geometric scatter('mean') but implemented with index_add_.

    TODO(scaffold): out = zeros(dim_size, D); out.index_add_(0, index, src);
    counts = zeros(dim_size).index_add_(0, index, ones); return out / counts.clamp(min=1).
    """
    raise NotImplementedError("gnn_layer.model.scatter_mean: scaffold")


class SAGEConv:
    """Single GraphSAGE mean-aggregation layer (pure torch). Built lazily.

    Forward: h_v' = ReLU(W_self h_v + W_neigh * mean_{u in N(v)} h_u).
    """

    def __init__(self, in_dim: int, out_dim: int):
        # TODO(scaffold): build nn.Linear(in_dim,out_dim) self+neigh once torch present.
        raise NotImplementedError("gnn_layer.model.SAGEConv.__init__: scaffold")

    def forward(self, x, edge_index):
        raise NotImplementedError("gnn_layer.model.SAGEConv.forward: scaffold")


def build_model(graph: "HeteroGraph", config: "GnnLayerConfig"):
    """Construct the stacked-SAGE encoder + the enabled task heads (an nn.Module).

    Returns a module whose ``forward(graph)`` yields a dict:
      {'node_emb': {...}, 'soft_vaamr': logits, 'progression': scalar,
       'vce': logits, 'purer': logits, 'microskill': logits}

    TODO(scaffold): assemble nn.ModuleList of SAGEConv + Linear heads per
    config.objectives.
    """
    raise NotImplementedError("gnn_layer.model.build_model: scaffold")


def compute_losses(outputs: dict, targets: dict, config: "GnnLayerConfig"):
    """Weighted sum of the enabled task losses + contrastive + link-prediction.

    TODO(scaffold): KL(soft_vaamr || ballot mixture) + MSE(progression) +
    BCE(vce/microskill) + CE(purer) + InfoNCE contrastive + link-prediction BCE.
    """
    raise NotImplementedError("gnn_layer.model.compute_losses: scaffold")
