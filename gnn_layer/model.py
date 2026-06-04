"""
gnn_layer/model.py
------------------
Pure-PyTorch GraphSAGE + multi-task heads for the QRA GNN layer.

NO torch-geometric: mean aggregation is implemented with ``index_add_`` (scatter).
Heads (on node embeddings): soft_vaamr (5, KL to ballot mixture), progression (1,
MSE to E[stage]), vce_multilabel (BCE), purer (CE, 5), microskill_multilabel (BCE, 8).
Auxiliary losses: supervised contrastive (InfoNCE) + temporal-chain link prediction.

torch imported at construction (the package still imports lazily because this module's
torch import is inside the functions/classes that are only built when used).
"""

from typing import Dict, List, Optional


def _torch():
    import torch
    return torch


def scatter_sum(src, index, dim_size):
    import torch
    out = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


def scatter_weighted_mean(src, index, weight, dim_size):
    """(sum_j w_ij x_j) / (sum_j w_ij) bucketed by destination ``index``."""
    import torch
    eps = 1e-12
    num = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
    num.index_add_(0, index, src * weight.unsqueeze(1))
    den = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    den.index_add_(0, index, weight)
    return num / den.clamp(min=eps).unsqueeze(1)


def _build_modules():
    """Define nn.Module classes lazily (so importing the file needs no torch)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SAGEConv(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.lin_self = nn.Linear(in_dim, out_dim)
            self.lin_neigh = nn.Linear(in_dim, out_dim)

        def forward(self, x, edge_index, edge_weight=None):
            src, dst = edge_index[0], edge_index[1]
            if edge_index.numel() == 0:
                agg = torch.zeros(x.size(0), x.size(1), dtype=x.dtype, device=x.device)
            else:
                w = edge_weight if edge_weight is not None else torch.ones(
                    src.size(0), dtype=x.dtype, device=x.device)
                agg = scatter_weighted_mean(x[src], dst, w, x.size(0))
            return self.lin_self(x) + self.lin_neigh(agg)

    class MultiTaskGNN(nn.Module):
        def __init__(self, in_dim, hidden_dim, n_layers, dropout, head_sizes: Dict[str, int]):
            super().__init__()
            self.convs = nn.ModuleList()
            d = in_dim
            for _ in range(max(1, n_layers)):
                self.convs.append(SAGEConv(d, hidden_dim))
                d = hidden_dim
            self.dropout = dropout
            self.heads = nn.ModuleDict(
                {name: nn.Linear(hidden_dim, size) for name, size in head_sizes.items()}
            )

        def encode(self, x, edge_index, edge_weight=None):
            h = x
            for conv in self.convs:
                h = conv(h, edge_index, edge_weight)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            return h

        def forward(self, x, edge_index, edge_weight=None):
            h = self.encode(x, edge_index, edge_weight)
            out = {'emb': h}
            for name, head in self.heads.items():
                out[name] = head(h)
            return out

    return SAGEConv, MultiTaskGNN


def build_model(graph, config, n_vce: int = 0, n_microskill: int = 8):
    """Construct the encoder + the heads implied by ``config.objectives``."""
    _, MultiTaskGNN = _build_modules()
    in_dim = int(graph.meta['embed_dim'])
    head_sizes: Dict[str, int] = {}
    obj = set(config.objectives)
    if 'soft_vaamr' in obj:
        head_sizes['soft_vaamr'] = 5
    if 'progression' in obj:
        head_sizes['progression'] = 1
    if 'vce_multilabel' in obj and n_vce > 0:
        head_sizes['vce'] = n_vce
    if 'purer' in obj:
        head_sizes['purer'] = 5
    if 'microskill_multilabel' in obj:
        head_sizes['microskill'] = n_microskill
    if not head_sizes:
        head_sizes['soft_vaamr'] = 5  # always have at least one head
    model = MultiTaskGNN(in_dim, int(config.hidden_dim), int(config.n_layers),
                         float(config.dropout), head_sizes)
    return model


def supervised_contrastive(emb, labels, temperature=0.1):
    """SupCon loss (Khosla et al.) over labeled node embeddings. Returns scalar."""
    import torch
    import torch.nn.functional as F
    if emb.size(0) < 2:
        return emb.new_zeros(())
    z = F.normalize(emb, dim=1)
    sim = z @ z.t() / temperature
    n = z.size(0)
    mask_self = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask_self, float('-inf'))
    labels = labels.view(-1, 1)
    pos = (labels == labels.t()) & (~mask_self)
    logp = F.log_softmax(sim, dim=1)
    # The masked (-inf) diagonal gives logp=-inf there; zero it so the 0*-inf=nan
    # that would arise from (logp * pos) at excluded positions cannot occur.
    logp = logp.masked_fill(mask_self, 0.0)
    pos_counts = pos.sum(1)
    valid = pos_counts > 0
    if valid.sum() == 0:
        return emb.new_zeros(())
    loss = -(logp * pos).sum(1)[valid] / pos_counts[valid].clamp(min=1)
    return loss.mean()


def link_prediction_loss(emb, pos_edges, neg_edges):
    """Binary link prediction on temporal-chain edges (positives) vs random (negatives)."""
    import torch
    import torch.nn.functional as F
    if pos_edges.numel() == 0 or neg_edges.numel() == 0:
        return emb.new_zeros(())
    def _score(e):
        return (emb[e[0]] * emb[e[1]]).sum(-1)
    pos = _score(pos_edges)
    neg = _score(neg_edges)
    logits = torch.cat([pos, neg])
    target = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    return F.binary_cross_entropy_with_logits(logits, target)


def compute_losses(outputs, targets, config) -> Dict[str, "object"]:
    """Weighted sum of the enabled task losses. Returns {'total': scalar, <term>: scalar}.

    ``targets`` is a dict of precomputed tensors (assembled in train.py):
      vaamr_idx, vaamr_mix; (progression reuses vaamr_idx + prog_val);
      vce_idx,vce_target; purer_idx,purer_label; micro_idx,micro_target;
      contrast_idx,contrast_label; pos_edges,neg_edges.
    """
    import torch
    import torch.nn.functional as F
    losses: Dict[str, object] = {}
    emb = outputs['emb']
    zero = emb.new_zeros(())

    if 'soft_vaamr' in outputs and 'vaamr_idx' in targets and targets['vaamr_idx'].numel():
        idx = targets['vaamr_idx']
        logp = F.log_softmax(outputs['soft_vaamr'][idx], dim=1)
        losses['soft_vaamr'] = F.kl_div(logp, targets['vaamr_mix'], reduction='batchmean')

    if 'progression' in outputs and 'vaamr_idx' in targets and targets['vaamr_idx'].numel():
        idx = targets['vaamr_idx']
        pred = outputs['progression'][idx].squeeze(-1)
        losses['progression'] = F.mse_loss(pred, targets['prog_val'])

    if 'vce' in outputs and targets.get('vce_idx') is not None and targets['vce_idx'].numel():
        idx = targets['vce_idx']
        losses['vce'] = F.binary_cross_entropy_with_logits(outputs['vce'][idx], targets['vce_target'])

    if 'purer' in outputs and targets.get('purer_idx') is not None and targets['purer_idx'].numel():
        idx = targets['purer_idx']
        losses['purer'] = F.cross_entropy(outputs['purer'][idx], targets['purer_label'])

    if 'microskill' in outputs and targets.get('micro_idx') is not None and targets['micro_idx'].numel():
        idx = targets['micro_idx']
        losses['microskill'] = F.binary_cross_entropy_with_logits(outputs['microskill'][idx], targets['micro_target'])

    if 'contrastive' in config.objectives and targets.get('contrast_idx') is not None and targets['contrast_idx'].numel() > 1:
        losses['contrastive'] = supervised_contrastive(
            emb[targets['contrast_idx']], targets['contrast_label'], config.contrastive_temp)

    if 'link_prediction' in config.objectives and targets.get('pos_edges') is not None:
        losses['link_prediction'] = link_prediction_loss(
            emb, targets['pos_edges'], targets['neg_edges'])

    total = zero
    for v in losses.values():
        total = total + v
    losses['total'] = total
    return losses
