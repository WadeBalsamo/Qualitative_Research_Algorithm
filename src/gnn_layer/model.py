"""
gnn_layer/model.py
------------------
Pure-PyTorch GraphSAGE + multi-task heads for the QRA GNN layer.

NO torch-geometric: mean aggregation is implemented with ``index_add_`` (scatter).
Heads (on node embeddings): soft_vaamr (5, KL to ballot mixture), progression (1,
MSE to E[stage]), vce_multilabel (BCE), purer (CE, 5).
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
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # softplus(b)=1 at this bias, so a freshly-built gate leaves aggregation identical
    # to the fixed-weight path until training moves it (backward-compatible init).
    _NEUTRAL_GATE = math.log(math.e - 1.0)

    class SAGEConv(nn.Module):
        def __init__(self, in_dim, out_dim, n_edge_types: int = 0):
            super().__init__()
            self.lin_self = nn.Linear(in_dim, out_dim)
            self.lin_neigh = nn.Linear(in_dim, out_dim)
            # Learnable per-edge-type gate (Track A1). n_edge_types==0 → no gate, the
            # aggregation is byte-identical to the original fixed-weight SAGEConv.
            self.n_edge_types = int(n_edge_types)
            if self.n_edge_types > 0:
                self.edge_type_gate = nn.Parameter(
                    torch.full((self.n_edge_types,), _NEUTRAL_GATE))
            else:
                self.edge_type_gate = None

        def forward(self, x, edge_index, edge_weight=None, edge_type_ids=None):
            src, dst = edge_index[0], edge_index[1]
            if edge_index.numel() == 0:
                agg = torch.zeros(x.size(0), x.size(1), dtype=x.dtype, device=x.device)
            else:
                w = edge_weight if edge_weight is not None else torch.ones(
                    src.size(0), dtype=x.dtype, device=x.device)
                if self.edge_type_gate is not None and edge_type_ids is not None:
                    gate = F.softplus(self.edge_type_gate)[edge_type_ids]
                    w = w * gate
                agg = scatter_weighted_mean(x[src], dst, w, x.size(0))
            return self.lin_self(x) + self.lin_neigh(agg)

    class MultiTaskGNN(nn.Module):
        def __init__(self, in_dim, hidden_dim, n_layers, dropout, head_sizes: Dict[str, int],
                     n_edge_types: int = 0):
            super().__init__()
            self.convs = nn.ModuleList()
            d = in_dim
            for _ in range(max(1, n_layers)):
                self.convs.append(SAGEConv(d, hidden_dim, n_edge_types=n_edge_types))
                d = hidden_dim
            self.dropout = dropout
            self.heads = nn.ModuleDict(
                {name: nn.Linear(hidden_dim, size) for name, size in head_sizes.items()}
            )

        def encode(self, x, edge_index, edge_weight=None, edge_type_ids=None):
            h = x
            for conv in self.convs:
                h = conv(h, edge_index, edge_weight, edge_type_ids)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            return h

        def forward(self, x, edge_index, edge_weight=None, edge_type_ids=None):
            h = self.encode(x, edge_index, edge_weight, edge_type_ids)
            out = {'emb': h}
            for name, head in self.heads.items():
                out[name] = head(h)
            return out

    return SAGEConv, MultiTaskGNN


def build_model(graph, config, n_vce: int = 0):
    """Construct the encoder + the heads implied by ``config.objectives``."""
    _, MultiTaskGNN = _build_modules()
    in_dim = int(graph.meta['embed_dim'])
    # Learnable per-edge-type gates activate only with typed precipitates edges (Track A1);
    # otherwise n_edge_types=0 keeps the model parameter-identical to the fixed-weight path.
    n_edge_types = (len(graph.meta.get('edge_type_vocab', []))
                    if getattr(config, 'precipitates_edges', False) else 0)
    # soft-VAAMR head width: 5 (default, byte-identical) or 6 when the "No code" class
    # is enabled (config.vaamr_n_classes == 6). All other heads are unchanged.
    n_vaamr = int(getattr(config, 'vaamr_n_classes', 5) or 5)
    head_sizes: Dict[str, int] = {}
    obj = set(config.objectives)
    if 'soft_vaamr' in obj:
        head_sizes['soft_vaamr'] = n_vaamr
    if 'progression' in obj:
        head_sizes['progression'] = 1
    if 'vce_multilabel' in obj and n_vce > 0:
        head_sizes['vce'] = n_vce
    if 'purer' in obj:
        head_sizes['purer'] = 5
    if not head_sizes:
        head_sizes['soft_vaamr'] = n_vaamr  # always have at least one head
    model = MultiTaskGNN(in_dim, int(config.hidden_dim), int(config.n_layers),
                         float(config.dropout), head_sizes, n_edge_types=n_edge_types)
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


def _batch_class_counts(mix):
    """Count of rows whose dominant (argmax) class is each class, over the batch.

    ``mix`` is the [N, C] soft-target matrix for the labeled VAAMR nodes currently in
    the loss (already fold-subset during cross-validation), so these are the training
    batch's class frequencies — exactly what the rebalancing flags key on.
    """
    import torch
    labels = mix.argmax(dim=1)
    return torch.bincount(labels, minlength=mix.size(1)).to(mix.dtype)


def _inv_freq_row_weights(mix):
    """Per-row inverse-frequency weight keyed by each row's dominant class.

    weight_i = N / count(argmax_i); absent classes contribute nothing. Normalized to
    mean 1 over the batch rows so the overall loss scale matches the unweighted mean
    while mass is redistributed toward rare classes.
    """
    import torch
    labels = mix.argmax(dim=1)
    counts = _batch_class_counts(mix)
    n = max(int(labels.numel()), 1)
    inv = mix.new_zeros(mix.size(1))
    nz = counts > 0
    inv[nz] = n / counts[nz]
    w = inv[labels]
    return w / w.mean().clamp(min=1e-12)


def _soft_vaamr_loss(logits, mix, config):
    """Soft-VAAMR KL with optional class-balance / focal / logit-adjustment (TAM).

    With no flags set this returns EXACTLY the original
    ``F.kl_div(F.log_softmax(logits), mix, reduction='batchmean')`` — byte-identical.
    When any flag is active the per-row KL is computed with ``reduction='none'`` and a
    per-row weight (class-balance * focal) is applied before a mean reduction (which
    equals batchmean when the weight is all-ones), and TAM shifts the logits by
    log(class_prior) at train time. 5- and 6-class compatible (everything keys off the
    width of ``mix``).
    """
    import torch
    import torch.nn.functional as F
    balance = bool(getattr(config, 'vaamr_class_balance', False))
    gamma = float(getattr(config, 'vaamr_focal_gamma', 0.0) or 0.0)
    tam = bool(getattr(config, 'vaamr_tam', False))

    if not balance and gamma <= 0.0 and not tam:
        # Fast path — identical computation to the original unweighted batchmean KL.
        logp = F.log_softmax(logits, dim=1)
        return F.kl_div(logp, mix, reduction='batchmean')

    # Logit adjustment (TAM): add log(class prior) to the logits at train time so rare
    # classes carry an effective margin (inference still reads the raw logits).
    adj = logits
    if tam:
        counts = _batch_class_counts(mix)
        prior = counts / counts.sum().clamp(min=1.0)
        adj = logits + torch.log(prior.clamp(min=1e-12)).unsqueeze(0)

    logp = F.log_softmax(adj, dim=1)
    kl_row = F.kl_div(logp, mix, reduction='none').sum(dim=1)        # [N] per-row KL
    weight = torch.ones_like(kl_row)
    if balance:
        weight = weight * _inv_freq_row_weights(mix)
    if gamma > 0.0:
        # p_true = the model's softmax mass on each row's argmax-target class (RAW
        # logits — the model's actual confidence, independent of the TAM shift).
        labels = mix.argmax(dim=1)
        p_true = F.softmax(logits, dim=1).gather(1, labels.unsqueeze(1)).squeeze(1)
        weight = weight * (1.0 - p_true).clamp(min=0.0).pow(gamma)
    return (weight * kl_row).mean()


def _vaamr_hard_ce_loss(logits, mix):
    """Class-weighted hard-label CE on the labeled VAAMR nodes.

    target = argmax of the soft mixture; class weight = inverse batch frequency. A hard
    classification signal to sit alongside the soft KL. 5- and 6-class compatible.
    """
    import torch
    import torch.nn.functional as F
    labels = mix.argmax(dim=1)
    counts = _batch_class_counts(mix)
    n = max(int(labels.numel()), 1)
    w = logits.new_zeros(mix.size(1))
    nz = counts > 0
    w[nz] = n / counts[nz]
    return F.cross_entropy(logits, labels, weight=w)


def compute_losses(outputs, targets, config) -> Dict[str, "object"]:
    """Weighted sum of the enabled task losses. Returns {'total': scalar, <term>: scalar}.

    ``targets`` is a dict of precomputed tensors (assembled in train.py):
      vaamr_idx, vaamr_mix; (progression reuses vaamr_idx + prog_val);
      vce_idx,vce_target; purer_idx,purer_label;
      contrast_idx,contrast_label; pos_edges,neg_edges.
    """
    import torch
    import torch.nn.functional as F
    losses: Dict[str, object] = {}
    emb = outputs['emb']
    zero = emb.new_zeros(())

    if 'soft_vaamr' in outputs and 'vaamr_idx' in targets and targets['vaamr_idx'].numel():
        idx = targets['vaamr_idx']
        logits = outputs['soft_vaamr'][idx]
        mix = targets['vaamr_mix']
        losses['soft_vaamr'] = _soft_vaamr_loss(logits, mix, config)
        hce_w = float(getattr(config, 'vaamr_hard_ce_weight', 0.0) or 0.0)
        if hce_w > 0.0:
            losses['vaamr_hard_ce'] = hce_w * _vaamr_hard_ce_loss(logits, mix)

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
