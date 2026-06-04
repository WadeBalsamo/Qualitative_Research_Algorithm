"""
gnn_layer/train.py
------------------
Target assembly, training loop, and checkpoint export for the GNN layer.

Full-batch transductive training over the corpus graph with early stopping.
Deterministic given ``config.seed``. Supports the three label modes via the
soft-target dict handed in by the runner.
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple


def set_seed(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def _device(config):
    import torch
    if config.device:
        return torch.device(config.device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def assemble_targets(
    graph,
    soft_targets: Dict[str, "object"],
    config,
    df_all=None,
    vce_codes: Optional[List[str]] = None,
    micro_codes: Optional[List[str]] = None,
) -> dict:
    """Build the tensor dict consumed by model.compute_losses.

    Always builds (for the default objectives): vaamr_idx/vaamr_mix, prog_val,
    contrast_idx/label, and pos_edges/neg_edges. Optionally builds vce/purer/micro
    targets when those objectives are enabled and vocabularies/labels are available.
    """
    import numpy as np
    import torch
    from .soft_labels import mixture_to_progression

    idx_of = graph.index_of
    targets: dict = {}

    # ---- VAAMR soft targets (participant nodes present in soft_targets) ----
    v_idx, v_mix = [], []
    for sid, mix in soft_targets.items():
        if sid in idx_of:
            v_idx.append(idx_of[sid]); v_mix.append(np.asarray(mix, dtype=np.float32))
    if v_idx:
        targets['vaamr_idx'] = torch.tensor(v_idx, dtype=torch.long)
        targets['vaamr_mix'] = torch.tensor(np.stack(v_mix), dtype=torch.float32)
        targets['prog_val'] = torch.tensor(
            [mixture_to_progression(m) for m in v_mix], dtype=torch.float32)
        # contrastive labels = argmax stage among labeled nodes
        targets['contrast_idx'] = targets['vaamr_idx']
        targets['contrast_label'] = torch.tensor(
            [int(np.argmax(m)) for m in v_mix], dtype=torch.long)
    else:
        targets['vaamr_idx'] = torch.zeros((0,), dtype=torch.long)
        targets['vaamr_mix'] = torch.zeros((0, 5), dtype=torch.float32)
        targets['prog_val'] = torch.zeros((0,), dtype=torch.float32)
        targets['contrast_idx'] = torch.zeros((0,), dtype=torch.long)
        targets['contrast_label'] = torch.zeros((0,), dtype=torch.long)

    # ---- link prediction edges (positives = temporal chain) ----
    edge_types = graph.meta.get('edge_types', [])
    ei = graph.edge_index
    pos = []
    if ei.numel():
        cols = ei.t().tolist()
        for c, (a, b) in enumerate(cols):
            if c < len(edge_types) and edge_types[c] == 'temporal' and a < b:
                pos.append((a, b))
    n_nodes = graph.x.size(0)
    if pos:
        pos_t = torch.tensor(pos, dtype=torch.long).t().contiguous()
        q = pos_t.size(1)
        neg_a = torch.randint(0, n_nodes, (q,))
        neg_b = torch.randint(0, n_nodes, (q,))
        targets['pos_edges'] = pos_t
        targets['neg_edges'] = torch.stack([neg_a, neg_b], dim=0)
    else:
        targets['pos_edges'] = torch.zeros((2, 0), dtype=torch.long)
        targets['neg_edges'] = torch.zeros((2, 0), dtype=torch.long)

    # ---- optional supervised heads from df labels ----
    if df_all is not None:
        speaker_of = graph.meta.get('speaker_of', {})

        def _codes(v):
            return v if isinstance(v, list) else []

        row_by_id = {str(r.get('segment_id')): r for _, r in df_all.iterrows()}

        if 'vce_multilabel' in config.objectives and vce_codes:
            cmap = {c: i for i, c in enumerate(vce_codes)}
            idxs, tgts = [], []
            for sid, gi in idx_of.items():
                r = row_by_id.get(sid)
                if r is None or speaker_of.get(sid) != 'participant':
                    continue
                lab = _codes(r.get('codebook_labels_ensemble'))
                vec = np.zeros(len(vce_codes), dtype=np.float32)
                for c in lab:
                    if c in cmap:
                        vec[cmap[c]] = 1.0
                idxs.append(gi); tgts.append(vec)
            if idxs:
                targets['vce_idx'] = torch.tensor(idxs, dtype=torch.long)
                targets['vce_target'] = torch.tensor(np.stack(tgts), dtype=torch.float32)

        if 'purer' in config.objectives:
            idxs, labs = [], []
            for sid, gi in idx_of.items():
                r = row_by_id.get(sid)
                if r is None or speaker_of.get(sid) != 'therapist':
                    continue
                pv = r.get('purer_primary')
                try:
                    pv = int(pv)
                except (ValueError, TypeError):
                    continue
                if 0 <= pv < 5:
                    idxs.append(gi); labs.append(pv)
            if idxs:
                targets['purer_idx'] = torch.tensor(idxs, dtype=torch.long)
                targets['purer_label'] = torch.tensor(labs, dtype=torch.long)

        if 'microskill_multilabel' in config.objectives and micro_codes:
            cmap = {c: i for i, c in enumerate(micro_codes)}
            idxs, tgts = [], []
            for sid, gi in idx_of.items():
                r = row_by_id.get(sid)
                if r is None or speaker_of.get(sid) != 'therapist':
                    continue
                lab = _codes(r.get('microskill_labels_ensemble'))
                vec = np.zeros(len(micro_codes), dtype=np.float32)
                for c in lab:
                    if c in cmap:
                        vec[cmap[c]] = 1.0
                idxs.append(gi); tgts.append(vec)
            if idxs:
                targets['micro_idx'] = torch.tensor(idxs, dtype=torch.long)
                targets['micro_target'] = torch.tensor(np.stack(tgts), dtype=torch.float32)

    return targets


def train_model(graph, targets: dict, config, n_vce: int = 0,
                n_microskill: int = 8) -> Tuple["object", dict]:
    """Train the model; return (trained_module, metrics_dict)."""
    import torch
    from .model import build_model, compute_losses

    set_seed(int(config.seed))
    dev = _device(config)
    model = build_model(graph, config, n_vce=n_vce, n_microskill=n_microskill).to(dev)

    x = graph.x.to(dev)
    edge_index = graph.edge_index.to(dev)
    edge_weight = graph.edge_weight.to(dev) if graph.edge_weight is not None else None
    t = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in targets.items()}

    opt = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    best = float('inf'); best_state = None; bad = 0
    history = []
    model.train()
    for epoch in range(int(config.epochs)):
        opt.zero_grad()
        out = model(x, edge_index, edge_weight)
        losses = compute_losses(out, t, config)
        loss = losses['total']
        loss.backward()
        opt.step()
        lv = float(loss.detach().cpu())
        history.append(lv)
        if lv < best - 1e-4:
            best = lv; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(config.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    metrics = {'best_loss': best, 'epochs_run': len(history),
               'final_terms': {k: float(v.detach().cpu()) for k, v in
                               compute_losses(model(x, edge_index, edge_weight), t, config).items()}}
    return model, metrics


def export_checkpoint(model, config, model_dir: str, metrics: Optional[dict] = None) -> str:
    import torch
    os.makedirs(model_dir, exist_ok=True)
    weights_path = os.path.join(model_dir, 'weights.pt')
    torch.save(model.state_dict(), weights_path)
    manifest = {
        'config': {k: getattr(config, k) for k in config.__dataclass_fields__},
        'seed': int(config.seed),
        'metrics': metrics or {},
    }
    with open(os.path.join(model_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    return weights_path


def load_checkpoint(model_dir: str, graph, config, n_vce: int = 0, n_microskill: int = 8):
    import torch
    from .model import build_model
    model = build_model(graph, config, n_vce=n_vce, n_microskill=n_microskill)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'weights.pt'), map_location='cpu'))
    model.eval()
    return model
