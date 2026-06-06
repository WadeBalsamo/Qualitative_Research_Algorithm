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
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
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
) -> dict:
    """Build the tensor dict consumed by model.compute_losses.

    Always builds (for the default objectives): vaamr_idx/vaamr_mix, prog_val,
    contrast_idx/label, and pos_edges/neg_edges. Optionally builds vce/purer
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

    return targets


def train_model(graph, targets: dict, config, n_vce: int = 0) -> Tuple["object", dict]:
    """Train the model; return (trained_module, metrics_dict)."""
    import torch
    from .model import build_model, compute_losses

    set_seed(int(config.seed))
    dev = _device(config)
    model = build_model(graph, config, n_vce=n_vce).to(dev)

    x = graph.x.to(dev)
    edge_index = graph.edge_index.to(dev)
    edge_weight = graph.edge_weight.to(dev) if graph.edge_weight is not None else None
    edge_type_ids = (graph.edge_type_ids.to(dev)
                     if getattr(graph, 'edge_type_ids', None) is not None else None)
    t = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in targets.items()}

    opt = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    best = float('inf'); best_state = None; bad = 0
    history = []
    model.train()
    for epoch in range(int(config.epochs)):
        opt.zero_grad()
        out = model(x, edge_index, edge_weight, edge_type_ids)
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
                               compute_losses(model(x, edge_index, edge_weight, edge_type_ids),
                                              t, config).items()}}
    return model, metrics


def _subset_targets(targets: dict, keep_v_pos, keep_p_pos) -> dict:
    """Return a copy of ``targets`` keeping only the given supervised rows.

    Used by cross-validation: held-out VAAMR/PURER nodes are removed from the
    supervised heads (so they never contribute to the loss) while staying in the
    graph (they still receive messages — standard transductive semi-supervised
    eval). Auxiliary self-supervised targets (link prediction, vce)
    are left untouched.
    """
    import torch
    t = dict(targets)
    if 'vaamr_idx' in targets and targets['vaamr_idx'].numel():
        kv = torch.as_tensor(list(keep_v_pos), dtype=torch.long)
        t['vaamr_idx'] = targets['vaamr_idx'][kv]
        t['vaamr_mix'] = targets['vaamr_mix'][kv]
        t['prog_val'] = targets['prog_val'][kv]
        t['contrast_idx'] = targets['contrast_idx'][kv]
        t['contrast_label'] = targets['contrast_label'][kv]
    if 'purer_idx' in targets and targets['purer_idx'].numel():
        kp = torch.as_tensor(list(keep_p_pos), dtype=torch.long)
        t['purer_idx'] = targets['purer_idx'][kp]
        t['purer_label'] = targets['purer_label'][kp]
    return t


def crossval_predictions(graph, targets: dict, config, n_vce: int = 0,
                         return_conf: bool = False, return_logits: bool = False) -> dict:
    """Out-of-sample VAAMR/PURER predictions via k-fold cross-validation.

    For each fold, that fold's VAAMR and PURER target rows are masked from the
    supervised loss; a fresh model is trained on the rest and used to predict the
    held-out nodes. The returned predictions are therefore on segments the model
    did NOT learn from — the basis for the reliability gate that decides when the
    graph can scale without LLMs. Per-fold seeds are deterministic.

    Returns {'vaamr': [(segment_id, pred_int), ...], 'purer': [(segment_id, pred_int), ...]}.
    When ``return_conf=True`` each tuple is extended to (segment_id, pred_int, conf_float)
    — the held-out max-softmax confidence — for abstention-floor calibration.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    node_ids = list(graph.node_ids)
    out: Dict[str, list] = {'vaamr': [], 'purer': []}
    if return_logits:
        out['vaamr_logits'] = []
        out['purer_logits'] = []

    v_idx = targets.get('vaamr_idx')
    p_idx = targets.get('purer_idx')
    n_v = int(v_idx.numel()) if v_idx is not None else 0
    n_p = int(p_idx.numel()) if p_idx is not None else 0
    if n_v == 0 and n_p == 0:
        return out

    folds = int(getattr(config, 'validation_folds', 5) or 1)
    rng = np.random.default_rng(int(config.seed))

    def _make_folds(n):
        if n == 0:
            return []
        perm = rng.permutation(n)
        if folds <= 1:
            h = max(1, int(round(n * float(getattr(config, 'validation_holdout', 0.2)))))
            return [perm[:h]]  # single held-out fold
        return [f for f in np.array_split(perm, min(folds, n)) if len(f)]

    v_folds = _make_folds(n_v)
    p_folds = _make_folds(n_p)
    k = max(len(v_folds), len(p_folds))

    for fi in range(k):
        v_hold = v_folds[fi] if fi < len(v_folds) else np.array([], dtype=int)
        p_hold = p_folds[fi] if fi < len(p_folds) else np.array([], dtype=int)
        keep_v = sorted(set(range(n_v)) - set(int(i) for i in v_hold))
        keep_p = sorted(set(range(n_p)) - set(int(i) for i in p_hold))
        fold_cfg = _replace_seed(config, int(config.seed) + 1 + fi)
        sub = _subset_targets(targets, keep_v, keep_p)
        model, _ = train_model(graph, sub, fold_cfg, n_vce=n_vce)
        dev = _device(fold_cfg)
        x = graph.x.to(dev); ei = graph.edge_index.to(dev)
        ew = graph.edge_weight.to(dev) if graph.edge_weight is not None else None
        eti = (graph.edge_type_ids.to(dev)
               if getattr(graph, 'edge_type_ids', None) is not None else None)
        model.eval()
        with torch.no_grad():
            res = model(x, ei, ew, eti)
        if len(v_hold) and 'soft_vaamr' in res:
            logits_v = res['soft_vaamr'].cpu().numpy()
            probs = F.softmax(res['soft_vaamr'], dim=1).cpu().numpy()
            for pos in v_hold:
                gi = int(v_idx[int(pos)])
                row = probs[gi]
                rec = (node_ids[gi], int(row.argmax()))
                out['vaamr'].append(rec + (float(row.max()),) if return_conf else rec)
                if return_logits:
                    out['vaamr_logits'].append((node_ids[gi], logits_v[gi].copy()))
        if len(p_hold) and 'purer' in res:
            logits_p = res['purer'].cpu().numpy()
            probs = F.softmax(res['purer'], dim=1).cpu().numpy()
            for pos in p_hold:
                gi = int(p_idx[int(pos)])
                row = probs[gi]
                rec = (node_ids[gi], int(row.argmax()))
                out['purer'].append(rec + (float(row.max()),) if return_conf else rec)
                if return_logits:
                    out['purer_logits'].append((node_ids[gi], logits_p[gi].copy()))
    return out


def calibrate_abstain_floors(graph, targets, config, df_all, n_vce: int = 0) -> dict:
    """Derive per-VAAMR-stage abstention floors from held-out CV (A2 calibration).

    For each stage, collect the held-out predictions argmaxing to that stage and choose the
    smallest confidence floor whose KEPT (>= floor) held-out precision meets
    ``config.abstain_target_precision``. Rare/hard stages naturally get higher floors because
    they need more confidence to clear the precision bar. Reference labels are the LLM
    consensus (``final_label``) — the same target the gate uses.

    Returns {'floors': {stage: float}, 'per_stage': [...], 'target_precision': float}.
    Floors are clamped to [0,1]; a stage with no held-out support gets the global
    ``abstain_threshold`` (or 0.0) as a neutral default.
    """
    import numpy as np

    target = float(getattr(config, 'abstain_target_precision', 0.80))
    base = getattr(config, 'abstain_threshold', None)
    default_floor = float(base) if base is not None else 0.0

    cv = crossval_predictions(graph, targets, config, n_vce=n_vce, return_conf=True)
    ref = {str(r.get('segment_id')): r.get('final_label') for _, r in df_all.iterrows()}

    def _ref_int(sid):
        v = ref.get(str(sid))
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

    # Gather (pred, correct, conf) for VAAMR held-out rows that have a reference label.
    rows = []
    for rec in cv.get('vaamr', []):
        if len(rec) < 3:
            continue
        sid, pred, conf = rec[0], int(rec[1]), float(rec[2])
        r = _ref_int(sid)
        if r is None:
            continue
        rows.append((pred, int(pred == r), conf))

    floors: dict = {}
    per_stage = []
    for stage in range(5):
        cand = [(c, ok) for (p, ok, c) in rows if p == stage]
        if not cand:
            floors[stage] = round(default_floor, 4)
            per_stage.append({'stage': stage, 'support': 0, 'floor': floors[stage],
                              'kept_precision': None, 'kept_n': 0})
            continue
        confs = sorted(set(c for c, _ in cand))
        chosen = None
        best = None
        # Smallest floor meeting the precision target on KEPT (conf >= floor) predictions.
        for f in confs:
            kept = [ok for c, ok in cand if c >= f]
            if not kept:
                continue
            prec = sum(kept) / len(kept)
            if best is None or prec > best[1]:
                best = (f, prec, len(kept))
            if prec >= target:
                chosen = (f, prec, len(kept))
                break
        if chosen is None:
            # Target unreachable — use the floor with the best achievable precision.
            chosen = best if best is not None else (default_floor, None, 0)
        f, prec, kept_n = chosen
        floors[stage] = round(float(min(1.0, max(0.0, f))), 4)
        per_stage.append({'stage': stage, 'support': len(cand), 'floor': floors[stage],
                          'kept_precision': (round(prec, 4) if prec is not None else None),
                          'kept_n': kept_n})

    return {'floors': floors, 'per_stage': per_stage, 'target_precision': target}


def _replace_seed(config, seed: int):
    """Return a shallow copy of the config with a different seed (for fold determinism)."""
    import copy
    c = copy.copy(config)
    try:
        c.seed = int(seed)
    except Exception:
        pass
    return c


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


def load_checkpoint(model_dir: str, graph, config, n_vce: int = 0):
    import torch
    from .model import build_model
    model = build_model(graph, config, n_vce=n_vce)
    # strict=False so a checkpoint trained with a head that no longer exists in the
    # current objective set still loads (extra state-dict keys are ignored).
    model.load_state_dict(
        torch.load(os.path.join(model_dir, 'weights.pt'), map_location='cpu'),
        strict=False)
    # Move to the configured device (CUDA when available) so scale-mode inference uses
    # the GPU; _graph_tensors_on_model_device then aligns the graph onto the same device.
    model.to(_device(config))
    model.eval()
    return model
