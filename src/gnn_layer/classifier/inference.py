"""
gnn_layer/inference.py
----------------------
Per-segment GNN outputs + cue-block embeddings.

Per-segment outputs (VAAMR mixture, progression coordinate, embedding) are written
to 03_analysis_data/gnn/ keyed by segment_id — never folded into master_segments.
"""

from typing import Dict, List, Optional

# Rare VAAMR stages (Metacognition, Reappraisal) — the over-smoothing risk; they get a
# higher abstention floor by default. Mirrors gnn_layer.validation.RARE_STAGES.
_RARE_VAAMR_STAGES = (3, 4)


def resolve_abstain_floors(config) -> Optional[Dict[int, float]]:
    """Per-VAAMR-stage max-prob floors for abstention, or None if abstention is disabled.

    Precedence: explicit ``abstain_per_stage`` (also where calibration writes its result)
    > global ``abstain_threshold`` (with a higher ``abstain_rare_stage_threshold`` for the
    rare stages 3/4). None means the graph never abstains.
    """
    if config is None:
        return None
    per_stage = getattr(config, 'abstain_per_stage', None)
    if per_stage:
        return {int(k): float(v) for k, v in per_stage.items()}
    base = getattr(config, 'abstain_threshold', None)
    if base is None:
        return None
    floors = {s: float(base) for s in range(5)}
    rare = getattr(config, 'abstain_rare_stage_threshold', None)
    if rare is not None:
        for s in _RARE_VAAMR_STAGES:
            floors[s] = float(rare)
    return floors


def resolve_purer_abstain_floor(config) -> Optional[float]:
    """Global max-prob floor for PURER abstention, or None when disabled.

    PURER has no rare-stage concept here, so it uses the global ``abstain_threshold`` only.
    """
    if config is None:
        return None
    base = getattr(config, 'abstain_threshold', None)
    return float(base) if base is not None else None


def _graph_tensors_on_model_device(model, graph):
    """Return (x, edge_index, edge_weight, edge_type_ids) moved onto the model's device.

    train_model leaves the trained model on the training device (e.g. cuda) while
    graph.x / graph.edge_index stay on CPU. Forwarding CPU tensors through a CUDA
    model raises a device-mismatch RuntimeError, so align them here before any
    inference forward pass.
    """
    eti = getattr(graph, 'edge_type_ids', None)
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = None
    if dev is None:
        return graph.x, graph.edge_index, graph.edge_weight, eti
    x = graph.x.to(dev)
    edge_index = graph.edge_index.to(dev)
    edge_weight = graph.edge_weight.to(dev) if graph.edge_weight is not None else None
    eti = eti.to(dev) if eti is not None else None
    return x, edge_index, edge_weight, eti


def infer_segment_positions(model, graph, config) -> dict:
    """Forward pass → per-segment mixture / progression / embedding.

    Returns dict of row-aligned lists/arrays: segment_id, node_type, progression_coord,
    vaamr_mixture (np.ndarray [N,5] or None), gnn_embedding (np.ndarray [N,H]).
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    model.eval()
    _x, _ei, _ew, _eti = _graph_tensors_on_model_device(model, graph)
    with torch.no_grad():
        out = model(_x, _ei, _ew, _eti)
    emb = out['emb'].cpu().numpy()
    res = {
        'segment_id': list(graph.node_ids),
        'node_type': list(graph.node_types),
        'gnn_embedding': emb,
    }
    if 'soft_vaamr' in out:
        from .calibration import apply_temperature
        _T = getattr(config, 'calibration_temperature', None) if config is not None else None
        mix = F.softmax(apply_temperature(out['soft_vaamr'], _T), dim=1).cpu().numpy()
        res['vaamr_mixture'] = mix
        res['progression_coord'] = (mix * np.arange(mix.shape[1])).sum(axis=1)
    elif 'progression' in out:
        res['vaamr_mixture'] = None
        res['progression_coord'] = out['progression'].squeeze(-1).cpu().numpy()
    else:
        res['vaamr_mixture'] = None
        res['progression_coord'] = np.zeros(len(graph.node_ids), dtype=np.float32)
    return res


def infer_head_predictions(model, graph, config=None) -> dict:
    """Per-segment predictions from the trained PURER / VAAMR heads (the independent read).

    The GNN's prediction heads are trained on LLM/ballot labels but their outputs were
    never extracted — discarding the GNN's independent measurement of each segment.
    This runs a forward pass and returns argmax predictions + max prob for the
    single-label heads (soft_vaamr, purer), which downstream triangulation compares
    against the LLM/human labels. Returns row-aligned lists keyed by segment_id.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    model.eval()
    _x, _ei, _ew, _eti = _graph_tensors_on_model_device(model, graph)
    with torch.no_grad():
        out = model(_x, _ei, _ew, _eti)
    res = {
        'segment_id': list(graph.node_ids),
        'node_type': list(graph.node_types),
    }
    from .calibration import apply_temperature
    _T = getattr(config, 'calibration_temperature', None) if config is not None else None
    vaamr_floors = resolve_abstain_floors(config)
    purer_floor = resolve_purer_abstain_floor(config)
    if 'soft_vaamr' in out:
        p = F.softmax(apply_temperature(out['soft_vaamr'], _T), dim=1).cpu().numpy()
        preds = p.argmax(axis=1)
        confs = p.max(axis=1)
        res['gnn_vaamr_pred'] = preds.tolist()
        res['gnn_vaamr_conf'] = confs.round(4).tolist()
        if vaamr_floors is not None:
            res['gnn_vaamr_abstain'] = [
                bool(confs[i] < vaamr_floors.get(int(preds[i]), 0.0))
                for i in range(len(preds))
            ]
    if 'purer' in out:
        p = F.softmax(out['purer'], dim=1).cpu().numpy()
        preds = p.argmax(axis=1)
        confs = p.max(axis=1)
        res['gnn_purer_pred'] = preds.tolist()
        res['gnn_purer_conf'] = confs.round(4).tolist()
        if purer_floor is not None:
            res['gnn_purer_abstain'] = [bool(c < purer_floor) for c in confs]
    return res

# NOTE: build_cue_blocks_with_segments + cue_block_embeddings moved to gnn_layer/cue_features.py
# (shared, model-free) so the discovery/mechanism layer does not import this classifier module.
