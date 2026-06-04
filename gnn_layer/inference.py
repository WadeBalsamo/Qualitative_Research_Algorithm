"""
gnn_layer/inference.py
----------------------
Per-segment GNN outputs + cue-block embeddings.

Per-segment outputs (VAAMR mixture, progression coordinate, embedding) are written
to 03_analysis_data/gnn/ keyed by segment_id — never folded into master_segments.
"""

from typing import Dict, List, Optional


def infer_segment_positions(model, graph, config) -> dict:
    """Forward pass → per-segment mixture / progression / embedding.

    Returns dict of row-aligned lists/arrays: segment_id, node_type, progression_coord,
    vaamr_mixture (np.ndarray [N,5] or None), gnn_embedding (np.ndarray [N,H]).
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        out = model(graph.x, graph.edge_index,
                    graph.edge_weight if graph.edge_weight is not None else None)
    emb = out['emb'].cpu().numpy()
    res = {
        'segment_id': list(graph.node_ids),
        'node_type': list(graph.node_types),
        'gnn_embedding': emb,
    }
    if 'soft_vaamr' in out:
        mix = F.softmax(out['soft_vaamr'], dim=1).cpu().numpy()
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
    with torch.no_grad():
        out = model(graph.x, graph.edge_index,
                    graph.edge_weight if graph.edge_weight is not None else None)
    res = {
        'segment_id': list(graph.node_ids),
        'node_type': list(graph.node_types),
    }
    if 'soft_vaamr' in out:
        p = F.softmax(out['soft_vaamr'], dim=1).cpu().numpy()
        res['gnn_vaamr_pred'] = p.argmax(axis=1).tolist()
        res['gnn_vaamr_conf'] = p.max(axis=1).round(4).tolist()
    if 'purer' in out:
        p = F.softmax(out['purer'], dim=1).cpu().numpy()
        res['gnn_purer_pred'] = p.argmax(axis=1).tolist()
        res['gnn_purer_conf'] = p.max(axis=1).round(4).tolist()
    return res


def build_cue_blocks_with_segments(df_all) -> List[dict]:
    """Cue blocks (FROM->CUE->TO) retaining the therapist segment ids in each block.

    Mirrors the timestamp-window logic of
    analysis.purer_analysis.compute_cue_block_purer_profiles but keeps the therapist
    segment ids (which CueBlock does not expose) so we can pool their GNN embeddings.
    Returns list of dicts: session_id, from_seg_id, to_seg_id, from_stage, to_stage,
    transition_type, therapist_seg_ids.
    """
    import pandas as pd
    required = {'session_id', 'speaker', 'final_label', 'start_time_ms', 'end_time_ms'}
    if not required.issubset(set(df_all.columns)):
        return []
    blocks: List[dict] = []
    for session_id, sdf in df_all.groupby('session_id'):
        part = sdf[(sdf['speaker'] == 'participant') & sdf['final_label'].notna()] \
            .sort_values('start_time_ms').reset_index(drop=True)
        if len(part) < 2:
            continue
        ther = sdf[sdf['speaker'] == 'therapist']
        for i in range(len(part) - 1):
            fr, to = part.iloc[i], part.iloc[i + 1]
            fe, ts = int(fr.get('end_time_ms', 0)), int(to.get('start_time_ms', 0))
            fs, tstg = int(fr['final_label']), int(to['final_label'])
            if ts > fe:
                between = ther[(ther['start_time_ms'] < ts) & (ther['end_time_ms'] > fe)]
            else:
                between = ther.iloc[:0]
            tids = [str(s) for s in between.get('segment_id', pd.Series(dtype=str)).tolist()]
            blocks.append({
                'session_id': str(session_id),
                'from_seg_id': str(fr.get('segment_id', '')),
                'to_seg_id': str(to.get('segment_id', '')),
                'from_stage': fs, 'to_stage': tstg,
                'transition_type': ('forward' if fs < tstg else 'backward' if fs > tstg else 'lateral'),
                'therapist_seg_ids': tids,
            })
    return blocks


def cue_block_embeddings(blocks: List[dict], seg_embeddings: Dict[str, "object"]):
    """Mean-pool GNN embeddings of each block's therapist segments.

    Returns (rows, matrix) where rows is the list of mediated (non-empty) blocks that
    had at least one embedded therapist segment, and matrix is np.ndarray [n_rows, H].
    """
    import numpy as np
    rows, vecs = [], []
    for b in blocks:
        embs = [seg_embeddings[s] for s in b['therapist_seg_ids'] if s in seg_embeddings]
        if not embs:
            continue
        rows.append(b)
        vecs.append(np.mean(np.stack(embs, axis=0), axis=0))
    if not vecs:
        import numpy as np
        return [], np.zeros((0, 0), dtype=np.float32)
    return rows, np.stack(vecs, axis=0).astype(np.float32)
