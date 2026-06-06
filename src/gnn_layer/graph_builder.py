"""
gnn_layer/graph_builder.py
--------------------------
Build the QRA graph from the assembled master DataFrame.

Implementation note: the heterogeneous design (segments + VAAMR/PURER/VCE
anchors) is realized as a *homogeneous projection* — every node lives in the same
Qwen3 embedding space, so one shared SAGE aggregation over the union of typed edges
is faithful and avoids a torch-geometric dependency. Edge *types* are retained in
``meta`` (and anchors are optional) so Capability-D ablations can rebuild the graph
with a family removed.

Always built: segment nodes + temporal-chain edges + kNN-similarity edges.
Optional (when ``anchor_features`` / ``anchor_edges`` are supplied by the runner,
which needs the embedding model): construct-anchor nodes + anchor/label +
cross-framework edges.

torch / numpy / sklearn imported lazily.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Canonical edge-type vocabulary. The integer id of an edge family is its position
# here; SAGEConv's learnable per-type gate is indexed by these ids, so the order is a
# stable contract across train / save / load / scale-mode (do NOT reorder). Families
# absent from a given graph simply leave their gate unused.
EDGE_TYPE_VOCAB: Tuple[str, ...] = ('temporal', 'knn', 'anchor', 'precipitates')
EDGE_TYPE_TO_ID: Dict[str, int] = {t: i for i, t in enumerate(EDGE_TYPE_VOCAB)}


@dataclass
class HeteroGraph:
    """Unified node/edge tensors (homogeneous projection of the heterogeneous design)."""
    x: "object" = None                       # FloatTensor [N, D]
    node_ids: List[str] = field(default_factory=list)
    node_types: List[str] = field(default_factory=list)
    edge_index: "object" = None              # LongTensor [2, E] (undirected: both dirs)
    edge_weight: "object" = None             # FloatTensor [E]
    edge_type_ids: "object" = None           # LongTensor [E] (id into EDGE_TYPE_VOCAB)
    index_of: Dict[str, int] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def type_indices(self, node_type: str) -> List[int]:
        return [i for i, t in enumerate(self.node_types) if t == node_type]


def _stack(vectors):
    import numpy as np
    return np.stack(vectors, axis=0).astype(np.float32)


def build_graph(
    df_all,
    segment_embeddings: Dict[str, "object"],
    config,
    framework: Optional[dict] = None,
    anchor_features: Optional[Dict[str, "object"]] = None,
    anchor_edges: Optional[List[Tuple[str, str, float]]] = None,
) -> HeteroGraph:
    """Assemble the graph. See module docstring for edge families."""
    import numpy as np
    import torch
    from sklearn.neighbors import NearestNeighbors

    # ---- segment nodes (only rows we have embeddings for) ----
    rows = []
    for _, r in df_all.iterrows():
        sid = str(r.get('segment_id'))
        if sid in segment_embeddings:
            rows.append(r)
    if not rows:
        raise ValueError("build_graph: no segments with embeddings")

    node_ids: List[str] = []
    node_types: List[str] = []
    feats: List["np.ndarray"] = []
    speaker_of: Dict[str, str] = {}
    for r in rows:
        sid = str(r.get('segment_id'))
        node_ids.append(sid)
        spk = str(r.get('speaker', '') or '')
        node_types.append('participant_segment' if spk == 'participant'
                          else 'therapist_segment' if spk == 'therapist'
                          else 'segment')
        feats.append(np.asarray(segment_embeddings[sid], dtype=np.float32))
        speaker_of[sid] = spk

    index_of = {sid: i for i, sid in enumerate(node_ids)}
    n_seg = len(node_ids)

    edges: List[Tuple[int, int]] = []
    weights: List[float] = []
    edge_types: List[str] = []

    def _add(a: int, b: int, w: float, t: str):
        edges.append((a, b)); weights.append(w); edge_types.append(t)
        edges.append((b, a)); weights.append(w); edge_types.append(t)

    # ---- temporal-chain edges (consecutive segments within a session) ----
    if 'session_id' in df_all.columns:
        seg_rows = [(sid, df_all_row) for sid, df_all_row in
                    [(str(r.get('segment_id')), r) for r in rows]]
        by_session: Dict[str, List[Tuple[int, int]]] = {}
        for sid, r in seg_rows:
            sess = str(r.get('session_id'))
            start = int(r.get('start_time_ms', 0) or 0)
            by_session.setdefault(sess, []).append((start, index_of[sid]))
        for sess, lst in by_session.items():
            lst.sort(key=lambda x: x[0])
            for (_, a), (_, b) in zip(lst, lst[1:]):
                _add(a, b, 1.0, 'temporal')

    # ---- kNN-similarity edges (cosine) ----
    X = _stack(feats)
    k = min(int(config.knn_k), max(1, n_seg - 1))
    if n_seg > 1 and k >= 1:
        nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(X)
        dist, idx = nn.kneighbors(X)
        for i in range(n_seg):
            for j_pos in range(1, idx.shape[1]):   # skip self at position 0
                j = int(idx[i, j_pos])
                sim = float(1.0 - dist[i, j_pos])
                _add(i, j, max(0.0, sim), 'knn')

    # ---- optional typed therapist->participant "precipitates" edges (Track A1) ----
    # For each cue block, connect each therapist cue segment to the following participant
    # segment so the participant representation can attend to the preceding cue. Gated by
    # config so the default graph is unchanged until the family earns its place via ablation.
    if getattr(config, 'precipitates_edges', False):
        from process.cue_blocks import cue_blocks_from_records
        records = [dict(r) for r in rows]
        specs = cue_blocks_from_records(records, require_stage=False)
        n_prec = 0
        for spec in specs:
            to_sid = str(spec.to_item.get('segment_id')) if isinstance(spec.to_item, dict) \
                else str(getattr(spec.to_item, 'segment_id', ''))
            if to_sid not in index_of:
                continue
            for th in spec.therapist_items:
                th_sid = str(th.get('segment_id')) if isinstance(th, dict) \
                    else str(getattr(th, 'segment_id', ''))
                if th_sid in index_of:
                    _add(index_of[th_sid], index_of[to_sid], 1.0, 'precipitates')
                    n_prec += 1

    # ---- optional anchor nodes + anchor/cross-framework edges ----
    if anchor_features:
        for aid, vec in anchor_features.items():
            if aid in index_of:
                continue
            index_of[aid] = len(node_ids)
            node_ids.append(aid)
            node_types.append('anchor')
            feats.append(np.asarray(vec, dtype=np.float32))
        X = _stack(feats)  # rebuild with anchors
        if anchor_edges:
            for src, dst, w in anchor_edges:
                if src in index_of and dst in index_of:
                    _add(index_of[src], index_of[dst], float(w), 'anchor')

    x = torch.tensor(_stack(feats), dtype=torch.float32)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weights, dtype=torch.float32)
        edge_type_ids = torch.tensor(
            [EDGE_TYPE_TO_ID[t] for t in edge_types], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float32)
        edge_type_ids = torch.zeros((0,), dtype=torch.long)

    g = HeteroGraph(
        x=x, node_ids=node_ids, node_types=node_types,
        edge_index=edge_index, edge_weight=edge_weight, edge_type_ids=edge_type_ids,
        index_of=index_of,
        meta={
            'n_segments': n_seg,
            'embed_dim': int(x.shape[1]),
            'edge_types': edge_types,
            'edge_type_vocab': list(EDGE_TYPE_VOCAB),
            'n_precipitates': sum(1 for t in edge_types if t == 'precipitates'),
            'speaker_of': speaker_of,
            'n_anchors': len(node_ids) - n_seg,
        },
    )
    return g


def attach_new_segments(graph: HeteroGraph, new_embeddings: Dict[str, "object"],
                        config,
                        node_type_of: Optional[Dict[str, str]] = None) -> HeteroGraph:
    """Inductively attach unseen segments via kNN edges into the existing nodes.

    New nodes connect only to existing nodes (never to each other), keeping
    predictions order-invariant. Returns a new extended HeteroGraph.

    ``node_type_of`` maps each new segment_id to its node type
    ('participant_segment' / 'therapist_segment'); without it attached nodes
    default to the generic 'segment' type, which the head-prediction router skips —
    so scale-mode callers must supply it (built from the df speaker column).
    """
    import numpy as np
    import torch
    from sklearn.neighbors import NearestNeighbors

    base_ids = list(graph.node_ids)
    base_X = graph.x.detach().cpu().numpy()
    new_ids = [sid for sid in new_embeddings if sid not in graph.index_of]
    if not new_ids:
        return graph

    new_X = _stack([np.asarray(new_embeddings[s], dtype=np.float32) for s in new_ids])
    all_ids = base_ids + new_ids
    all_X = np.concatenate([base_X, new_X], axis=0)
    index_of = {sid: i for i, sid in enumerate(all_ids)}

    edges = graph.edge_index.t().tolist() if graph.edge_index.numel() else []
    weights = graph.edge_weight.tolist() if graph.edge_weight.numel() else []
    type_ids = graph.edge_type_ids.tolist() \
        if getattr(graph, 'edge_type_ids', None) is not None and graph.edge_type_ids.numel() \
        else []
    _knn_id = EDGE_TYPE_TO_ID['knn']

    k = min(int(config.knn_k), max(1, len(base_ids)))
    nn = NearestNeighbors(n_neighbors=k, metric='cosine').fit(base_X)
    dist, idx = nn.kneighbors(new_X)
    base_offset = 0  # base nodes keep their indices
    for r, sid in enumerate(new_ids):
        gi = index_of[sid]
        for c in range(idx.shape[1]):
            j = int(idx[r, c]) + base_offset
            sim = max(0.0, float(1.0 - dist[r, c]))
            edges.append([gi, j]); weights.append(sim); type_ids.append(_knn_id)
            edges.append([j, gi]); weights.append(sim); type_ids.append(_knn_id)

    ntype = node_type_of or {}
    node_types = list(graph.node_types) + [ntype.get(sid, 'segment') for sid in new_ids]
    speaker_of = dict(graph.meta.get('speaker_of', {}))
    for sid in new_ids:
        nt = ntype.get(sid)
        if nt == 'participant_segment':
            speaker_of[sid] = 'participant'
        elif nt == 'therapist_segment':
            speaker_of[sid] = 'therapist'
    x = torch.tensor(all_X, dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    edge_type_ids = torch.tensor(type_ids, dtype=torch.long) if type_ids \
        else torch.zeros((0,), dtype=torch.long)
    return HeteroGraph(x=x, node_ids=all_ids, node_types=node_types,
                       edge_index=edge_index, edge_weight=edge_weight,
                       edge_type_ids=edge_type_ids, index_of=index_of,
                       meta=dict(graph.meta, n_new=len(new_ids), speaker_of=speaker_of))


def save_graph(graph: HeteroGraph, model_dir: str) -> str:
    """Persist a trained graph so scale-mode can attach to it inductively.

    Saves tensors + node bookkeeping to ``<model_dir>/graph.pt`` (a plain dict via
    torch.save). Paired with :func:`load_graph`.
    """
    import os
    import torch
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, 'graph.pt')
    torch.save({
        'x': graph.x,
        'node_ids': list(graph.node_ids),
        'node_types': list(graph.node_types),
        'edge_index': graph.edge_index,
        'edge_weight': graph.edge_weight,
        'edge_type_ids': getattr(graph, 'edge_type_ids', None),
        'meta': graph.meta,
    }, path)
    return path


def load_graph(model_dir: str) -> Optional[HeteroGraph]:
    """Reconstruct a HeteroGraph saved by :func:`save_graph`; None if absent."""
    import os
    import torch
    path = os.path.join(model_dir, 'graph.pt')
    if not os.path.isfile(path):
        return None
    d = torch.load(path, map_location='cpu')
    node_ids = list(d['node_ids'])
    return HeteroGraph(
        x=d['x'], node_ids=node_ids, node_types=list(d['node_types']),
        edge_index=d['edge_index'], edge_weight=d['edge_weight'],
        edge_type_ids=d.get('edge_type_ids'),
        index_of={sid: i for i, sid in enumerate(node_ids)},
        meta=d.get('meta', {}),
    )


def compute_cross_framework_lift(df_all, min_lift: float = 1.5) -> Dict[Tuple[str, str], float]:
    """Empirical (vaamr_stage, vce_code) lift over df_all for cross-framework edges.

    lift = P(code | stage) / P(code). Returns pairs whose lift >= min_lift.
    """
    import numpy as np
    out: Dict[Tuple[str, str], float] = {}
    if 'final_label' not in df_all.columns or 'codebook_labels_ensemble' not in df_all.columns:
        return out
    part = df_all[df_all.get('speaker', 'participant') == 'participant'] \
        if 'speaker' in df_all.columns else df_all
    n = len(part)
    if n == 0:
        return out

    def _codes(v):
        if isinstance(v, list):
            return v
        return []

    code_counts: Dict[str, int] = {}
    for v in part['codebook_labels_ensemble']:
        for c in _codes(v):
            code_counts[c] = code_counts.get(c, 0) + 1
    for stage, grp in part.groupby('final_label'):
        m = len(grp)
        if m == 0:
            continue
        for c in code_counts:
            in_stage = sum(1 for v in grp['codebook_labels_ensemble'] if c in _codes(v))
            p_code_given_stage = in_stage / m
            p_code = code_counts[c] / n
            if p_code > 0:
                lift = p_code_given_stage / p_code
                if lift >= min_lift:
                    out[(f'vaamr_{int(stage)}', c)] = round(lift, 3)
    return out
