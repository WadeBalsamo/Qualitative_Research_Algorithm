"""
gnn_layer/graph_builder.py
--------------------------
Build the QRA graph from the assembled master DataFrame.

Implementation note: the heterogeneous design (segments + VAAMR/PURER/VCE/microskill
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


@dataclass
class HeteroGraph:
    """Unified node/edge tensors (homogeneous projection of the heterogeneous design)."""
    x: "object" = None                       # FloatTensor [N, D]
    node_ids: List[str] = field(default_factory=list)
    node_types: List[str] = field(default_factory=list)
    edge_index: "object" = None              # LongTensor [2, E] (undirected: both dirs)
    edge_weight: "object" = None             # FloatTensor [E]
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
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float32)

    g = HeteroGraph(
        x=x, node_ids=node_ids, node_types=node_types,
        edge_index=edge_index, edge_weight=edge_weight, index_of=index_of,
        meta={
            'n_segments': n_seg,
            'embed_dim': int(x.shape[1]),
            'edge_types': edge_types,
            'speaker_of': speaker_of,
            'n_anchors': len(node_ids) - n_seg,
        },
    )
    return g


def attach_new_segments(graph: HeteroGraph, new_embeddings: Dict[str, "object"],
                        config) -> HeteroGraph:
    """Inductively attach unseen segments via kNN edges into the existing nodes.

    New nodes connect only to existing nodes (never to each other), keeping
    predictions order-invariant. Returns a new extended HeteroGraph.
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

    k = min(int(config.knn_k), max(1, len(base_ids)))
    nn = NearestNeighbors(n_neighbors=k, metric='cosine').fit(base_X)
    dist, idx = nn.kneighbors(new_X)
    base_offset = 0  # base nodes keep their indices
    for r, sid in enumerate(new_ids):
        gi = index_of[sid]
        for c in range(idx.shape[1]):
            j = int(idx[r, c]) + base_offset
            sim = max(0.0, float(1.0 - dist[r, c]))
            edges.append([gi, j]); weights.append(sim)
            edges.append([j, gi]); weights.append(sim)

    node_types = list(graph.node_types) + ['segment'] * len(new_ids)
    x = torch.tensor(all_X, dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return HeteroGraph(x=x, node_ids=all_ids, node_types=node_types,
                       edge_index=edge_index, edge_weight=edge_weight,
                       index_of=index_of, meta=dict(graph.meta, n_new=len(new_ids)))


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
