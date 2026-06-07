"""
gnn_layer/cue_features.py
-------------------------
Shared cue-block features — used by BOTH the discovery/mechanism work-streams (transition,
communities, motifs, coupling) and the analysis layer (mechanism, language_atlas, mindfulbert).

These are deliberately free of any trained-model dependency (they wrap process.cue_blocks and
mean-pool raw embeddings), so the discovery/mechanism layer never has to import the GNN classifier
subpackage. Moved here from the former gnn_layer/inference.py when the classifier was separated
into gnn_layer/classifier/ (see CLAUDE.md module map).
"""

from typing import Dict, List


def build_cue_blocks_with_segments(df_all) -> List[dict]:
    """Cue blocks (FROM->CUE->TO) retaining the therapist segment ids in each block.

    Uses the shared :func:`process.cue_blocks.cue_blocks_from_records` builder which applies the
    canonical timestamp-overlap window with an index-position fallback when ``end_time_ms == 0``
    (fixing the former empty-block bug for touching or zero timestamps).

    Returns list of dicts: session_id, from_seg_id, to_seg_id, from_stage, to_stage,
    transition_type, therapist_seg_ids.  Empty blocks (therapist_seg_ids=[]) are included so
    callers can compute empty-cue rates.
    """
    required = {'session_id', 'speaker', 'final_label', 'start_time_ms', 'end_time_ms'}
    if not required.issubset(set(df_all.columns)):
        return []

    from process.cue_blocks import cue_blocks_from_records as _cue_blocks_from_records
    specs = _cue_blocks_from_records(
        df_all.to_dict('records'), stage_key='final_label', require_stage=True
    )

    return [
        {
            'session_id': spec.session_id,
            'from_seg_id': str(spec.from_item.get('segment_id', '')),
            'to_seg_id': str(spec.to_item.get('segment_id', '')),
            'from_stage': spec.from_stage,
            'to_stage': spec.to_stage,
            'transition_type': spec.transition_type,
            'therapist_seg_ids': [str(r['segment_id']) for r in spec.therapist_items],
        }
        for spec in specs
    ]


def cue_block_embeddings(blocks: List[dict], seg_embeddings: Dict[str, "object"]):
    """Mean-pool the embeddings of each block's therapist segments.

    Returns (rows, matrix) where rows is the list of mediated (non-empty) blocks that had at least
    one embedded therapist segment, and matrix is np.ndarray [n_rows, H]. The embeddings are
    whatever the caller passes — the discovery/mechanism layer passes RAW Qwen vectors (decoupled
    from the classifier), per the H6/§4.7 finding that similarity neighbourhoods are stage-mixed.
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
        return [], np.zeros((0, 0), dtype=np.float32)
    return rows, np.stack(vecs, axis=0).astype(np.float32)
