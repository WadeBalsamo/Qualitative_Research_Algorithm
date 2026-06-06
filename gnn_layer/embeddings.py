"""
gnn_layer/embeddings.py
-----------------------
Qwen3 embedding of segments and construct-anchor texts for the GNN.

Reuses the existing, validated embedding path (codebook.embedding_classifier.
EmbeddingCodebookClassifier) rather than loading a second model: segment texts are
encoded as retrieval *queries* and anchor/definition texts as *passages*, exactly as
in VCE coding. The float16/CPU-fallback loader and batch size all come from there.

Heavy imports are deferred to call time so importing this module is cheap. The
``load_or_build_segment_embeddings`` cache lets the rest of the layer (and the test
suite) run with precomputed vectors and no model download.
"""

import hashlib
import os
from typing import Dict, List, Optional


# Cache the heavy 8B embedder for the duration of a run so segment AND anchor encoding
# reuse ONE loaded model (avoids loading the ~8 GB Qwen3 twice on the GPU). Keyed by the
# settings that determine the model so a config change rebuilds it. release_embedder()
# frees it (and CUDA memory) before GNN training allocates.
_EMBEDDER = None
_EMBEDDER_KEY = None


def _make_embedder(config):
    """Get (or build, then cache) an EmbeddingCodebookClassifier matching ``config``.

    Honors ``config.device`` (None → auto-CUDA when available) so the documented device
    knob governs the dominant compute, not just SentenceTransformer's auto-detection.
    """
    global _EMBEDDER, _EMBEDDER_KEY
    from codebook.config import EmbeddingClassifierConfig
    from codebook.embedding_classifier import EmbeddingCodebookClassifier
    key = (config.embedding_model, bool(config.use_query_prefix),
           int(config.embedding_batch_size), getattr(config, 'device', None))
    if _EMBEDDER is not None and _EMBEDDER_KEY == key:
        return _EMBEDDER
    emb_cfg = EmbeddingClassifierConfig(
        embedding_model=config.embedding_model,
        use_query_prefix=config.use_query_prefix,
        embedding_batch_size=config.embedding_batch_size,
        device=getattr(config, 'device', None),
    )
    _EMBEDDER = EmbeddingCodebookClassifier(emb_cfg)
    _EMBEDDER_KEY = key
    return _EMBEDDER


def release_embedder():
    """Drop the cached embedder and free CUDA memory (call before GNN training).

    Mirrors the segmenter's release_gpu_memory so the 8 GB embedder does not coexist with
    GNN training tensors on the GPU. Safe to call when no embedder is loaded.
    """
    global _EMBEDDER, _EMBEDDER_KEY
    _EMBEDDER = None
    _EMBEDDER_KEY = None
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def embed_segment_texts(texts: List[str], config):
    """Encode segment texts as query embeddings → (N, D) float32 ndarray."""
    return _make_embedder(config)._embed_queries(list(texts))


def embed_anchor_texts(texts: List[str], config):
    """Encode construct-anchor / definition texts as passage embeddings → (N, D)."""
    return _make_embedder(config)._embed(list(texts))


def _text_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]


def load_or_build_segment_embeddings(
    df_all,
    config,
    cache_path: Optional[str] = None,
) -> Dict[str, "object"]:
    """Return {segment_id: np.ndarray} for every row of ``df_all`` that has text.

    Uses an npz cache keyed by segment_id + text hash; only missing/stale rows are
    re-encoded. When ``config.cache_embeddings`` is False the cache is ignored.
    Raises RuntimeError only if encoding is required but the embedding model cannot
    be loaded — callers (runner) treat that as a skip.
    """
    import numpy as np

    seg_ids = [str(s) for s in df_all['segment_id'].tolist()]
    texts = ['' if t is None else str(t) for t in df_all['text'].tolist()]
    want_hash = {sid: _text_hash(tx) for sid, tx in zip(seg_ids, texts)}

    cached: Dict[str, "np.ndarray"] = {}
    cached_hash: Dict[str, str] = {}
    if config.cache_embeddings and cache_path and os.path.isfile(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            ids = list(data['segment_ids'])
            hashes = list(data['hashes'])
            mat = data['embeddings']
            for i, sid in enumerate(ids):
                cached[str(sid)] = mat[i]
                cached_hash[str(sid)] = str(hashes[i])
        except Exception:
            cached, cached_hash = {}, {}

    # Which rows need (re)encoding?
    todo = [i for i, sid in enumerate(seg_ids)
            if cached_hash.get(sid) != want_hash[sid]]

    if todo:
        new_vecs = embed_segment_texts([texts[i] for i in todo], config)
        new_vecs = np.asarray(new_vecs, dtype=np.float32)
        for k, i in enumerate(todo):
            cached[seg_ids[i]] = new_vecs[k]
            cached_hash[seg_ids[i]] = want_hash[seg_ids[i]]

    result = {sid: cached[sid] for sid in seg_ids if sid in cached}

    if todo and config.cache_embeddings and cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            all_ids = list(result.keys())
            mat = np.stack([result[i] for i in all_ids], axis=0)
            hashes = [cached_hash[i] for i in all_ids]
            np.savez_compressed(cache_path, segment_ids=np.array(all_ids),
                                hashes=np.array(hashes), embeddings=mat)
        except Exception:
            pass

    return result
