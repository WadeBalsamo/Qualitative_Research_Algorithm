"""
gnn_layer/embeddings.py
-----------------------
Qwen3 embedding of segments and construct-anchor texts for the GNN (SCAFFOLD).

Reuses the existing, validated embedding path rather than loading a second model:
``codebook.embedding_classifier.EmbeddingCodebookClassifier`` already lazy-loads
Qwen3-Embedding-8B at float16 with CPU fallback and exposes asymmetric query/passage
encoding. We call its ``_get_model`` / ``_embed_queries`` / ``_embed`` so segment
texts are encoded as retrieval *queries* and anchor/definition texts as *passages*,
identical to VCE coding.

All heavy imports are deferred to call time so importing this module is cheap.
"""

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # type-only; never imported at runtime
    import numpy as np
    from .config import GnnLayerConfig


def _make_embedder(config: "GnnLayerConfig"):
    """Build an EmbeddingCodebookClassifier configured to match ``config``.

    Returns the classifier instance; its model is lazy-loaded on first encode.
    """
    from codebook.config import EmbeddingClassifierConfig
    from codebook.embedding_classifier import EmbeddingCodebookClassifier
    emb_cfg = EmbeddingClassifierConfig(
        embedding_model=config.embedding_model,
        use_query_prefix=config.use_query_prefix,
        embedding_batch_size=config.embedding_batch_size,
    )
    return EmbeddingCodebookClassifier(emb_cfg)


def embed_segment_texts(texts: List[str], config: "GnnLayerConfig") -> "np.ndarray":
    """Encode segment texts as query embeddings → (N, D) float32 ndarray.

    TODO(scaffold): call ``_make_embedder(config)._embed_queries(texts)``.
    """
    raise NotImplementedError("gnn_layer.embeddings.embed_segment_texts: scaffold")


def embed_anchor_texts(texts: List[str], config: "GnnLayerConfig") -> "np.ndarray":
    """Encode construct-anchor / definition texts as passage embeddings → (N, D).

    TODO(scaffold): call ``_make_embedder(config)._embed(texts)``.
    """
    raise NotImplementedError("gnn_layer.embeddings.embed_anchor_texts: scaffold")


def load_or_build_segment_embeddings(
    df_all,
    config: "GnnLayerConfig",
    cache_path: Optional[str] = None,
) -> Dict[str, "np.ndarray"]:
    """Return {segment_id: embedding}, loading the npz cache if present and fresh.

    Cache lives at ``02_meta/gnn/segment_embeddings.npz`` (keyed by segment_id plus a
    text hash so stale rows are re-encoded). Builds via :func:`embed_segment_texts`
    on a cache miss when ``config.cache_embeddings`` is True.

    TODO(scaffold): implement np.load / np.savez_compressed caching + hash check.
    """
    raise NotImplementedError(
        "gnn_layer.embeddings.load_or_build_segment_embeddings: scaffold"
    )
