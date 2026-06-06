"""
gnn_layer/embeddings_remote.py
------------------------------
OpenAI-compatible ``/v1/embeddings`` client for the GNN layer.

This is the ``embedding_backend='openai'`` path: instead of loading a local
SentenceTransformer (the default ``'local'`` path in ``embeddings.py``), segment
and anchor texts are POSTed to an OpenAI-compatible embeddings endpoint — e.g.
LM Studio serving ``text-embedding-qwen3-embedding-8b`` — and the returned vectors
are used as GNN node features. No second 8 GB model is loaded in-process.

Query / passage parity with the SentenceTransformer path
--------------------------------------------------------
The local path (``codebook.embedding_classifier.EmbeddingCodebookClassifier``)
encodes SEGMENT texts as retrieval *queries* (``_embed_queries`` →
``model.encode(prompt_name='query')``) and ANCHOR / definition texts as plain
*passages* (``_embed``). SentenceTransformer applies a query prompt by simple
concatenation — ``prompt + sentence`` with NO separator (SentenceTransformer.py:
``sentences = [prompt + sentence for sentence in sentences]``) — using the prompt
declared in the model's ``config_sentence_transformers.json``. For
``Qwen/Qwen3-Embedding-8B`` that ``"query"`` prompt is::

    Instruct: Given a web search query, retrieve relevant passages that answer the query\\nQuery:

(verbatim, no trailing space — see ``QWEN3_QUERY_PROMPT``).

We MIRROR that here: when ``is_query=True`` and ``config.use_query_prefix`` we
prepend the same query prompt to every text before sending; when ``is_query=False``
we send the raw text (the Qwen3 ``"document"`` prompt is the empty string, so
passages get no prefix on the local path either). The prompt is resolved from the
model's own ``config_sentence_transformers.json`` when ``config.embedding_model``
is a HuggingFace repo id present in the offline cache (genuine reuse of the model's
declared convention), and otherwise falls back to ``QWEN3_QUERY_PROMPT`` — which is
the case for the LM Studio *served* id (``text-embedding-qwen3-embedding-8b`` is not
an HF repo id).

Normalization
-------------
Returns RAW embeddings exactly as the endpoint sends them (no client-side L2
normalization is applied here). NOTE the empirical asymmetry vs the local path:
the LM Studio Qwen3-Embedding-8B ``/v1/embeddings`` endpoint returns ALREADY
L2-NORMALIZED vectors (observed ‖v‖₂ = 1.0000 for a short string), whereas the
local SentenceTransformer path returns UN-normalized vectors by default. This is
harmless for the cosine-similarity edges/anchors used by the GNN (``sklearn``
``cosine_similarity`` normalizes internally), but callers that compare raw feature
magnitudes across the two backends should be aware of it.

HTTP
----
Uses ``requests`` (the same HTTP library as ``classification_tools/llm_client.py``),
lazily imported at call time so importing this module stays cheap. Batches by
``config.embedding_batch_size`` (falls back to ``DEFAULT_BATCH_SIZE`` when unset),
with a per-request timeout and a small exponential backoff on transient failures
(connection errors, HTTP 429, HTTP 5xx). Other 4xx responses raise immediately.
"""

import os
from typing import Dict, List, Optional


# Verbatim ``"query"`` prompt from Qwen3-Embedding-8B's config_sentence_transformers.json
# (no trailing space). SentenceTransformer prepends this by ``prompt + sentence``, so the
# remote path must do the same to produce comparable query embeddings.
QWEN3_QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery:"
)

# Embedding batch size used when ``config.embedding_batch_size`` is unset / non-positive.
# Embeddings are cheap relative to chat completions, so a larger default is fine.
DEFAULT_BATCH_SIZE = 16


def _resolve_query_prompt(model_id: Optional[str]) -> str:
    """Return the query prompt to prepend, mirroring the SentenceTransformer path.

    When ``model_id`` is a HuggingFace repo id (contains ``/``) already present in the
    OFFLINE hub cache, read its ``config_sentence_transformers.json`` and reuse the
    declared ``prompts['query']`` — byte-identical to what SentenceTransformer would
    prepend. Otherwise (e.g. an LM Studio served id with no slash, or an un-cached
    model) fall back to ``QWEN3_QUERY_PROMPT``. Best-effort and network-free: any
    failure falls back to the constant.
    """
    if model_id and '/' in model_id:
        try:
            from huggingface_hub import try_to_load_from_cache
            path = try_to_load_from_cache(
                repo_id=model_id, filename='config_sentence_transformers.json')
            if isinstance(path, str) and os.path.isfile(path):
                import json
                with open(path, 'r', encoding='utf-8') as f:
                    prompts = (json.load(f).get('prompts') or {})
                query_prompt = prompts.get('query')
                if query_prompt:
                    return str(query_prompt)
        except Exception:
            pass
    return QWEN3_QUERY_PROMPT


def _post_embeddings(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, object],
    *,
    timeout: float,
    max_retries: int,
    retry_base_delay: float,
) -> Dict[str, object]:
    """POST one batch and return the parsed JSON body, with retry/backoff.

    Retries connection errors, HTTP 429, and HTTP 5xx up to ``max_retries`` attempts,
    sleeping ``retry_base_delay * 2**attempt`` between tries (never after the last).
    Other 4xx responses are non-retryable and raise immediately. Raises RuntimeError
    when every attempt fails.
    """
    import time

    import requests

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:  # connection reset / timeout / DNS
            last_error = exc
        else:
            code = resp.status_code
            if code < 400:
                try:
                    return resp.json()
                except ValueError as exc:  # malformed body — transient, retry
                    last_error = exc
            elif code == 429 or code >= 500:
                last_error = RuntimeError(
                    f"transient HTTP {code} from embeddings endpoint: "
                    f"{(getattr(resp, 'text', '') or '')[:200]}")
            else:
                raise RuntimeError(
                    f"embeddings endpoint returned HTTP {code}: "
                    f"{(getattr(resp, 'text', '') or '')[:200]}")
        if attempt < max_retries - 1:
            time.sleep(retry_base_delay * (2 ** attempt))

    raise RuntimeError(
        f"embeddings request to {url} failed after {max_retries} attempts: {last_error}")


def embed_texts_remote(
    texts: List[str],
    config,
    is_query: bool,
    *,
    timeout: float = 60.0,
    max_retries: int = 3,
    retry_base_delay: float = 1.0,
) -> "object":
    """Embed ``texts`` via an OpenAI-compatible ``/v1/embeddings`` endpoint.

    Parameters
    ----------
    texts : list[str]
        Texts to embed. ``None`` entries are coerced to ``''``.
    config :
        A ``GnnLayerConfig`` (or any object exposing ``embedding_base_url``,
        ``embedding_model``, ``embedding_api_key``, ``embedding_batch_size``,
        ``use_query_prefix``).
    is_query : bool
        ``True`` for SEGMENT texts (encoded as retrieval queries: the query prompt
        is prepended when ``config.use_query_prefix``). ``False`` for ANCHOR /
        definition texts (encoded as plain passages — never prefixed).

    Returns
    -------
    numpy.ndarray
        ``(N, D)`` float32 array of RAW (un-normalized) embeddings, one row per input
        text in input order. For Qwen3-Embedding-8B, ``D == 4096``.
    """
    import numpy as np

    base_url = getattr(config, 'embedding_base_url', None)
    if not base_url:
        raise ValueError(
            "embedding_backend='openai' requires config.embedding_base_url "
            "(e.g. 'http://10.0.0.58:1234/v1')")

    texts = ['' if t is None else str(t) for t in texts]
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    url = base_url.rstrip('/') + '/embeddings'
    headers: Dict[str, str] = {'Content-Type': 'application/json'}
    api_key = getattr(config, 'embedding_api_key', None)
    if api_key:  # LM Studio needs none; only send when explicitly configured
        headers['Authorization'] = f'Bearer {api_key}'

    use_prefix = bool(is_query and getattr(config, 'use_query_prefix', False))
    prefix = _resolve_query_prompt(getattr(config, 'embedding_model', None)) if use_prefix else ''
    model_id = getattr(config, 'embedding_model', None)

    batch_size = int(getattr(config, 'embedding_batch_size', 0) or 0)
    if batch_size <= 0:
        batch_size = DEFAULT_BATCH_SIZE

    out: List["np.ndarray"] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        payload = {'model': model_id, 'input': [prefix + t for t in batch]}
        data = _post_embeddings(
            url, headers, payload,
            timeout=timeout, max_retries=max_retries, retry_base_delay=retry_base_delay)

        rows = data.get('data') if isinstance(data, dict) else None
        if not isinstance(rows, list) or len(rows) != len(batch):
            got = len(rows) if isinstance(rows, list) else rows
            raise RuntimeError(
                f"embeddings response malformed: expected {len(batch)} vectors, got {got}")
        # OpenAI guarantees input order, but sort by 'index' defensively.
        rows = sorted(rows, key=lambda r: r.get('index', 0))
        for r in rows:
            out.append(np.asarray(r['embedding'], dtype=np.float32))

    return np.vstack(out).astype(np.float32, copy=False)
