"""
embedding_classifier.py
-----------------------
Multi-label codebook classification using sentence-transformer embeddings.

Default model: Qwen/Qwen3-Embedding-8B
  4096-dim embeddings purpose-built for complex semantic retrieval.
  Downloads ~16 GB from HuggingFace Hub on first use (then cached).
  Loaded at float16 to halve VRAM/RAM footprint (~8 GB GPU memory).

Asymmetric encoding (enabled by default via use_query_prefix=True):
  Segment texts   → encoded with the model's built-in 'query' instruction prefix,
                    treating each excerpt as a retrieval query over construct space.
  Codebook texts  → encoded as plain passages (definition, criteria, exemplars).
  This is the recommended encoding strategy for Qwen3-Embedding and similar
  instruction-following embedding models.

Scoring formula per (segment i, code j):
    score[i, j] = cosine_sim(query_i, definition_j)
                + criteria_weight  * cosine_sim(query_i, criteria_j)
                + exemplar_weight  * cosine_sim(query_i, exemplars_j)

Two-pass exemplar accumulation:
  Pass 1 — classify with any pre-imported exemplars.
  Exemplar accumulation — collect high-confidence pass-1 segments per code.
  Pass 2 — re-classify with merged exemplars for improved recall.

The LLM codebook classifier (classification_tools/llm_classifier.py) uses the
configured LM Studio backend for zero-shot prompting. This independently codes the segments, and an ensemble step
(codebook/ensemble.py) reconciles both sets of results to ensure agreeability.

Note that this process seems to code much less than the 3 model classification loop but is more computationally efficient, especially in large codebooks, and enables multiple codes to be applied. Some tuning could be done here to further understand associations of codes and themes.


"""

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional, Tuple

from classification_tools.data_structures import Segment
from .codebook_schema import Codebook, CodeAssignment
from .config import EmbeddingClassifierConfig


# ---------------------------------------------------------------------------
# Download / readiness utility (can be called before pipeline starts)
# ---------------------------------------------------------------------------

def ensure_embedding_model_ready(model_id: str) -> bool:
    """
    Check whether *model_id* is cached locally; if not, download it.

    Returns True when the model is ready, False if the download failed.
    Prints clear progress messages so the user knows what is happening.
    """
    try:
        from huggingface_hub import try_to_load_from_cache, snapshot_download
        from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError
    except ImportError:
        # huggingface_hub not available — sentence_transformers will handle it
        return True

    # Quick cache check: look for config.json in the local HF cache
    try:
        cached = try_to_load_from_cache(repo_id=model_id, filename='config.json')
        is_cached = cached is not None and not isinstance(cached, type(None))
    except Exception:
        is_cached = False

    if is_cached:
        return True

    # Model not cached — need to download
    import os
    size_hint = _model_size_hint(model_id)
    print(f"\n  Embedding model not cached: {model_id}")
    if size_hint:
        print(f"  Download size: ~{size_hint}")
    print(f"  Downloading from HuggingFace Hub...")
    print(f"  (Files will be cached at ~/.cache/huggingface/ for future runs)")

    try:
        snapshot_download(repo_id=model_id, ignore_patterns=['*.pt', '*.bin'])
        print(f"  Download complete: {model_id}")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        print(f"  You can pre-download manually:")
        print(f"    python -c \"from sentence_transformers import SentenceTransformer; "
              f"SentenceTransformer('{model_id}', trust_remote_code=True)\"")
        return False


def _model_size_hint(model_id: str) -> str:
    """Return a human-readable size hint for known models."""
    hints = {
        'Qwen/Qwen3-Embedding-8B': '~16 GB (float16 weights)',
        'Qwen/Qwen3-Embedding-4B': '~8 GB',
        'Qwen/Qwen3-Embedding-0.6B': '~1.2 GB',
        'BAAI/bge-large-en-v1.5': '~1.3 GB',
        'BAAI/bge-m3': '~2.3 GB',
        'all-MiniLM-L6-v2': '~90 MB',
        'all-mpnet-base-v2': '~420 MB',
    }
    for key, hint in hints.items():
        if key in model_id:
            return hint
    if '8B' in model_id or '8b' in model_id:
        return '~16 GB'
    if '4B' in model_id or '4b' in model_id:
        return '~8 GB'
    return ''


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class EmbeddingCodebookClassifier:
    """
    Applies a qualitative codebook to segments via sentence-transformer embeddings.

    The model is lazy-loaded on first use and kept alive for both passes.
    For large models (Qwen3-Embedding-8B etc.) the weights are loaded at
    float16 to reduce memory, and encode() calls use a configurable batch size.
    """

    def __init__(self, config: Optional[EmbeddingClassifierConfig] = None):
        self.config = config or EmbeddingClassifierConfig()
        self._model = None        # lazy-loaded SentenceTransformer
        self._embed_dim: int = 0  # set after first encode, used for empty-array fallback

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _get_model(self):
        """
        Lazy-load the sentence-transformer model.

        Loads at float16 when torch is available (halves VRAM/RAM usage for
        large models).  trust_remote_code=True is required for Qwen3-Embedding.
        Falls back gracefully to default precision if model_kwargs unsupported.
        """
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer
        model_id = self.config.embedding_model
        print(f"  Loading embedding model: {model_id}")

        # Attempt float16 loading (sentence-transformers v3+ supports model_kwargs)
        loaded = False
        _last_exc: Optional[Exception] = None
        try:
            import torch
            dtype = torch.float16 if (torch.cuda.is_available() or
                                       hasattr(torch.backends, 'mps')) else torch.float32
            self._model = SentenceTransformer(
                model_id,
                model_kwargs={'torch_dtype': dtype},
                trust_remote_code=True,
            )
            loaded = True
        except TypeError:
            # Older sentence-transformers: model_kwargs not supported
            pass
        except Exception as e:
            _last_exc = e
            _oom = 'out of memory' in str(e).lower() or 'cuda error' in str(e).lower()
            if _oom:
                # GPU OOM: free memory and fall back to CPU so the pipeline can continue
                import gc
                try:
                    import torch
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                print(f"  Warning: CUDA OOM loading {model_id}; retrying on CPU (slower).")
                try:
                    self._model = SentenceTransformer(
                        model_id, trust_remote_code=True, device='cpu'
                    )
                    loaded = True
                except Exception as cpu_exc:
                    _last_exc = cpu_exc
            if not loaded:
                raise RuntimeError(
                    f"Failed to load embedding model '{model_id}'.\n"
                    f"  Ensure the model is downloaded (first run needs ~"
                    f"{_model_size_hint(model_id) or '?? GB'} disk space).\n"
                    f"  Pre-download: python -c \"from sentence_transformers import "
                    f"SentenceTransformer; SentenceTransformer('{model_id}', "
                    f"trust_remote_code=True)\"\n"
                    f"  Error: {_last_exc}"
                ) from _last_exc

        if not loaded:
            # Fallback: load without dtype hint (v2 compatibility)
            try:
                self._model = SentenceTransformer(model_id, trust_remote_code=True)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load embedding model '{model_id}': {e}"
                ) from e

        dim = self._model.get_embedding_dimension()
        self._embed_dim = dim or 1024
        print(f"  Embedding model ready  ({self._embed_dim}-dim embeddings).")
        return self._model

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts as *passage* embeddings (no query prefix).

        Used for codebook definitions, criteria, and exemplar strings.
        Returns (N, D) float32 ndarray; returns a (0, D) zero array if texts is empty.
        """
        if not texts:
            dim = self._embed_dim if self._embed_dim else self._get_model().get_sentence_embedding_dimension() or 1024
            return np.zeros((0, dim), dtype=np.float32)
        return self._get_model().encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=self.config.embedding_batch_size,
        )

    def _embed_queries(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts as *query* embeddings.

        When use_query_prefix=True and the model defines a 'query' prompt
        (Qwen3-Embedding and similar), the instruction prefix is prepended.
        This is the asymmetric encoding path used for segment texts.

        Falls back to plain passage encoding if the model has no 'query' prompt.
        """
        if not texts:
            dim = self._embed_dim if self._embed_dim else self._get_model().get_sentence_embedding_dimension() or 1024
            return np.zeros((0, dim), dtype=np.float32)

        model = self._get_model()
        encode_kwargs: dict = {
            'show_progress_bar': False,
            'convert_to_numpy': True,
            'batch_size': self.config.embedding_batch_size,
        }

        if self.config.use_query_prefix:
            prompts = getattr(model, 'prompts', {})
            if 'query' in prompts:
                encode_kwargs['prompt_name'] = 'query'

        return model.encode(texts, **encode_kwargs)

    # ------------------------------------------------------------------
    # Exemplar I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _load_exemplars(path: str) -> Dict[str, List[str]]:
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def _save_exemplars(path: str, exemplars: Dict[str, List[str]]) -> None:
        with open(path, 'w') as f:
            json.dump(exemplars, f, indent=2)

    # ------------------------------------------------------------------
    # Single-pass classification
    # ------------------------------------------------------------------

    def _run_single_pass(
        self,
        segments: List[Segment],
        codebook: Codebook,
        exemplar_texts_by_code: Dict[str, List[str]],
    ) -> Dict[str, List[CodeAssignment]]:
        """
        One pass of embedding-based codebook classification.

        Encoding strategy
        -----------------
        - Segment texts → query embeddings (asymmetric, with instruction prefix)
        - Definition, criteria, exemplar texts → passage embeddings (no prefix)

        Scoring (per segment i, per code j)
        ------------------------------------
        score[i, j] = cosine_sim(seg_i, def_j)
                    + criteria_weight  * cosine_sim(seg_i, crit_j)
                    + exemplar_weight  * cosine_sim(seg_i, exm_j)  [if exemplars exist]

        Assignment threshold
        --------------------
        baseline_j = max(column mean of scores[:, j], global mean of scores)
        code j is assigned to segment i when: score[i, j] > baseline_j * similarity_threshold
        confidence = score[i, j] / (baseline_j * threshold), clipped to [0, 1]
        """
        if not segments:
            return {}

        targets = codebook.to_embedding_targets()
        n_codes = len(targets)
        cw = self.config.criteria_weight
        ew = self.config.exemplar_weight
        max_words = self.config.max_exemplar_tokens

        # ---- Build text lists ----
        seg_texts = [seg.text for seg in segments]
        def_texts = [t['definition'] for t in targets]
        crit_texts = [t['criteria'] for t in targets]

        exm_texts: List[str] = []
        has_exemplar: List[bool] = []
        for t in targets:
            parts: List[str] = []
            if t['exemplars']:
                parts.append(t['exemplars'])
            for ext in exemplar_texts_by_code.get(t['code_id'], []):
                parts.append(ext)
            combined = ' '.join(parts)
            words = combined.split()
            if len(words) > max_words:
                combined = ' '.join(words[:max_words])
            exm_texts.append(combined)
            has_exemplar.append(bool(combined.strip()))

        # ---- Embed ----
        # Segments are queries; codebook texts are passages (asymmetric encoding)
        seg_embs  = self._embed_queries(seg_texts)  # (n_segs, D)
        def_embs  = self._embed(def_texts)           # (n_codes, D)
        crit_embs = self._embed(crit_texts)          # (n_codes, D)

        # ---- Similarity matrices ----
        def_sim  = cosine_similarity(seg_embs, def_embs)    # (n_segs, n_codes)
        crit_sim = cosine_similarity(seg_embs, crit_embs)   # (n_segs, n_codes)
        scores   = def_sim + cw * crit_sim

        if any(has_exemplar):
            # Exemplars are ground-truth passages; encode as passages too
            exm_embs = self._embed(exm_texts)               # (n_codes, D)
            exm_sim  = cosine_similarity(seg_embs, exm_embs)
            for j in range(n_codes):
                if has_exemplar[j]:
                    scores[:, j] += ew * exm_sim[:, j]

        # ---- Per-code baselines ----
        col_means   = scores.mean(axis=0)
        global_mean = float(scores.mean())
        baselines   = np.maximum(col_means, global_mean)  # (n_codes,)

        # ---- Code assignment ----
        thresh   = self.config.similarity_threshold
        results: Dict[str, List[CodeAssignment]] = {}

        for i, seg in enumerate(segments):
            assignments: List[CodeAssignment] = []
            for j, code_def in enumerate(codebook.codes):
                baseline = baselines[j]
                if baseline <= 0:
                    continue
                if scores[i, j] > baseline * thresh:
                    confidence = min(1.0, max(0.0, scores[i, j] / (baseline * thresh)))
                    assignments.append(CodeAssignment(
                        code_id=code_def.code_id,
                        category=code_def.category,
                        confidence=round(confidence, 4),
                        method='embedding',
                    ))
            results[seg.segment_id] = assignments

        n_assigned   = sum(len(v) for v in results.values())
        n_segs_coded = sum(1 for v in results.values() if v)
        print(f"  → {n_assigned} code assignments across {n_segs_coded}/{len(segments)} segments")
        return results

    # ------------------------------------------------------------------
    # Exemplar accumulation
    # ------------------------------------------------------------------

    def _accumulate_exemplars(
        self,
        segments: List[Segment],
        results: Dict[str, List[CodeAssignment]],
    ) -> Dict[str, List[str]]:
        """
        Collect high-confidence segment texts as exemplars for pass 2.

        Per code: collect segments with confidence >= exemplar_confidence_threshold,
        sort by confidence descending, and truncate combined text to max_exemplar_tokens.
        """
        threshold = self.config.exemplar_confidence_threshold
        max_words = self.config.max_exemplar_tokens
        seg_text  = {seg.segment_id: seg.text for seg in segments}

        code_candidates: Dict[str, List[Tuple[float, str]]] = {}
        for seg_id, assignments in results.items():
            for a in assignments:
                if a.confidence >= threshold:
                    code_candidates.setdefault(a.code_id, []).append(
                        (a.confidence, seg_text.get(seg_id, ''))
                    )

        exemplars: Dict[str, List[str]] = {}
        for code_id, candidates in code_candidates.items():
            candidates.sort(key=lambda x: x[0], reverse=True)
            collected: List[str] = []
            total_words = 0
            for _, text in candidates:
                words = text.split()
                if total_words + len(words) > max_words:
                    remaining = max_words - total_words
                    if remaining > 0 and not collected:
                        collected.append(' '.join(words[:remaining]))
                    break
                collected.append(text)
                total_words += len(words)
            if collected:
                exemplars[code_id] = collected

        return exemplars

    # ------------------------------------------------------------------
    # Main entry-point
    # ------------------------------------------------------------------

    def classify_segments(
        self,
        segments: List[Segment],
        codebook: Codebook,
    ) -> Dict[str, List[CodeAssignment]]:
        """
        Apply codebook to segments via sentence-transformer embedding similarity.

        Two-pass (default):
          Pass 1 — classify using any pre-imported exemplars.
          Exemplar accumulation — gather high-confidence pass-1 segments per code.
          Pass 2 — re-classify with merged imported + discovered exemplars.

        Returns dict mapping segment_id → list of CodeAssignments.
        """
        imported_exemplars: Dict[str, List[str]] = {}
        if self.config.exemplar_import_path:
            imported_exemplars = self._load_exemplars(self.config.exemplar_import_path)
            print(f"  Imported exemplars for {len(imported_exemplars)} codes "
                  f"from {self.config.exemplar_import_path}")

        print(f"  Codebook embedding: {len(segments)} segments × {len(codebook.codes)} codes "
              f"({'two-pass' if self.config.two_pass else 'single-pass'}, "
              f"model: {self.config.embedding_model})")

        # Pass 1
        pass1_results = self._run_single_pass(segments, codebook, imported_exemplars)

        if not self.config.two_pass:
            return pass1_results

        # Exemplar accumulation
        discovered = self._accumulate_exemplars(segments, pass1_results)
        n_disc = sum(len(v) for v in discovered.values())
        print(f"  Exemplar accumulation: {n_disc} texts across {len(discovered)} codes")

        if self.config.exemplar_export_path and discovered:
            self._save_exemplars(self.config.exemplar_export_path, discovered)

        # Merge imported + discovered
        merged: Dict[str, List[str]] = {}
        for code_id in set(list(imported_exemplars) + list(discovered)):
            merged[code_id] = (
                imported_exemplars.get(code_id, [])
                + discovered.get(code_id, [])
            )

        # Pass 2
        print(f"  Pass 2 (enriched exemplars for {len(merged)} codes)...")
        return self._run_single_pass(segments, codebook, merged)
