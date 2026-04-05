"""
embedding_classifier.py
-----------------------
Multi-label codebook classification using embeddings from local Ollama models.

Embeddings are retrieved via the Ollama /api/embeddings endpoint (no GPU or
HuggingFace weights required – just a running Ollama service).

Classification uses a dual-axis scoring system:
  1. Primary model (mixtral:8x7b)  – cosine similarity + cosine distance axes
  2. Secondary model (mistral:7b-instruct, optional) – Euclidean distance axis
     If secondary_model is empty, the primary model handles all three axes.

All three conditions must pass for a code to be assigned (triple-veto).

Supports a two-pass approach: pass 1 discovers high-confidence exemplars that
enrich comparison targets for pass 2.  Segment and target embeddings that do
not change between passes are cached to avoid redundant HTTP requests.

Accepts Segment objects and Codebook instances.
"""

import json
import time
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from typing import Dict, List, Optional, Tuple

from classification_tools.data_structures import Segment
from .codebook_schema import Codebook, CodeAssignment
from .config import EmbeddingClassifierConfig


class EmbeddingCodebookClassifier:
    """
    Applies a qualitative codebook to segments via Ollama embedding similarity.

    Uses one or two local Ollama models for a triple-metric scoring system:
      1. Cosine similarity   (primary model)
      2. Euclidean distance  (secondary model, or primary if not set)
      3. Cosine distance     (primary model)

    All three conditions must be met for a code to be assigned.

    Supports a two-pass approach where pass 1 discovers high-confidence
    exemplar segments that enrich the comparison targets for pass 2.
    """

    def __init__(self, config: Optional[EmbeddingClassifierConfig] = None):
        self.config = config or EmbeddingClassifierConfig()
        # Cache keyed by (model_name, text) — avoids re-fetching identical
        # embeddings across two-pass runs or repeated target lookups.
        self._emb_cache: Dict[Tuple[str, str], Optional[np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Ollama embedding helpers
    # ------------------------------------------------------------------

    def _ollama_base_url(self) -> str:
        return f"http://{self.config.ollama_host}:{self.config.ollama_port}"

    def _get_embedding(self, text: str, model: str) -> Optional[np.ndarray]:
        """Fetch a text embedding from the Ollama /api/embeddings endpoint.

        Returns a (1, dim) float32 ndarray, or None on failure.
        Uses config-driven retry with exponential backoff.
        """
        for attempt in range(self.config.max_retries):
            try:
                resp = requests.post(
                    f"{self._ollama_base_url()}/api/embeddings",
                    json={"model": model, "prompt": text},
                    timeout=60,
                )
                vec = resp.json().get("embedding")
                if vec:
                    return np.array(vec, dtype=np.float32).reshape(1, -1)
                return None
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_base_delay * (2 ** attempt))
                else:
                    print(f"  Ollama embedding error ({model}): {e}")
                    return None
        return None

    def _embed_cached(self, text: str, model: str) -> Optional[np.ndarray]:
        """Return a cached embedding, fetching from Ollama on first access."""
        key = (model, text)
        if key not in self._emb_cache:
            self._emb_cache[key] = self._get_embedding(text, model)
        return self._emb_cache[key]

    def _embed_all(self, texts: List[str], model: str) -> List[Optional[np.ndarray]]:
        return [self._embed_cached(t, model) for t in texts]

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
    # Target text construction
    # ------------------------------------------------------------------

    def _build_target_texts(
        self,
        targets: List[Dict[str, str]],
        exemplar_texts_by_code: Dict[str, List[str]],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Build definition, criteria, and exemplar text lists from codebook targets.

        Returns (definition_texts, criteria_texts, exemplar_texts).
        """
        max_words = self.config.max_exemplar_tokens
        definition_texts = [t['definition'] for t in targets]
        criteria_texts = [t['criteria'] for t in targets]

        exemplar_texts = []
        for t in targets:
            parts = []
            if t['exemplars']:
                parts.append(t['exemplars'])
            for ext_text in exemplar_texts_by_code.get(t['code_id'], []):
                parts.append(ext_text)
            combined = ' '.join(parts)
            words = combined.split()
            if len(words) > max_words:
                combined = ' '.join(words[:max_words])
            exemplar_texts.append(combined)

        return definition_texts, criteria_texts, exemplar_texts

    # ------------------------------------------------------------------
    # Vectorized matrix helpers
    # ------------------------------------------------------------------

    def _build_sim_matrix(
        self,
        sent_embs: List[Optional[np.ndarray]],
        def_embs: List[Optional[np.ndarray]],
        crit_embs: List[Optional[np.ndarray]],
        exm_embs: List[Optional[np.ndarray]],
        weights: np.ndarray,
        cw: float,
        ew: float,
    ) -> np.ndarray:
        """Vectorized cosine-similarity matrix (n_sents × n_codes)."""
        n_sents, n_codes = len(sent_embs), len(def_embs)
        matrix = np.zeros((n_sents, n_codes))

        s_idx = [i for i, e in enumerate(sent_embs) if e is not None]
        c_idx = [j for j in range(n_codes)
                 if def_embs[j] is not None and crit_embs[j] is not None]
        if not s_idx or not c_idx:
            return matrix

        S = np.vstack([sent_embs[i] for i in s_idx])
        D = np.vstack([def_embs[j] for j in c_idx])
        C = np.vstack([crit_embs[j] for j in c_idx])

        scores = cosine_similarity(S, D) + cw * cosine_similarity(S, C)

        for jj, j in enumerate(c_idx):
            if exm_embs[j] is not None:
                scores[:, jj] += ew * cosine_similarity(S, exm_embs[j])[:, 0]

        scores *= weights[np.array(c_idx)][np.newaxis, :]
        matrix[np.ix_(np.array(s_idx), np.array(c_idx))] = scores
        return matrix

    def _build_dist_matrix(
        self,
        sent_embs: List[Optional[np.ndarray]],
        def_embs: List[Optional[np.ndarray]],
        crit_embs: List[Optional[np.ndarray]],
        exm_embs: List[Optional[np.ndarray]],
        weights: np.ndarray,
        cw: float,
        ew: float,
    ) -> np.ndarray:
        """Vectorized Euclidean-distance matrix (n_sents × n_codes)."""
        n_sents, n_codes = len(sent_embs), len(def_embs)
        matrix = np.zeros((n_sents, n_codes))

        s_idx = [i for i, e in enumerate(sent_embs) if e is not None]
        c_idx = [j for j in range(n_codes)
                 if def_embs[j] is not None and crit_embs[j] is not None]
        if not s_idx or not c_idx:
            return matrix

        S = np.vstack([sent_embs[i] for i in s_idx])
        D = np.vstack([def_embs[j] for j in c_idx])
        C = np.vstack([crit_embs[j] for j in c_idx])

        scores = cdist(S, D, 'euclidean') + cw * cdist(S, C, 'euclidean')

        for jj, j in enumerate(c_idx):
            if exm_embs[j] is not None:
                scores[:, jj] += ew * cdist(S, exm_embs[j], 'euclidean')[:, 0]

        scores *= weights[np.array(c_idx)][np.newaxis, :]
        matrix[np.ix_(np.array(s_idx), np.array(c_idx))] = scores
        return matrix

    def _build_tert_matrix(
        self,
        sent_embs: List[Optional[np.ndarray]],
        tert_embs: List[Optional[np.ndarray]],
    ) -> np.ndarray:
        """Vectorized cosine-distance matrix (n_sents × n_codes)."""
        n_sents, n_codes = len(sent_embs), len(tert_embs)
        matrix = np.zeros((n_sents, n_codes))

        s_idx = [i for i, e in enumerate(sent_embs) if e is not None]
        t_idx = [j for j, e in enumerate(tert_embs) if e is not None]
        if not s_idx or not t_idx:
            return matrix

        S = np.vstack([sent_embs[i] for i in s_idx])
        T = np.vstack([tert_embs[j] for j in t_idx])

        matrix[np.ix_(np.array(s_idx), np.array(t_idx))] = 1.0 - cosine_similarity(S, T)
        return matrix

    # ------------------------------------------------------------------
    # Single-pass classification
    # ------------------------------------------------------------------

    def _run_single_pass(
        self,
        segments: List[Segment],
        codebook: Codebook,
        exemplar_texts_by_code: Dict[str, List[str]],
    ) -> Dict[str, List[CodeAssignment]]:
        """Run one full pass of Ollama-based embedding classification.

        Uses the primary Ollama model (and optionally a secondary model) to
        compute embeddings, then applies a triple-metric veto.
        Embeddings are retrieved via _embed_cached, so definitions/criteria
        and segment texts computed in pass 1 are reused in pass 2 at no cost.
        """
        primary = self.config.primary_model
        secondary = self.config.secondary_model or primary

        texts = [seg.text for seg in segments]
        targets = codebook.to_embedding_targets()

        max_codes = self.config.max_codes_per_sentence
        if max_codes is None:
            max_codes = min(int(len(codebook.codes) * 0.33), 6)
        max_codes = max(1, max_codes)

        n_codes = len(codebook.codes)
        weights = np.ones(n_codes)
        cw = self.config.criteria_weight
        ew = self.config.exemplar_weight

        definition_texts, criteria_texts, exemplar_texts = self._build_target_texts(
            targets, exemplar_texts_by_code,
        )
        tertiary_target_texts = [d + ' ' + c for d, c in zip(definition_texts, criteria_texts)]

        # ---- Primary model: similarity + tertiary axes ----
        print(f"  [embedding] Primary embeddings ({primary})...")
        sent_embs_p = self._embed_all(texts, primary)
        def_embs_p = self._embed_all(definition_texts, primary)
        crit_embs_p = self._embed_all(criteria_texts, primary)
        exm_embs_p = [self._embed_cached(t, primary) if t else None for t in exemplar_texts]
        tert_embs_p = self._embed_all(tertiary_target_texts, primary)

        sim_matrix = self._build_sim_matrix(sent_embs_p, def_embs_p, crit_embs_p, exm_embs_p, weights, cw, ew)
        tert_matrix = self._build_tert_matrix(sent_embs_p, tert_embs_p)

        # ---- Secondary model: distance axis ----
        if secondary != primary:
            print(f"  [embedding] Secondary embeddings ({secondary})...")
            sent_embs_s = self._embed_all(texts, secondary)
            def_embs_s = self._embed_all(definition_texts, secondary)
            crit_embs_s = self._embed_all(criteria_texts, secondary)
            exm_embs_s = [self._embed_cached(t, secondary) if t else None for t in exemplar_texts]
        else:
            sent_embs_s, def_embs_s, crit_embs_s, exm_embs_s = sent_embs_p, def_embs_p, crit_embs_p, exm_embs_p

        dist_matrix = self._build_dist_matrix(sent_embs_s, def_embs_s, crit_embs_s, exm_embs_s, weights, cw, ew)

        # ---- Triple-veto scoring ----
        sim_thresh = self.config.similarity_threshold
        dist_thresh = self.config.distance_threshold
        tert_thresh = self.config.tertiary_threshold
        max_candidates = max(1, max_codes // 3)

        results: Dict[str, List[CodeAssignment]] = {}
        for i, segment in enumerate(segments):
            if sent_embs_p[i] is None:
                results[segment.segment_id] = []
                continue

            nearest = np.argsort(dist_matrix[i])[:max_candidates]
            most_similar = np.argsort(-sim_matrix[i])[:max_candidates]
            closest_tertiary = np.argsort(tert_matrix[i])[:max_candidates]
            candidates = set(nearest) | set(most_similar) | set(closest_tertiary)

            assignments: List[CodeAssignment] = []
            for j in candidates:
                avg_sim = max(np.mean(sim_matrix[:, j]), np.mean(sim_matrix))
                avg_dist = max(np.mean(dist_matrix[:, j]), np.mean(dist_matrix))
                avg_tert = max(np.mean(tert_matrix[:, j]), np.mean(tert_matrix))

                if (sim_matrix[i, j] > avg_sim * sim_thresh
                        and dist_matrix[i, j] < avg_dist / dist_thresh
                        and tert_matrix[i, j] < avg_tert / tert_thresh):
                    code = codebook.codes[j]
                    confidence = min(1.0, max(0.0, sim_matrix[i, j] / (avg_sim * sim_thresh)))
                    assignments.append(CodeAssignment(
                        code_id=code.code_id,
                        category=code.category,
                        confidence=round(confidence, 4),
                        method='embedding',
                    ))

            results[segment.segment_id] = assignments

        return results

    # ------------------------------------------------------------------
    # Exemplar accumulation from pass-1 results
    # ------------------------------------------------------------------

    def _accumulate_exemplars(
        self,
        segments: List[Segment],
        results: Dict[str, List[CodeAssignment]],
    ) -> Dict[str, List[str]]:
        """Collect high-confidence segment texts as exemplars."""
        threshold = self.config.exemplar_confidence_threshold
        max_words = self.config.max_exemplar_tokens

        seg_text = {seg.segment_id: seg.text for seg in segments}

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
    # Main classification entry-point
    # ------------------------------------------------------------------

    def classify_segments(
        self,
        segments: List[Segment],
        codebook: Codebook,
    ) -> Dict[str, List[CodeAssignment]]:
        """
        Apply codebook to all segments via Ollama embedding similarity.

        If two_pass is enabled (default):
          1. Pass 1 with any pre-populated exemplars
          2. Accumulates high-confidence segments as new exemplars
          3. Pass 2 with merged exemplars for improved accuracy

        Segment and codebook-definition embeddings are cached internally,
        so pass 2 only fetches embeddings for the new exemplar texts.

        Returns a dict mapping segment_id -> list of CodeAssignments.
        """
        imported_exemplars: Dict[str, List[str]] = {}
        if self.config.exemplar_import_path:
            imported_exemplars = self._load_exemplars(self.config.exemplar_import_path)

        pass1_results = self._run_single_pass(segments, codebook, imported_exemplars)

        if not self.config.two_pass:
            return pass1_results

        discovered_exemplars = self._accumulate_exemplars(segments, pass1_results)

        if self.config.exemplar_export_path and discovered_exemplars:
            self._save_exemplars(self.config.exemplar_export_path, discovered_exemplars)

        merged_exemplars: Dict[str, List[str]] = {}
        for code_id in set(list(imported_exemplars.keys()) + list(discovered_exemplars.keys())):
            merged_exemplars[code_id] = (
                imported_exemplars.get(code_id, [])
                + discovered_exemplars.get(code_id, [])
            )

        return self._run_single_pass(segments, codebook, merged_exemplars)
