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
enrich comparison targets for pass 2.

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
        self._base_url = (
            f"http://{self.config.ollama_host}:{self.config.ollama_port}"
        )

    # ------------------------------------------------------------------
    # Ollama embedding helper
    # ------------------------------------------------------------------

    def _get_embedding(self, text: str, model: str) -> Optional[np.ndarray]:
        """Fetch a text embedding from the Ollama /api/embeddings endpoint.

        Returns a (1, dim) float32 ndarray, or None on failure.
        """
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": model, "prompt": text},
                    timeout=60,
                )
                data = resp.json()
                vec = data.get("embedding")
                if vec:
                    return np.array(vec, dtype=np.float32).reshape(1, -1)
                return None
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  Ollama embedding error for model {model}: {e}")
                    return None
        return None

    def _embed_all(self, texts: List[str], model: str) -> List[Optional[np.ndarray]]:
        """Embed every text using the given Ollama model."""
        return [self._get_embedding(t, model) for t in texts]

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
            code_id = t['code_id']
            parts = []
            if t['exemplars']:
                parts.append(t['exemplars'])
            for ext_text in exemplar_texts_by_code.get(code_id, []):
                parts.append(ext_text)
            combined = ' '.join(parts)
            words = combined.split()
            if len(words) > max_words:
                combined = ' '.join(words[:max_words])
            exemplar_texts.append(combined)

        return definition_texts, criteria_texts, exemplar_texts

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
        """
        primary = self.config.primary_model
        secondary = self.config.secondary_model or primary

        texts = [seg.text for seg in segments]
        targets = codebook.to_embedding_targets()

        max_codes = self.config.max_codes_per_sentence
        if max_codes is None:
            max_codes = min(int(len(codebook.codes) * 0.33), 6)
        max_codes = max(1, max_codes)

        n_sents = len(texts)
        n_codes = len(codebook.codes)
        weights = np.ones(n_codes)
        cw = self.config.criteria_weight
        ew = self.config.exemplar_weight

        definition_texts, criteria_texts, exemplar_texts = self._build_target_texts(
            targets, exemplar_texts_by_code,
        )
        tertiary_target_texts = [
            d + ' ' + c for d, c in zip(definition_texts, criteria_texts)
        ]

        # ---- Axis 1 & 3: Primary model embeddings ----
        print(f"  [embedding] Computing primary embeddings ({primary})...")
        sent_embs_p = self._embed_all(texts, primary)
        def_embs_p = self._embed_all(definition_texts, primary)
        crit_embs_p = self._embed_all(criteria_texts, primary)
        exm_embs_p = [
            self._get_embedding(t, primary) if t else None
            for t in exemplar_texts
        ]
        tert_embs_p = self._embed_all(tertiary_target_texts, primary)

        # Cosine similarity matrix (axis 1)
        sim_matrix = np.zeros((n_sents, n_codes))
        for i in range(n_sents):
            if sent_embs_p[i] is None:
                continue
            for j in range(n_codes):
                if def_embs_p[j] is None or crit_embs_p[j] is None:
                    continue
                score = (
                    cosine_similarity(sent_embs_p[i], def_embs_p[j])[0, 0]
                    + cw * cosine_similarity(sent_embs_p[i], crit_embs_p[j])[0, 0]
                )
                if exm_embs_p[j] is not None:
                    score += ew * cosine_similarity(sent_embs_p[i], exm_embs_p[j])[0, 0]
                sim_matrix[i, j] = weights[j] * score

        # Cosine distance matrix (axis 3) – uses same primary embeddings
        tert_matrix = np.zeros((n_sents, n_codes))
        for i in range(n_sents):
            if sent_embs_p[i] is None:
                continue
            for j in range(n_codes):
                if tert_embs_p[j] is None:
                    continue
                tert_matrix[i, j] = 1.0 - cosine_similarity(
                    sent_embs_p[i], tert_embs_p[j]
                )[0, 0]

        # ---- Axis 2: Secondary model (Euclidean distance) ----
        if secondary != primary:
            print(f"  [embedding] Computing secondary embeddings ({secondary})...")
            sent_embs_s = self._embed_all(texts, secondary)
            def_embs_s = self._embed_all(definition_texts, secondary)
            crit_embs_s = self._embed_all(criteria_texts, secondary)
            exm_embs_s = [
                self._get_embedding(t, secondary) if t else None
                for t in exemplar_texts
            ]
        else:
            # Reuse primary embeddings for the distance axis
            sent_embs_s = sent_embs_p
            def_embs_s = def_embs_p
            crit_embs_s = crit_embs_p
            exm_embs_s = exm_embs_p

        dist_matrix = np.zeros((n_sents, n_codes))
        for i in range(n_sents):
            if sent_embs_s[i] is None:
                continue
            for j in range(n_codes):
                if def_embs_s[j] is None or crit_embs_s[j] is None:
                    continue
                score = (
                    cdist(sent_embs_s[i], def_embs_s[j], 'euclidean')[0, 0]
                    + cw * cdist(sent_embs_s[i], crit_embs_s[j], 'euclidean')[0, 0]
                )
                if exm_embs_s[j] is not None:
                    score += ew * cdist(sent_embs_s[i], exm_embs_s[j], 'euclidean')[0, 0]
                dist_matrix[i, j] = weights[j] * score

        # ---- Triple-veto scoring ----
        results: Dict[str, List[CodeAssignment]] = {}
        sim_thresh = self.config.similarity_threshold
        dist_thresh = self.config.distance_threshold
        tert_thresh = self.config.tertiary_threshold

        max_candidates = max(1, max_codes // 3)

        for i, segment in enumerate(segments):
            if sent_embs_p[i] is None:
                results[segment.segment_id] = []
                continue

            assignments: List[CodeAssignment] = []

            nearest = np.argsort(dist_matrix[i])[:max_candidates]
            most_similar = np.argsort(-sim_matrix[i])[:max_candidates]
            closest_tertiary = np.argsort(tert_matrix[i])[:max_candidates]

            candidates = set(nearest) | set(most_similar) | set(closest_tertiary)

            for j in candidates:
                avg_sim = max(np.mean(sim_matrix[:, j]), np.mean(sim_matrix))
                avg_dist = max(np.mean(dist_matrix[:, j]), np.mean(dist_matrix))
                avg_tert = max(np.mean(tert_matrix[:, j]), np.mean(tert_matrix))

                passes_similarity = sim_matrix[i, j] > avg_sim * sim_thresh
                passes_distance = dist_matrix[i, j] < avg_dist / dist_thresh
                passes_tertiary = tert_matrix[i, j] < avg_tert / tert_thresh

                if passes_similarity and passes_distance and passes_tertiary:
                    code = codebook.codes[j]
                    confidence = min(1.0, max(0.0,
                        sim_matrix[i, j] / (avg_sim * sim_thresh)
                    ))
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
