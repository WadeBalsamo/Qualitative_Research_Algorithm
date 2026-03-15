"""
embedding_classifier.py
-----------------------
Multi-label codebook classification using hidden-state embeddings from
three causal LLMs (cosine similarity, Euclidean distance, cosine distance).

Each model provides an independent embedding perspective via mean-pooled
final hidden layer outputs.  Models are loaded and unloaded sequentially
by default to manage VRAM.

Supports a two-pass approach: pass 1 discovers high-confidence exemplar
segments that enrich the comparison targets for pass 2.

Accepts Segment objects and Codebook instances.
"""

import json
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from typing import Dict, List, Optional, Tuple

from shared.data_structures import Segment
from shared.model_loader import load_model, unload_model
from .codebook_schema import Codebook, CodeAssignment
from .config import EmbeddingClassifierConfig


class EmbeddingCodebookClassifier:
    """
    Applies a qualitative codebook to segments via LLM embedding similarity.

    Uses a triple-veto scoring system with three distinct causal LLMs:
    1. Cosine similarity  (Model 1 – Llama 4 Maverick)
    2. Euclidean distance  (Model 2 – Mixtral 8x7B)
    3. Cosine distance     (Model 3 – Qwen 3 Next, replaces sentiment)

    All three conditions must be met for a code to be assigned.

    Supports a two-pass approach where pass 1 discovers high-confidence
    exemplar segments that enrich the comparison targets for pass 2.
    """

    def __init__(self, config: Optional[EmbeddingClassifierConfig] = None):
        self.config = config or EmbeddingClassifierConfig()

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    @staticmethod
    def _embed_text_with_model(
        text: str,
        model,
        tokenizer,
    ) -> Optional[np.ndarray]:
        """Compute mean-pooled embedding from a causal LLM's final hidden layer.

        Returns a (1, hidden_dim) float32 ndarray, or None on failure.
        """
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            # Move inputs to model device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # Last hidden layer: (batch=1, seq_len, hidden_dim)
            hidden = outputs.hidden_states[-1]
            # Mean-pool over sequence length → (1, hidden_dim)
            pooled = hidden.mean(dim=1).float().cpu().numpy()
            return pooled
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Batch embedding for a given model
    # ------------------------------------------------------------------

    def _embed_all(
        self,
        texts: List[str],
        model,
        tokenizer,
    ) -> List[Optional[np.ndarray]]:
        """Embed every text with the provided model/tokenizer pair."""
        return [
            self._embed_text_with_model(t, model, tokenizer)
            for t in texts
        ]

    # ------------------------------------------------------------------
    # Exemplar I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _load_exemplars(path: str) -> Dict[str, List[str]]:
        """Load exemplar texts from a JSON file.

        File format: {"code_id": ["text1", "text2"], ...}
        """
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def _save_exemplars(path: str, exemplars: Dict[str, List[str]]) -> None:
        """Save exemplar texts to a JSON file."""
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
        """Build the three target text lists from codebook targets and external exemplars.

        Returns (definition_texts, criteria_texts, exemplar_texts) where
        exemplar_texts[j] may be empty string if no exemplars are available.
        """
        max_words = self.config.max_exemplar_tokens
        definition_texts = [t['definition'] for t in targets]
        criteria_texts = [t['criteria'] for t in targets]

        exemplar_texts = []
        for t in targets:
            code_id = t['code_id']
            # Combine codebook exemplars with external exemplars
            parts = []
            if t['exemplars']:
                parts.append(t['exemplars'])
            for ext_text in exemplar_texts_by_code.get(code_id, []):
                parts.append(ext_text)

            combined = ' '.join(parts)
            # Truncate to max_words
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
        """Run one full pass of triple-LLM embedding classification.

        Contains all model-loading, embedding, matrix-building, and
        triple-veto logic.
        """
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
        # For the tertiary axis: definition + criteria
        tertiary_target_texts = [
            d + ' ' + c for d, c in zip(definition_texts, criteria_texts)
        ]

        # ---- Phase 1: Similarity axis (Llama 4 Maverick) ----
        model, tokenizer = load_model(self.config.similarity_model)

        sent_embs_1 = self._embed_all(texts, model, tokenizer)
        def_embs_1 = self._embed_all(definition_texts, model, tokenizer)
        crit_embs_1 = self._embed_all(criteria_texts, model, tokenizer)
        exm_embs_1 = [
            self._embed_text_with_model(t, model, tokenizer) if t else None
            for t in exemplar_texts
        ]

        if self.config.sequential_loading:
            unload_model(self.config.similarity_model)

        sim_matrix = np.zeros((n_sents, n_codes))
        for i in range(n_sents):
            if sent_embs_1[i] is None:
                continue
            for j in range(n_codes):
                if def_embs_1[j] is None or crit_embs_1[j] is None:
                    continue
                score = (
                    cosine_similarity(sent_embs_1[i], def_embs_1[j])[0, 0]
                    + cw * cosine_similarity(sent_embs_1[i], crit_embs_1[j])[0, 0]
                )
                if exm_embs_1[j] is not None:
                    score += ew * cosine_similarity(sent_embs_1[i], exm_embs_1[j])[0, 0]
                sim_matrix[i, j] = weights[j] * score

        # ---- Phase 2: Distance axis (Mixtral 8x7B) ----
        model, tokenizer = load_model(self.config.distance_model)

        sent_embs_2 = self._embed_all(texts, model, tokenizer)
        def_embs_2 = self._embed_all(definition_texts, model, tokenizer)
        crit_embs_2 = self._embed_all(criteria_texts, model, tokenizer)
        exm_embs_2 = [
            self._embed_text_with_model(t, model, tokenizer) if t else None
            for t in exemplar_texts
        ]

        if self.config.sequential_loading:
            unload_model(self.config.distance_model)

        dist_matrix = np.zeros((n_sents, n_codes))
        for i in range(n_sents):
            if sent_embs_2[i] is None:
                continue
            for j in range(n_codes):
                if def_embs_2[j] is None or crit_embs_2[j] is None:
                    continue
                score = (
                    cdist(sent_embs_2[i], def_embs_2[j], 'euclidean')[0, 0]
                    + cw * cdist(sent_embs_2[i], crit_embs_2[j], 'euclidean')[0, 0]
                )
                if exm_embs_2[j] is not None:
                    score += ew * cdist(sent_embs_2[i], exm_embs_2[j], 'euclidean')[0, 0]
                dist_matrix[i, j] = weights[j] * score

        # ---- Phase 3: Tertiary axis (Qwen 3 Next – replaces sentiment) ----
        model, tokenizer = load_model(self.config.tertiary_model)

        sent_embs_3 = self._embed_all(texts, model, tokenizer)
        tert_target_embs = self._embed_all(tertiary_target_texts, model, tokenizer)

        if self.config.sequential_loading:
            unload_model(self.config.tertiary_model)

        tert_matrix = np.zeros((n_sents, n_codes))
        for i in range(n_sents):
            if sent_embs_3[i] is None:
                continue
            for j in range(n_codes):
                if tert_target_embs[j] is None:
                    continue
                # Cosine distance = 1 - cosine_similarity
                tert_matrix[i, j] = 1.0 - cosine_similarity(
                    sent_embs_3[i], tert_target_embs[j]
                )[0, 0]

        # ---- Triple-veto scoring ----
        results: Dict[str, List[CodeAssignment]] = {}
        sim_thresh = self.config.similarity_threshold
        dist_thresh = self.config.distance_threshold
        tert_thresh = self.config.tertiary_threshold

        max_candidates = max(1, max_codes // 3)

        for i, segment in enumerate(segments):
            if sent_embs_1[i] is None:
                results[segment.segment_id] = []
                continue

            assignments: List[CodeAssignment] = []

            nearest = np.argsort(dist_matrix[i])[:max_candidates]
            most_similar = np.argsort(-sim_matrix[i])[:max_candidates]
            closest_tertiary = np.argsort(tert_matrix[i])[:max_candidates]

            candidates = set(nearest) | set(most_similar) | set(closest_tertiary)

            for j in candidates:
                avg_sim = max(
                    np.mean(sim_matrix[:, j]),
                    np.mean(sim_matrix),
                )
                avg_dist = max(
                    np.mean(dist_matrix[:, j]),
                    np.mean(dist_matrix),
                )
                avg_tert = max(
                    np.mean(tert_matrix[:, j]),
                    np.mean(tert_matrix),
                )

                passes_tertiary = tert_matrix[i, j] < avg_tert / tert_thresh
                passes_similarity = sim_matrix[i, j] > avg_sim * sim_thresh
                passes_distance = dist_matrix[i, j] < avg_dist / dist_thresh

                if passes_tertiary and passes_similarity and passes_distance:
                    code = codebook.codes[j]
                    # Compute a normalized confidence from the similarity score
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
        """Collect high-confidence segment texts as exemplars from classification results.

        Per code: sorts by confidence descending, concatenates texts, and
        truncates to max_exemplar_tokens words.
        """
        threshold = self.config.exemplar_confidence_threshold
        max_words = self.config.max_exemplar_tokens

        # Build segment_id -> text lookup
        seg_text = {seg.segment_id: seg.text for seg in segments}

        # Collect (confidence, text) per code
        code_candidates: Dict[str, List[Tuple[float, str]]] = {}
        for seg_id, assignments in results.items():
            for a in assignments:
                if a.confidence >= threshold:
                    code_candidates.setdefault(a.code_id, []).append(
                        (a.confidence, seg_text.get(seg_id, ''))
                    )

        # Sort by confidence descending, truncate to word budget
        exemplars: Dict[str, List[str]] = {}
        for code_id, candidates in code_candidates.items():
            candidates.sort(key=lambda x: x[0], reverse=True)
            collected: List[str] = []
            total_words = 0
            for _, text in candidates:
                words = text.split()
                if total_words + len(words) > max_words:
                    # Take partial if we have nothing yet
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
        Apply codebook to all segments via triple-LLM embedding similarity.

        If two_pass is enabled (default), runs two passes:
          1. Pass 1 with any pre-populated exemplars
          2. Accumulates high-confidence segments as new exemplars
          3. Pass 2 with merged exemplars for improved accuracy

        Returns a dict mapping segment_id -> list of assigned CodeAssignments.
        """
        # Load pre-populated exemplars if available
        imported_exemplars: Dict[str, List[str]] = {}
        if self.config.exemplar_import_path:
            imported_exemplars = self._load_exemplars(self.config.exemplar_import_path)

        # Pass 1
        pass1_results = self._run_single_pass(segments, codebook, imported_exemplars)

        if not self.config.two_pass:
            return pass1_results

        # Accumulate exemplars from pass-1 results
        discovered_exemplars = self._accumulate_exemplars(segments, pass1_results)

        # Save discovered exemplars if export path is set
        if self.config.exemplar_export_path and discovered_exemplars:
            self._save_exemplars(self.config.exemplar_export_path, discovered_exemplars)

        # Merge imported + discovered exemplars
        merged_exemplars: Dict[str, List[str]] = {}
        for code_id in set(list(imported_exemplars.keys()) + list(discovered_exemplars.keys())):
            merged_exemplars[code_id] = (
                imported_exemplars.get(code_id, [])
                + discovered_exemplars.get(code_id, [])
            )

        # Pass 2 with enriched exemplars
        return self._run_single_pass(segments, codebook, merged_exemplars)
