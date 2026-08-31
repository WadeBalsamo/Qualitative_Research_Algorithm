"""
analysis/cue_distill.py
-----------------------
Algorithmic distillation of proximal therapist cues for the language atlas.

Two operations, both embedding-based (sentence-transformers MiniLM — loadable
under the project's transformers pin), with a deterministic non-embedding
fallback so report generation never blocks on model availability:

  * prototype(cues)          — index of the cue nearest the group centroid:
                               "the centroid proximal cue" for a VAAMR
                               transition type.
  * key_sentences(cue, cues) — the 1–k sentences of `cue` most similar to the
                               group centroid, returned in original order.
                               These are REAL sentences (extractive), never
                               paraphrase.

Fallback (embedder unavailable): prototype = median-length cue;
key sentences = the last k sentences (temporally nearest the TO turn).
"""

import re
from typing import List, Optional, Tuple

import numpy as np

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?…])\s+')
_DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def split_sentences(text: str) -> List[str]:
    """Split transcript text into sentences (punctuation-based, keeps enders)."""
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(str(text or '').strip())]
    return [p for p in parts if p]


class CueDistiller:
    """Embedding-backed cue prototype + key-sentence selection with fallback."""

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        self._failed = False
        self._cache = {}

    @property
    def method(self) -> str:
        return 'tail-fallback' if self._failed else 'embedding'

    def _embed(self, texts: List[str]) -> Optional[np.ndarray]:
        if self._failed:
            return None
        missing = [t for t in texts if t not in self._cache]
        if missing:
            try:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self.model_name)
                vecs = self._model.encode(missing, show_progress_bar=False,
                                          normalize_embeddings=True)
                for t, v in zip(missing, np.asarray(vecs)):
                    self._cache[t] = v
            except Exception:
                self._failed = True
                return None
        return np.stack([self._cache[t] for t in texts])

    def _centroid(self, cues: List[str]) -> Optional[np.ndarray]:
        vecs = self._embed(cues)
        if vecs is None:
            return None
        c = vecs.mean(axis=0)
        n = np.linalg.norm(c)
        return c / n if n > 0 else c

    def prototype(self, cues: List[str]) -> int:
        """Index of the centroid-nearest cue; median-length cue on fallback."""
        if not cues:
            return 0
        if len(cues) == 1:
            return 0
        vecs = self._embed(cues)
        if vecs is None:
            lens = sorted(range(len(cues)), key=lambda i: len(cues[i].split()))
            return lens[len(lens) // 2]
        centroid = vecs.mean(axis=0)
        sims = vecs @ (centroid / (np.linalg.norm(centroid) or 1.0))
        return int(np.argmax(sims))

    def contrast_axis(self, fwd_cues: List[str], bwd_cues: List[str]
                      ) -> Optional[np.ndarray]:
        """Normalized `centroid(fwd) − centroid(bwd)` direction: positive dot
        products mark language characteristic of forward-preceding cues,
        negative of backward-preceding. None when either pool is empty or the
        embedder is unavailable."""
        if not fwd_cues or not bwd_cues:
            return None
        cf = self._centroid(fwd_cues)
        cb = self._centroid(bwd_cues)
        if cf is None or cb is None:
            return None
        axis = cf - cb
        n = np.linalg.norm(axis)
        return axis / n if n > 0 else None

    def score_sentences(self, text: str, axis: np.ndarray
                        ) -> Optional[List[Tuple[str, float]]]:
        """(sentence, axis score) for each sentence of `text`; None on fallback."""
        sents = split_sentences(text)
        if not sents:
            return []
        vecs = self._embed(sents)
        if vecs is None:
            return None
        return list(zip(sents, (vecs @ axis).tolist()))

    def key_sentences(self, cue: str, group_cues: List[str], k: int = 3,
                      axis: Optional[np.ndarray] = None, sign: float = 1.0
                      ) -> Tuple[List[str], bool]:
        """Up to k sentences of `cue`, in original order.

        Ranking: by `sign * axis` score when a contrast axis is given (sign
        +1 for forward groups, −1 for backward), else by similarity to the
        group centroid. Fallback (no embedder): the last k sentences.
        Returns (sentences, elided); elided means ellipses are needed."""
        sents = split_sentences(cue)
        if len(sents) <= k:
            return sents, False
        vecs = self._embed(sents)
        if vecs is None:
            return sents[-k:], True
        if axis is not None:
            sims = (vecs @ axis) * sign
        else:
            centroid = self._centroid(group_cues if group_cues else [cue])
            if centroid is None:
                return sents[-k:], True
            sims = vecs @ centroid
        top = sorted(np.argsort(sims)[::-1][:k])
        elided = any(b - a > 1 for a, b in zip(top, top[1:])) or top[0] > 0 \
            or top[-1] < len(sents) - 1
        return [sents[i] for i in top], elided
