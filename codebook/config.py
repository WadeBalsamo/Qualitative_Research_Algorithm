"""
config.py
---------
Configuration dataclasses for codebook classification methods.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EmbeddingClassifierConfig:
    """Config for the sentence-transformer embedding codebook classifier.

    Default model: Qwen/Qwen3-Embedding-8B
    ----------------------------------------
    An 8-billion-parameter dedicated embedding model with 4096-dim output,
    well-suited for complex semantic retrieval over psychological and
    phenomenological construct descriptions.  First use downloads ~16 GB
    from HuggingFace Hub (cached at ~/.cache/huggingface/).

    Asymmetric encoding (use_query_prefix=True, default):
      Segment texts  → encoded with the model's 'query' instruction prefix,
                        treating each transcript excerpt as a retrieval query.
      Codebook texts → encoded as plain passages (definition, criteria, exemplars).
    This improves recall for rare or subtle construct expressions.

    Scoring formula per (segment i, code j):
        score[i, j] = cosine_sim(seg_i, definition_j)
                    + criteria_weight  * cosine_sim(seg_i, criteria_j)
                    + exemplar_weight  * cosine_sim(seg_i, exemplars_j)
                                        (only when exemplars exist for code j)

    A code is assigned when:
        score[i, j] > max(per-code column mean, global mean) * similarity_threshold

    Confidence = score / (baseline * threshold), capped to [0, 1].
    """
    # Embedding model: any sentence-transformers-compatible model ID.
    # Qwen/Qwen3-Embedding-8B is the default for high-quality psychological construct retrieval.
    # Use 'all-MiniLM-L6-v2' for a lightweight alternative (no download needed).
    embedding_model: str = 'Qwen/Qwen3-Embedding-8B'

    # Asymmetric encoding: encode segment texts with the model's 'query' instruction prefix
    # and codebook texts as plain passages.  Improves retrieval of nuanced construct expressions.
    # Set to False for symmetric models that do not define a 'query' prompt (e.g. all-MiniLM).
    use_query_prefix: bool = True

    # Batch size for model.encode() calls.  Reduce for large models on limited VRAM.
    # 8B model: 8 is a safe default on 24 GB VRAM; use 4 if you see OOM errors.
    embedding_batch_size: int = 8

    similarity_threshold: float = 1.375           # score must exceed this multiple of per-code baseline
    max_codes_per_sentence: Optional[int] = None  # None = auto (33% of codebook, max 6)
    criteria_weight: float = 0.5
    exemplar_weight: float = 0.5
    exemplar_import_path: Optional[str] = None    # JSON: {code_id: [text, ...], ...}
    exemplar_export_path: Optional[str] = None    # where to write discovered exemplars
    max_exemplar_tokens: int = 512                # word-count cap per code
    exemplar_confidence_threshold: float = 0.8   # min confidence to qualify for accumulation
    two_pass: bool = True                         # enable two-pass exemplar accumulation


@dataclass
class LLMCodebookConfig:
    """Config for the LLM-based codebook classifier (uses same backend as theme classifier)."""
    n_runs: int = 1
    max_codes_per_segment: int = 5
    confidence_threshold: float = 0.5
    randomize_codebook: bool = True
    save_interval: int = 20
    output_dir: str = ''


@dataclass
class EnsembleConfig:
    """Config for combining embedding + LLM codebook results."""
    require_agreement: bool = False
    flag_disagreements: bool = True
    preferred_method: str = 'llm'  # 'llm', 'embedding', or 'both'
