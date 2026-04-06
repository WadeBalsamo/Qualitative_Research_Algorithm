"""
config.py
---------
Configuration dataclasses for codebook classification methods.

The embedding classifier uses local Ollama models via /api/embeddings.
Primary: mixtral:8x7b  (ollama pull mixtral:8x7b)
Secondary (optional): mistral:7b-instruct  (ollama pull mistral:7b-instruct)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EmbeddingClassifierConfig:
    """Config for the Ollama embedding codebook classifier.

    Uses one or two local Ollama models for embedding-based classification:
      - primary_model: used for cosine-similarity and cosine-distance axes
      - secondary_model: used for the Euclidean-distance axis; if empty, the
        primary model is reused (single-model mode)

    Embeddings are obtained via the Ollama /api/embeddings endpoint.
    """
    primary_model: str = 'mixtral:8x7b'
    secondary_model: str = 'mistral:7b-instruct'   # set to '' to use primary for all axes
    ollama_host: str = '0.0.0.0'
    ollama_port: int = 11434
    similarity_threshold: float = 1.375
    distance_threshold: float = 1.325
    tertiary_threshold: float = 1.35
    max_codes_per_sentence: Optional[int] = None  # None = auto (33% of codebook, max 6)
    criteria_weight: float = 0.5
    exemplar_weight: float = 0.5
    exemplar_import_path: Optional[str] = None      # JSON file with pre-populated exemplars
    exemplar_export_path: Optional[str] = None      # where to write discovered exemplars
    max_exemplar_tokens: int = 512                  # word-count cap per code
    exemplar_confidence_threshold: float = 0.8      # min confidence to qualify
    two_pass: bool = True                           # enable two-pass classification


@dataclass
class LLMCodebookConfig:
    """Config for the LLM-based codebook classifier."""
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
