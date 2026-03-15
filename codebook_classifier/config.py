"""
config.py
---------
Configuration dataclasses for codebook classification methods.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EmbeddingClassifierConfig:
    """Config for the triple-LLM embedding codebook classifier.

    Uses three causal LLMs from shared/model_loader.py, each providing
    an independent embedding perspective via hidden-state mean-pooling:
      1. similarity_model  – cosine similarity axis
      2. distance_model    – Euclidean distance axis
      3. tertiary_model    – cosine distance axis (replaces sentiment)
    """
    similarity_model: str = 'meta-llama/Llama-4-Maverick-17B-128E-Instruct'
    distance_model: str = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    tertiary_model: str = 'Qwen/Qwen3-Next-80B-A3B-Instruct'
    similarity_threshold: float = 1.375
    distance_threshold: float = 1.325
    tertiary_threshold: float = 1.35
    max_codes_per_sentence: Optional[int] = None  # None = auto (33% of codebook, max 6)
    criteria_weight: float = 0.5
    exemplar_weight: float = 0.5
    sequential_loading: bool = True  # load/unload models one at a time to manage VRAM
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
