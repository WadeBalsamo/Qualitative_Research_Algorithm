"""
config.py
---------
Configuration for theme/stage classification via LLM zero-shot prompting.

Supports single-model or multi-model cross-referencing.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ThemeClassificationConfig:
    """Parameters for zero-shot LLM theme classification."""
    model: str = 'meta-llama/Llama-4-Maverick-17B-128E-Instruct'  # Primary model (used when models list is empty)
    summarization_model: str = 'nvidia/nemotron-3-nano-4b'  # Lighter model for cue/session/participant report summaries
    models: List[str] = field(default_factory=list)  # For multi-model cross-referencing
    # Per-run model assignment: when len == n_runs, run[i] uses per_run_models[i].
    # This improves interrater reliability by making each run an independent rater.
    # Required when n_runs is 2 or 3 (wizard enforces this).
    per_run_models: List[str] = field(default_factory=list)
    temperature: float = 0.0
    n_runs: int = 3
    randomize_codebook: bool = True
    api_key: str = field(default_factory=lambda: os.environ.get('OPENROUTER_API_KEY', ''))
    backend: str = 'huggingface'  # 'openrouter', 'replicate', 'ollama', or 'huggingface'
    replicate_api_token: str = field(default_factory=lambda: os.environ.get('REPLICATE_API_TOKEN', ''))
    max_new_tokens: int = 512
    output_dir: str = './data/output/llm_labels/'
    save_interval: int = 20
    ollama_host: str = '0.0.0.0'
    ollama_port: int = 11434
    lmstudio_base_url: str = 'http://127.0.0.1:1234/v1'  # LM Studio server URL
    context_window_segments: int = 2  # Number of preceding segments to include as context (0 = disabled)
    # Prompt exemplar control
    zero_shot_prompt: bool = False          # True = definitions only, no examples
    prompt_n_exemplars: Optional[int] = None        # None = all available
    prompt_include_subtle: bool = True
    prompt_n_subtle: Optional[int] = None           # None = all available
    prompt_include_adversarial: bool = True
    prompt_n_adversarial: Optional[int] = None      # None = all available
    # Evidence-based secondary vote reconciliation
    evidence_secondary_weight: float = 0.6  # Weight applied to secondary/dissenting votes
    evidence_presence_threshold: float = 0.5  # Min pooled evidence score for secondary label
    # Merging short segments: set to 0 to disable (e.g. for content-validity test runs)
    min_classifiable_words: int = 10
