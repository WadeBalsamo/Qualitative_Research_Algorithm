"""
config.py
---------
Configuration for theme/stage classification via LLM zero-shot prompting.

Supports single-model or multi-model cross-referencing.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class ThemeClassificationConfig:
    """Parameters for zero-shot LLM theme classification."""
    model: str = 'meta-llama/Llama-4-Maverick-17B-128E-Instruct'  # Primary model (used when models list is empty)
    models: List[str] = field(default_factory=list)  # For multi-model cross-referencing
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
