"""
config.py
---------
Configuration for theme/stage classification via LLM zero-shot prompting.

Uses a local Ollama backend with mixtral:8x7b by default.
Pull the model with: ollama pull mixtral:8x7b
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ThemeClassificationConfig:
    """Parameters for zero-shot LLM theme classification."""
    # Primary model served via local Ollama (ollama pull mixtral:8x7b)
    model: str = 'mixtral:8x7b'
    models: List[str] = field(default_factory=list)  # Optional second model for cross-referencing
    temperature: float = 0.0
    n_runs: int = 3
    randomize_codebook: bool = True
    backend: str = 'ollama'
    max_new_tokens: int = 512
    output_dir: str = './data/output/llm_labels/'
    save_interval: int = 20
    ollama_host: str = '0.0.0.0'
    ollama_port: int = 11434
