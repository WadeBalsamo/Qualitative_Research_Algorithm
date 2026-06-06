"""
tests.integration — real-model tier (tiny embeddings + tiny Ollama LLM).

These tests download/run real models and write under tests/testrun-outputs/.
They are NOT collected by the unit runner. Bootstraps the repo root onto
sys.path.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # tests/integration/ -> repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
