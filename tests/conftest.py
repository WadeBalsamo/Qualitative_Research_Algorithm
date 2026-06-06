"""
conftest.py — repo-root test configuration.

Inserts the repository root onto sys.path so first-party packages
(``process``, ``gnn_layer``, ``analysis``, ``classification_tools``,
``codebook``, ``theme_framework``) import without ``pip install -e .``.
Works for both ``pytest`` and ``python -m unittest discover``.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent  # tests/ -> repo root
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(_REPO_ROOT))
