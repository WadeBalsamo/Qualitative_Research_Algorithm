"""
tests.unit — hermetic unit tier.

Bootstraps the repository root onto sys.path so test modules import
first-party packages whether run via the dedicated runner, pytest, or
``python -m unittest discover -s tests/unit``.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # tests/unit/ -> repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
