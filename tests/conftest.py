"""
conftest.py — repo-root pytest configuration.

Inserts src/ into sys.path so packages in src/ are importable without
requiring `pip install -e .`.  Works for both `pytest` and
`python -m unittest discover -s src/tests`.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
