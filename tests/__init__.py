"""
Test package bootstrap: ensure src/ is on sys.path so packages are
importable whether run via pytest, python -m unittest, or run_all_tests.py.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent  # src/tests/ → src/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
