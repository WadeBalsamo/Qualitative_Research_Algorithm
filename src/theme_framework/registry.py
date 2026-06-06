"""
theme_framework/registry.py
----------------------------
Framework name → ThemeFramework dispatch.

Usage:
    from theme_framework.registry import load, FRAMEWORKS

    fw = load('vaamr')   # ThemeFramework parsed from VAAMR_FRAMEWORK.md
    fw = load('purer')   # ThemeFramework parsed from PURER_FRAMEWORK.md
    fw = load(None)      # → None  (means "no framework for this side")

To register a new framework:
    1. Drop a .md file in the repo root following the PARSER CONTRACT.
    2. Add its name (lowercase) and path to FRAMEWORKS below.
"""

import functools
from pathlib import Path
from typing import Optional

from .markdown_loader import load_framework_md
from .theme_schema import ThemeFramework

_REPO_ROOT = Path(__file__).resolve().parents[2]

FRAMEWORKS: dict[str, Path] = {
    'vaamr': _REPO_ROOT / 'frameworks' / 'VAAMR_FRAMEWORK.md',
    'purer': _REPO_ROOT / 'frameworks' / 'PURER_FRAMEWORK.md',
}


@functools.lru_cache(maxsize=None)
def _cached_load(name: str) -> ThemeFramework:
    """Load and cache a framework by name. Raises KeyError for unknown names."""
    if name not in FRAMEWORKS:
        raise KeyError(f"Unknown framework: {name!r}. Available: {sorted(FRAMEWORKS)}")
    return load_framework_md(FRAMEWORKS[name])


def load(name: Optional[str]) -> Optional[ThemeFramework]:
    """
    Return a ThemeFramework by registry name, or None if name is None.

    Raises KeyError for unknown non-None names.
    """
    if name is None:
        return None
    return _cached_load(name)
