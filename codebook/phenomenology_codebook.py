"""
phenomenology.py
----------------
Meditation phenomenology codebook. 

Contains 54 codes across 6 domains (Affective, Cognitive, Perceptual,
Sense of Self, Social, Somatic) based on the Varieties of Contemplative
Experience (VCE) codebook.

"""

import functools
from pathlib import Path
from .codebook_schema import Codebook, CodeDefinition, slugify  # noqa: F401
from .markdown_loader import load_codebook_md

_CODEBOOK_MD = Path(__file__).resolve().parents[1] / "PHENOMENOLOGY_CODEBOOK.md"


def get_phenomenology_codebook() -> Codebook:
    """Return the meditation phenomenology codebook (parsed from PHENOMENOLOGY_CODEBOOK.md)."""
    return load_codebook_md(_CODEBOOK_MD)

