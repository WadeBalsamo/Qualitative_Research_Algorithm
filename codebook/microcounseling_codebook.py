"""
microcounseling_codebook.py
---------------------------
Therapist-side microcounseling-skills codebook.

Contains 8 codes across 5 domains (Reflective, Affirmative, Relational,
Autonomy-Supportive, Inquiry, Neutral) describing concrete, observable therapist
microskills. This is the behavioural sub-layer of the PURER therapist framework —
the structural twin of how the Varieties of Contemplative Experience (VCE) codebook
is the content sub-layer of the participant-side VAAMR framework.

Mirror of codebook/phenomenology_codebook.py: parsed at runtime from the
hot-reloadable MICROCOUNSELING_CODEBOOK.md via the shared codebook markdown loader.
"""

from pathlib import Path
from .codebook_schema import Codebook, CodeDefinition, slugify  # noqa: F401
from .markdown_loader import load_codebook_md

_CODEBOOK_MD = Path(__file__).resolve().parents[1] / "MICROCOUNSELING_CODEBOOK.md"


def get_microcounseling_codebook() -> Codebook:
    """Return the therapist microcounseling-skills codebook (parsed from MICROCOUNSELING_CODEBOOK.md)."""
    return load_codebook_md(_CODEBOOK_MD)
