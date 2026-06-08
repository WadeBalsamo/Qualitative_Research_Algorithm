"""
constructs
----------
Coding-scheme DEFINITIONS — the categories the pipeline classifies into. This is
the definition layer; the classifiers that apply these definitions live in
``classification_tools``.

Two parallel definition families, each markdown-driven (parsed from frameworks/*.md):

  - Theme frameworks (single-label): ThemeFramework/ThemeDefinition schema +
    markdown_loader + registry; VAAMR (participant stages) and PURER (therapist
    moves) as reference presets (vaamr.py / purer.py).
  - Codebook (multi-label): the ``constructs.codebook`` subpackage — Codebook/
    CodeDefinition schema + its own markdown_loader (deliberately beside the
    framework loader) + the VCE phenomenology codebook preset.
"""

from .theme_schema import ThemeFramework, ThemeDefinition
