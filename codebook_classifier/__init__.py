"""
codebook_classifier
-------------------
Multi-label codebook application via embedding similarity and LLM methods.

Supports any user-defined codebook conforming to the Codebook/CodeDefinition
schema, with the meditation phenomenology codebook as the reference preset.
"""

from .codebook_schema import Codebook, CodeDefinition, CodeAssignment
