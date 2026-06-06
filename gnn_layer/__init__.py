"""
gnn_layer
=========
GNN representation-and-discovery layer for QRA.

A Qwen3-embedding graph-neural-network analysis layer that *augments* (never
replaces) the existing LLM/embedding classifiers. It runs at analyze-time (after
master_segments assembly), reads the assembled DataFrame, and writes its own
artifacts to ``03_analysis_data/gnn/`` + ``06_reports/`` (model + cached embeddings
under ``02_meta/gnn/``). It never mutates frozen segments or master_segments.

Capabilities (see methodology.md §8.5 for the as-built specification):
  A. Continuous VAAMR positioning (model superposition via soft ballot targets)
  B. Cue granularization + emergent-motif discovery (flagship)
  C. Independent (non-LLM) measurement substrate for construct validity
  D. Principled ablation & sub-typing
  E. Inductive participant<->therapist coupling (CF/IC discovered, not imposed)

STATUS: fully implemented. Every module has a working implementation (there is no
``NotImplementedError`` in the package). Heavy imports (torch, sklearn,
sentence-transformers) are performed lazily inside functions so this package imports
cleanly without those models being loaded.

The top-level entry point is :func:`gnn_layer.runner.run_gnn_analysis`, invoked from
``analysis/runner.py`` only when ``config.gnn_layer.enabled`` is True (ON by default).
"""

from .config import GnnLayerConfig

__all__ = ["GnnLayerConfig"]
