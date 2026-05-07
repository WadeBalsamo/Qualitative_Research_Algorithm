"""
process/__init__.py
-------------------
Top-level package init for the QRA classification pipeline.

Exports run_full_pipeline (the 8-stage orchestrator) and observer base
classes. Submodules handle ingestion, segmentation I/O, classification
overlays, assembly, configuration, and output indexing.
"""

from .orchestrator import run_full_pipeline, PipelineObserver, SilentObserver
