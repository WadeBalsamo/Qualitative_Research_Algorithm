"""
process
-------
Unified orchestrator for transcript ingestion, theme classification,
codebook classification, cross-validation, and dataset assembly.
"""

from .orchestrator import run_full_pipeline, PipelineObserver, SilentObserver
