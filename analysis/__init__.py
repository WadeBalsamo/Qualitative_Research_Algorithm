"""
analysis/
---------
Post-hoc results analysis module for the QRA pipeline.

Reads from master_segment_dataset.csv (produced by the pipeline) and generates:
  - Per-participant longitudinal reports
  - Per-session analyses
  - Per-construct (VA-MR stage + codebook code) reports
  - Graph-ready CSVs for visualization
  - A cross-participant longitudinal summary
  - Session stage progression (with per-participant granularity)
  - Matplotlib analysis figures (heatmaps, trajectories, transitions)
"""

from .runner import run_analysis

__all__ = ['run_analysis']
