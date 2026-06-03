"""
gnn_layer/reports.py
--------------------
CSV / text-report writers for the GNN layer (SCAFFOLD).

Mirrors the export style of analysis/purer_analysis.py (export_purer_csvs /
generate_purer_report) and uses process/output_paths helpers. All GNN outputs are
NEW artifacts; nothing here touches frozen segments or master_segments.

Artifacts written:
  03_analysis_data/gnn/segment_positions.csv     (Capability A)
  03_analysis_data/gnn/cue_motifs.csv            (Capability B)
  03_analysis_data/gnn/gnn_vs_llm_lift.csv       (Capability C)
  03_analysis_data/gnn/coupling_factors.csv      (Capability E)
  06_reports/report_gnn_emergent_motifs.txt      (Capability B)
  06_reports/report_gnn_coupling.txt             (Capability E)
"""

import os
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def write_segment_positions(positions: dict, output_dir: str) -> str:
    """Write per-segment VAAMR mixture + progression coordinate CSV.

    TODO(scaffold): DataFrame(positions).to_csv(gnn_data_dir/segment_positions.csv).
    """
    raise NotImplementedError("gnn_layer.reports.write_segment_positions: scaffold")


def write_cue_motifs(motif_stats: dict, exemplars: dict, output_dir: str) -> str:
    """Write the cue-motif table (id, influence, purity, exemplars).

    TODO(scaffold): flatten motif_stats+exemplars; to_csv(gnn_data_dir/cue_motifs.csv).
    """
    raise NotImplementedError("gnn_layer.reports.write_cue_motifs: scaffold")


def write_gnn_vs_llm_lift(comparison: "pd.DataFrame", output_dir: str) -> str:
    """Write the GNN-vs-LLM lift comparison CSV.

    TODO(scaffold): comparison.to_csv(gnn_data_dir/gnn_vs_llm_lift.csv).
    """
    raise NotImplementedError("gnn_layer.reports.write_gnn_vs_llm_lift: scaffold")


def write_emergent_motifs_report(flagged: List[int], motif_stats: dict,
                                 exemplars: dict, output_dir: str) -> str:
    """Write the human-readable emergent-influential-motifs report.

    TODO(scaffold): format flagged motifs with influence + exemplar cues to
    06_reports/report_gnn_emergent_motifs.txt.
    """
    raise NotImplementedError("gnn_layer.reports.write_emergent_motifs_report: scaffold")


def write_coupling_report(factors: dict, interpretation: dict, output_dir: str) -> List[str]:
    """Write coupling_factors.csv + report_gnn_coupling.txt.

    TODO(scaffold): emit latent factors, CF/IC interpretation, exemplar cues.
    """
    raise NotImplementedError("gnn_layer.reports.write_coupling_report: scaffold")
