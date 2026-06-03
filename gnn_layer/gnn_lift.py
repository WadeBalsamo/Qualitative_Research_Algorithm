"""
gnn_layer/gnn_lift.py
---------------------
Capability C — GNN as an independent (non-LLM) measurement substrate (SCAFFOLD).

Methodology Sec 5.2 names the central construct-validity threat: VAAMR<->VCE and
PURER<->transition lift are both LLM-on-LLM and may reflect shared training data
rather than phenomenological structure. A GNN trained on Qwen3 GEOMETRY (and, for a
clean claim, a self-supervised variant with NO LLM labels) is an independent rater.

This module recomputes the lift tables from GNN-derived assignments using the SAME
formula as analysis.purer_analysis.compute_purer_transition_influence
( lift = P(to | x) / P(to) ) and presents them side-by-side with the LLM-derived
tables. Convergence GNN<->LLM is stronger evidence than LLM<->LLM and complements the
planned permutation control (methodology Sec 8.2).
"""

from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from .config import GnnLayerConfig


def gnn_vaamr_vce_lift(df_all, gnn_assignments: dict) -> "pd.DataFrame":
    """(VAAMR stage x VCE code) lift computed from GNN assignments.

    TODO(scaffold): build co-occurrence from gnn_assignments; apply lift formula.
    """
    raise NotImplementedError("gnn_layer.gnn_lift.gnn_vaamr_vce_lift: scaffold")


def gnn_purer_transition_lift(df_all, gnn_assignments: dict) -> "pd.DataFrame":
    """(PURER move x VAAMR transition) lift from GNN assignments.

    TODO(scaffold): reuse the conditional-lift logic over GNN-derived labels.
    """
    raise NotImplementedError("gnn_layer.gnn_lift.gnn_purer_transition_lift: scaffold")


def gnn_purer_microskill_lift(df_all, gnn_assignments: dict) -> "pd.DataFrame":
    """(PURER move x microskill) lift from GNN assignments — the therapist twin.

    TODO(scaffold): co-occurrence of PURER and microskill labels; lift formula.
    """
    raise NotImplementedError("gnn_layer.gnn_lift.gnn_purer_microskill_lift: scaffold")


def compare_gnn_vs_llm(gnn_table: "pd.DataFrame", llm_table: "pd.DataFrame") -> "pd.DataFrame":
    """Join GNN-derived and LLM-derived lift tables for convergence reporting.

    TODO(scaffold): align on (row,col) keys; emit both values + agreement flags.
    """
    raise NotImplementedError("gnn_layer.gnn_lift.compare_gnn_vs_llm: scaffold")
