"""
analysis/irr_join.py
--------------------
Close the "human-subset integration gap" (design_decisions.md §4, A0-pre).

``master_segments.csv`` ships ``in_human_coded_subset`` / ``human_label`` columns
but they are EMPTY — the human consensus codes actually live in ``qra.db``
(``irr_human_codes``, written by :mod:`process.irr_import`). The GNN reliability
gate's human axis (:func:`gnn_layer.validation.evaluate_crossval._human_axis`)
reads those two DataFrame columns, so until they are populated the human axis is
dark.

This module reads the human CONSENSUS codes (reusing
:func:`process.irr_import.read_human_codes` + the usable-consensus filter
:func:`analysis.irr_analysis._consensus_rows`), resolves each to its
``segment_id``, and writes the two columns onto the assembled DataFrame.

Encoding matches the rest of the IRR stack (``process.irr_import``):
``human_label`` is the consensus PRIMARY VAAMR theme_id (0–4), with ``"No code"``
encoded as ``-1`` (``irr_import.ABSTAIN_CODE``). Only *usable* consensus rows
(``source != 'unresolved'``, a resolved ``segment_id``, a non-null primary) are
marked; every other row is left non-human. The number is therefore directly
comparable to the GNN↔human κ reported in ``06_reports/01_reliability/irr_report.txt``.
"""

from typing import Dict

from process import irr_import
from analysis.irr_analysis import _consensus_rows


def human_consensus_map(output_dir: str) -> Dict[str, int]:
    """``{segment_id: consensus primary int}`` for usable human consensus rows.

    "No code" stays encoded as ``-1`` (``irr_import.ABSTAIN_CODE``). Unresolved
    consensus, rows without a resolved ``segment_id``, and rows with a null
    primary are skipped. Returns ``{}`` when the project has no imported human
    codes (so callers can treat it as a no-op guard).
    """
    codes = irr_import.read_human_codes(output_dir)
    if not codes:
        return {}
    out: Dict[str, int] = {}
    for c in _consensus_rows(codes):
        sid = c.get('segment_id')
        prim = c.get('primary')
        if sid is None or prim is None:
            continue
        out[str(sid)] = int(prim)
    return out


def populate_human_columns(df_all, output_dir: str):
    """Set ``in_human_coded_subset`` / ``human_label`` on ``df_all`` from the
    human consensus codes in ``qra.db``.

    For every df row whose ``segment_id`` carries a usable consensus, sets
    ``in_human_coded_subset=True`` and ``human_label`` to the consensus primary
    (``"No code"`` → ``-1``). Non-human rows are left non-human
    (``in_human_coded_subset=False``, ``human_label`` = <NA>). The two columns are
    always ensured to exist even when there are no human codes, so the downstream
    ``bool(row.get('in_human_coded_subset'))`` / ``int(row.get('human_label'))``
    reads in :mod:`gnn_layer.validation` and :mod:`gnn_layer.soft_labels` behave.

    Guarded + idempotent: a no-op (df returned unchanged, columns ensured) when no
    human codes are present. Returns ``df_all`` (mutated in place and returned).
    """
    if df_all is None or 'segment_id' not in getattr(df_all, 'columns', []):
        return df_all

    cmap = human_consensus_map(output_dir)
    sid_str = df_all['segment_id'].astype(str)
    # ``map`` yields the consensus primary where present, NaN elsewhere — this is the
    # single source of truth for both columns, so they can never disagree.
    mapped = sid_str.map(cmap) if cmap else sid_str.map(lambda _s: None)
    df_all['in_human_coded_subset'] = mapped.notna()
    df_all['human_label'] = mapped.astype('Int64')
    return df_all
