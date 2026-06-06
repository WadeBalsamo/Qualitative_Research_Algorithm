"""
analysis/reports/irr_items.py
-----------------------------
Per-item, per-test-set IRR detail — the line-by-line content view.

For every worksheet item it lays out, side by side:
  * the segment text,
  * the human consensus + source + reasoning (notes), and each rater's code,
  * the LLM consensus + justification, and each LLM rater's code + justification,
  * the GNN prediction (held-out / out-of-fold when available, else distillation),
  * the LLM↔GNN consensus.

One file per test-set: ``04_validation/irr/irr_items_testset_<n>.txt``.
(The flat machine-readable equivalent is ``04_validation/irr/irr_item_detail.csv``.)
"""

import os
from typing import Dict, List

from process import output_paths as _paths


def _wrap(text: str, width: int = 92, indent: str = '      ') -> List[str]:
    import textwrap
    if not text:
        return [indent + '(none)']
    return [indent + ln for ln in textwrap.wrap(text, width=width)] or [indent + text]


def write_irr_item_details(results: dict, output_dir: str,
                           item_details: List[dict] = None) -> List[str]:
    """Write one detail file per test-set. Returns the list of paths written."""
    if item_details is None:
        item_details = results.get('_item_details', [])
    if not item_details:
        return []

    by_ws: Dict[int, List[dict]] = {}
    for it in item_details:
        by_ws.setdefault(it['worksheet_n'], []).append(it)

    gnn_axis = results.get('gnn_axis', 'distillation')
    out_dir = _paths.irr_validation_dir(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    paths = []
    for ws in sorted(by_ws):
        L: List[str] = []
        L.append("=" * 96)
        L.append(f"IRR ITEM DETAIL — TEST-SET {ws}")
        L.append("=" * 96)
        L.append("")
        L.append("Each item shows the segment text, then the three measurement substrates:")
        L.append("  HUMAN  — consensus of record, the reasoning behind it, and each rater's code")
        L.append("  LLM    — multi-run consensus + each model/rater's code and justification")
        L.append(f"  GNN    — graph prediction ({gnn_axis}; 'heldout' = out-of-fold, never trained")
        L.append("           on this segment's own LLM label — the honest validity axis)")
        L.append("  LLM↔GNN — whether the two independent machine substrates agree")
        L.append("")

        for it in sorted(by_ws[ws], key=lambda x: x['item_num']):
            h, llm, gnn = it['human'], it['llm'], it['gnn']
            L.append("-" * 96)
            L.append(f"ITEM {it['item_num']:>3}   segment_id={it['segment_id']}")
            L.append("-" * 96)
            L.append("  TEXT:")
            L.extend(_wrap(it['text']))
            L.append("")
            # Human
            sec = f" (2°: {h['consensus_secondary']})" if h['consensus_secondary'] else ""
            L.append(f"  HUMAN consensus : {h['consensus']}{sec}   [source: {h['source']}]")
            rater_str = '   '.join(
                f"{r}={v['primary']}" + (f"/{v['secondary']}" if v['secondary'] else '')
                for r, v in h['raters'].items()
            )
            L.append(f"    raters        : {rater_str or '(none)'}")
            if h['notes']:
                L.append("    reasoning     :")
                L.extend(_wrap(h['notes'], indent='        '))
            # LLM
            L.append(f"  LLM consensus   : {llm['consensus']}")
            if llm['justification']:
                L.append("    justification :")
                L.extend(_wrap(llm['justification'], indent='        '))
            for rv in llm['raters']:
                L.append(f"    · {rv['rater']}: {rv['label']}")
                if rv['justification']:
                    L.extend(_wrap(rv['justification'], indent='          '))
            # GNN
            conf = '' if gnn['confidence'] is None else f"  (conf {gnn['confidence']:.2f})"
            src = f"  [{gnn['source']}]" if gnn['source'] else ""
            L.append(f"  GNN prediction  : {gnn['prediction']}{conf}{src}")
            if gnn['distillation'] and gnn['source'] == 'heldout':
                L.append(f"    (distillation overlay for reference: {gnn['distillation']})")
            # LLM <-> GNN consensus
            lg = it['llm_gnn_consensus']
            if lg['agree'] is None:
                L.append(f"  LLM↔GNN         : {lg.get('note', 'n/a')}")
            elif lg['agree']:
                L.append(f"  LLM↔GNN         : AGREE → {lg['label']}")
            else:
                L.append(f"  LLM↔GNN         : DISAGREE")
            L.append("")

        path = os.path.join(out_dir, f'irr_items_testset_{ws}.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(L))
        paths.append(path)
    return paths
