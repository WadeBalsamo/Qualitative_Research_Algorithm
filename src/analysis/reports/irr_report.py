"""
analysis/reports/irr_report.py
------------------------------
The single human-facing inter-rater-reliability report (``06_reports/06b_irr_report.txt``).

Consumes the results dict from ``analysis.irr_analysis.run_irr_analysis`` and renders:
  * a headline κ table (Human↔Human per test-set, Human↔LLM, Human↔GNN),
  * Landis–Koch interpretation bands,
  * per-family detail (pairwise rater κ, per-class precision/recall, confusion matrices),
  * a ranked discrepancy walkthrough.

Mirrors the ``L = []`` builder + single-write pattern used across ``analysis.reports``.
"""

import os
from typing import List, Optional

from process import output_paths as _paths


# Landis & Koch (1977) agreement bands.
def _band(k: Optional[float]) -> str:
    if k is None:
        return 'n/a'
    if k < 0.0:
        return 'poor'
    if k <= 0.20:
        return 'slight'
    if k <= 0.40:
        return 'fair'
    if k <= 0.60:
        return 'moderate'
    if k <= 0.80:
        return 'substantial'
    return 'almost perfect'


def _kfmt(k: Optional[float]) -> str:
    return 'n/a' if k is None else f"{k:+.3f}"


def _pfmt(p: Optional[float]) -> str:
    return 'n/a' if p is None else f"{p * 100:5.1f}%"


def generate_irr_report(results: dict, output_dir: str,
                        discrepancies: Optional[List[dict]] = None) -> str:
    """Write the IRR report; returns its path."""
    if discrepancies is None:
        discrepancies = results.get('_discrepancies', [])

    L: List[str] = []
    L.append("=" * 78)
    L.append("INTER-RATER RELIABILITY (VAAMR)")
    L.append("=" * 78)
    L.append("")
    L.append(f"Generated: {results.get('generated_at', '')}")
    L.append("")
    _drift = results.get('testset_drift') or []
    if _drift:
        L.append("!" * 78)
        L.append("⚠ TEST-SET CONTENT DRIFT — these results may NOT reflect what humans coded")
        L.append("!" * 78)
        for d in _drift[:12]:
            if d.get('kind') == 'unresolved':
                L.append(f"  test-set {d['worksheet_n']} item {d['item_num']}: segment unresolved "
                         f"({d.get('segment_id')})")
            else:
                L.append(f"  test-set {d['worksheet_n']} item {d['item_num']} ({d.get('segment_id')}): "
                         f"SHA {(d.get('frozen_sha') or '')[:10]} → {(d.get('live_sha') or '')[:10]}")
        if len(_drift) > 12:
            L.append(f"  … and {len(_drift) - 12} more")
        L.append("  Re-import the affected worksheet(s) or re-freeze the test-set before trusting "
                 "the κ below.")
        L.append("")
    L.append("Three comparison families:")
    L.append("  1. Human ↔ Human   — agreement BETWEEN researchers, within each test-set.")
    L.append("  2. Human ↔ LLM     — human consensus vs the project's CURRENT VAAMR LLM labels.")
    L.append("  3. Human ↔ GNN     — human consensus vs the project's CURRENT GNN labels.")
    L.append("Machine labels are pulled live, so this always reflects the latest classification.")
    L.append("Unresolved human items (no consensus) are excluded from Human↔Machine but still")
    L.append("counted in Human↔Human. 'No code' is treated as a 6th category (an abstain ballot).")
    L.append("Item counts below: 'excluded' = no segment resolved for the item; 'deferred' = the")
    L.append("model produced no comparable label (abstain/missing). Both are shown, never hidden.")
    L.append("Statistics (proven libraries): Cohen's κ = scikit-learn; Fleiss' κ = statsmodels")
    L.append("(complete-case items); Krippendorff's α = krippendorff package (tolerates missing")
    L.append("ballots / unequal raters — the headline multi-rater statistic here).")
    L.append("")
    L.append("Landis–Koch bands: <0 poor · ≤.20 slight · ≤.40 fair · ≤.60 moderate ·")
    L.append("                   ≤.80 substantial · >.80 almost perfect")
    L.append("")

    # ---------------- Headline table ----------------
    L.append("-" * 78)
    L.append("HEADLINE")
    L.append("-" * 78)
    L.append(f"  {'Comparison':<36}{'n':>6}  {'stat':>8}  {'agree':>7}  band")
    hh = results.get('human_human', {})
    for ws in sorted(hh, key=lambda x: int(x)):
        block = hh[ws]
        pri = block.get('primary')
        if pri:
            k = pri.get('krippendorff_alpha')
            label = f"H↔H test-set {ws} (Krippendorff α)"
            L.append(f"  {label:<36}{pri.get('n_items_scored',0):>6}  "
                     f"{_kfmt(k):>8}  {_pfmt(pri.get('percent_agreement_pairwise')):>7}  {_band(k)}")
    for name, key in (("H↔LLM (consensus, Cohen κ)", 'human_vs_llm'),
                      ("H↔GNN (consensus, Cohen κ)", 'human_vs_gnn')):
        b = results.get(key, {})
        if b.get('n'):
            k = b.get('cohen_kappa')
            L.append(f"  {name:<36}{b['n']:>6}  {_kfmt(k):>8}  "
                     f"{_pfmt(b.get('percent_agreement')):>7}  {_band(k)}")
        else:
            note = b.get('note', 'no usable items')
            L.append(f"  {name:<36}{'—':>6}  ({note})")
    L.append("")

    # ---------------- Family 1 detail ----------------
    L.append("=" * 78)
    L.append("1. HUMAN ↔ HUMAN")
    L.append("=" * 78)
    for ws in sorted(hh, key=lambda x: int(x)):
        block = hh[ws]
        L.append("")
        L.append(f"Test-set {ws}  (raters: {', '.join(block.get('raters', []))}; "
                 f"{block.get('n_items','?')} items)")
        for field in ('primary', 'secondary'):
            sub = block.get(field)
            if not sub:
                if field == 'primary':
                    L.append("  primary: (insufficient multi-rater data)")
                continue
            L.append(f"  {field}:")
            ka = sub.get('krippendorff_alpha')
            fk = sub.get('fleiss_kappa')
            L.append(f"    Krippendorff α = {_kfmt(ka)} ({_band(ka)})"
                     f"   Fleiss κ = {_kfmt(fk)} (complete-case n={sub.get('fleiss_n_complete',0)})")
            L.append(f"    agreement: unanimous {_pfmt(sub.get('percent_agreement_unanimous'))}, "
                     f"pairwise {_pfmt(sub.get('percent_agreement_pairwise'))} "
                     f"(n={sub.get('n_items_scored',0)})")
            for pr in sub.get('pairwise', []):
                L.append(f"      {pr['rater_a']:>7} ↔ {pr['rater_b']:<7} "
                         f"n={pr['n']:<3} κ={_kfmt(pr['cohen_kappa'])} "
                         f"({_band(pr['cohen_kappa'])})  agree={_pfmt(pr['percent_agreement'])}")
    L.append("")

    # ---------------- Family 2: LLM ----------------
    b = results.get('human_vs_llm', {})
    L.append("=" * 78)
    L.append("2. HUMAN-CONSENSUS ↔ LLM")
    L.append("=" * 78)
    if not b.get('n'):
        L.append(f"  ({b.get('note', 'no usable items')})")
        L.append("")
    else:
        L.append(f"  Overall: n={b['n']}  κ={_kfmt(b.get('cohen_kappa'))} "
                 f"({_band(b.get('cohen_kappa'))})  agreement={_pfmt(b.get('percent_agreement'))}")
        if b.get('n_excluded_no_machine'):
            L.append(f"  excluded — segment unresolved / missing: {b['n_excluded_no_machine']}")
        if b.get('n_deferred'):
            L.append(f"  deferred — LLM produced no label (abstain/missing): {b['n_deferred']}")
        per_ws = b.get('per_worksheet', {})
        if per_ws:
            L.append("  per test-set:")
            for ws, sub in sorted(per_ws.items(), key=lambda x: int(x[0])):
                L.append(f"    test-set {ws}: n={sub['n']}  κ={_kfmt(sub.get('cohen_kappa'))} "
                         f"agree={_pfmt(sub.get('percent_agreement'))}")
        per_rater = b.get('per_llm_rater') or {}
        if per_rater:
            L.append("")
            L.append("  per LLM rater/model (human consensus vs that single rater's ballot):")
            for rid, sub in sorted(per_rater.items(),
                                   key=lambda x: (x[1].get('cohen_kappa') is None,
                                                  -(x[1].get('cohen_kappa') or 0))):
                L.append(f"    {rid:<32} n={sub['n']:<4} κ={_kfmt(sub.get('cohen_kappa'))} "
                         f"({_band(sub.get('cohen_kappa'))})  agree={_pfmt(sub.get('percent_agreement'))}")
        L.extend(_confusion_lines(b.get('confusion', {})))
        L.append("")

    # ---------------- Family 3: GNN (both axes + gate) ----------------
    L.extend(_gnn_section(results))

    # ---------------- Discrepancy walkthrough ----------------
    L.append("=" * 78)
    L.append(f"DISCREPANCIES  ({len(discrepancies)} items where consensus ≠ LLM and/or ≠ GNN)")
    L.append("=" * 78)
    if not discrepancies:
        L.append("  (none, or no machine labels available)")
    for row in discrepancies:
        flags = []
        if row.get('disagrees_llm'):
            flags.append('LLM')
        if row.get('disagrees_gnn'):
            flags.append('GNN')
        L.append("")
        L.append(f"  [test-set {row['worksheet_n']} item {row['item_num']}]  "
                 f"≠ {'/'.join(flags)}")
        L.append(f"    human consensus : {row['human_consensus']}  "
                 f"(source: {row['consensus_source']})")
        L.append(f"    raters          : {row['rater_codes']}")
        L.append(f"    LLM             : {row['llm_label']}"
                 f"{_conf_suffix(row.get('llm_confidence'))}")
        L.append(f"    GNN             : {row['gnn_label']}"
                 f"{_conf_suffix(row.get('gnn_confidence'))}")
        text = row.get('text', '')
        if text:
            L.append(f"    text            : {_truncate(text, 200)}")
    L.append("")

    L.append("-" * 78)
    L.append("Data: 04_validation/irr/  (irr_results.json, irr_pairwise.csv,")
    L.append("      irr_discrepancies.csv, irr_item_detail.csv, *.png)")
    L.append("Per-item content + reasonings (one file per test-set):")
    L.append("      04_validation/irr/irr_items_testset_<n>.txt")

    path = _paths.reports_irr_path(output_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path


def _confusion_lines(conf: dict) -> List[str]:
    if not conf:
        return []
    L = ["", "  Per-class (human consensus as reference):",
         f"    {'stage':<14}{'precision':>10}{'recall':>9}{'support':>9}"]
    for name, pc in conf.get('per_class', {}).items():
        L.append(f"    {name:<14}{pc['precision']:>10.2f}{pc['recall']:>9.2f}{pc['support']:>9}")
    L.append("")
    L.append("  Confusion matrix (rows = human consensus, cols = machine):")
    names = conf.get('label_names', [])
    L.append("      " + "".join(f"{n[:6]:>8}" for n in names))
    for i, rowname in enumerate(names):
        cells = "".join(f"{c:>8}" for c in conf['matrix'][i])
        L.append(f"    {rowname[:6]:<6}{cells}")
    return L


def _axis_line(label: str, blk: dict) -> str:
    if not blk or not blk.get('n'):
        return f"    {label:<34} (no overlap)"
    k = blk.get('cohen_kappa')
    return (f"    {label:<34} n={blk['n']:<4} κ={_kfmt(k)} ({_band(k)})  "
            f"agree={_pfmt(blk.get('percent_agreement'))}")


def _gnn_section(results: dict) -> List[str]:
    """Family 3 — BOTH GNN axes (held-out + distillation), each vs human and vs LLM,
    with the reliability-gate framing and reference bands."""
    g = results.get('human_vs_gnn', {})
    L = ["=" * 78, "3. HUMAN/LLM ↔ GNN  (held-out validity + distillation default)", "=" * 78]
    if not (g.get('heldout') or g.get('distillation')):
        L.append(f"  ({g.get('note', 'no usable items')})")
        L.append("")
        return L

    L.append("Two GNN quantities answer two questions — each labeled with its circularity:")
    L.append("  • HELD-OUT (out-of-fold): the segment's own LLM label was masked in training,")
    L.append("    so these are HONEST. vs human = independent construct validity; vs LLM = the")
    L.append("    reliability gate (reproducing the teacher on unseen segments).")
    L.append("  • DISTILLATION (in-sample consensus overlay): trained on every label and used")
    L.append("    as the operational DEFAULT. vs LLM is distillation FIDELITY (circular — high")
    L.append("    by construction, NOT validity); vs human is optimistic (teacher leakage).")
    L.append("")

    held = g.get('heldout') or {}
    dist = g.get('distillation') or {}
    L.append("  HELD-OUT GNN  (honest):")
    if not held.get('available'):
        L.append("    (no held-out predictions — run `qra gnn train` to generate them)")
    else:
        L.append(_axis_line("held-out ↔ human consensus", held.get('vs_human')))
        L.append(_axis_line("held-out ↔ LLM (gate)", held.get('vs_llm')))
        if (held.get('vs_human') or {}).get('n_deferred'):
            L.append(f"      (deferred, no held-out pred: {held['vs_human']['n_deferred']})")
    L.append("")
    L.append("  DISTILLATION GNN  (default mechanism; in-sample):")
    L.append(_axis_line("distillation ↔ human", dist.get('vs_human')))
    L.append(_axis_line("distillation ↔ LLM (fidelity)", dist.get('vs_llm')))
    L.append("")

    # Reference bands + gate verdict.
    hh = results.get('human_human', {})
    ref_alphas = [v['primary']['krippendorff_alpha'] for v in hh.values()
                  if v.get('primary') and v['primary'].get('krippendorff_alpha') is not None]
    llm = results.get('human_vs_llm', {})
    L.append("  Reference targets the GNN must approach before LLM-free scaling:")
    if ref_alphas:
        L.append(f"    human↔human (Krippendorff α range): "
                 f"{min(ref_alphas):+.3f} … {max(ref_alphas):+.3f}")
    L.append(f"    LLM↔human (Cohen κ): {_kfmt(llm.get('cohen_kappa'))}")
    held_h = (held.get('vs_human') or {}).get('cohen_kappa')
    if held.get('available') and held_h is not None and llm.get('cohen_kappa') is not None:
        verdict = ("GNN comparable to the LLM on the human axis — candidate for LLM-free "
                   "scaling (confirm with `qra gnn status`)." if held_h >= llm['cohen_kappa'] - 0.05
                   else "GNN below the LLM on the human axis — keep LLM consensus as default.")
        L.append(f"  Gate read: {verdict}")
    L.append("")
    # Confusion matrix for the operative axis (held-out when available).
    L.append(f"  Confusion — operative GNN axis ({g.get('operative_axis', 'distillation')}):")
    L.extend(_confusion_lines(g.get('confusion', {})))
    L.append("")
    return L


def _conf_suffix(c) -> str:
    return '' if c is None else f"  (conf {c:.2f})"


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + '…'
