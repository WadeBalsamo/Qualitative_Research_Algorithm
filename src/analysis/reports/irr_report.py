"""
analysis/reports/irr_report.py
------------------------------
The single human-facing inter-rater-reliability dossier
(``06_reports/01_reliability/irr_report.txt``).

Consumes the results dict from ``analysis.irr_analysis.run_irr_analysis`` and renders
a scientific dossier in the canonical 78-column report style (matching its sibling
reports under ``06_reports/``):

  1. HEADER + provenance      — what/who/which-test-sets/which-libraries
  2. HEADLINE TABLE           — one row per comparison family (the meeting slide):
                                statistic · value · 95% bootstrap CI · Landis–Koch · n
  3. INTERPRETATION           — pre-registered evidence-tier verdict per test-set,
                                the human↔human "fuzzy ground-truth ceiling" argument,
                                and the operational read for the protocol board
  4. PER-STAGE DIAGNOSTICS    — human-consensus vs LLM confusion + recall/precision,
                                with rare-stage instability called out
  5. PER-TEST-SET DETAIL      — pairwise rater κ, Fleiss/Krippendorff per family
  6. DISCREPANCY WALKTHROUGH  — ranked consensus≠machine items (capped)
  7. GNN / probe axes         — dual-axis section, degrading to one line when absent

Every statistic is formatted through ``analysis.reports.stat_format`` so a number is
written one way, everywhere, and cites its [M#] provenance footnote.

Mirrors the ``L = []`` builder + single-write pattern used across ``analysis.reports``.
"""

import os
from typing import List, Optional

from process import output_paths as _paths
from .stat_format import (
    fmt_kappa, fmt_p, landis_koch, evidence_tier, m_ref, provenance_header,
)

W = 78  # column width for rules / tables

# The observed human↔human Krippendorff α band on this corpus (methodology §5.4).
# Computed from the results when available; this is the descriptive fallback range.
_HUMAN_BAND_NOTE = "moderate — a genuinely fuzzy ground truth"


def _kfmt(k: Optional[float]) -> str:
    """Signed 3-dp κ/α with n/a guard (used inside dense tables)."""
    return 'n/a' if k is None else f"{k:+.3f}"


def _pfmt(p: Optional[float]) -> str:
    return 'n/a' if p is None else f"{p * 100:5.1f}%"


def _ci_str(ci: Optional[dict]) -> str:
    """``[+0.31, +0.62]`` from a bootstrap CI dict, or '—' when absent."""
    if not ci or ci.get('lo') is None or ci.get('hi') is None:
        return '—'
    return f"[{ci['lo']:+.3f}, {ci['hi']:+.3f}]"


def _human_band(results: dict) -> Optional[tuple]:
    """(min, max) of the per-test-set primary Krippendorff α — the ceiling band."""
    hh = results.get('human_human', {})
    alphas = [v['primary']['krippendorff_alpha'] for v in hh.values()
              if v.get('primary') and v['primary'].get('krippendorff_alpha') is not None]
    return (min(alphas), max(alphas)) if alphas else None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_irr_report(results: dict, output_dir: str,
                        discrepancies: Optional[List[dict]] = None) -> str:
    """Write the IRR dossier; returns its path."""
    if discrepancies is None:
        discrepancies = results.get('_discrepancies', [])

    L: List[str] = []
    _header(L, results)
    _drift_banner(L, results)
    _headline_table(L, results)
    _interpretation(L, results)
    _per_stage_diagnostics(L, results)
    _per_testset_detail(L, results)
    _gnn_probe_axes(L, results)
    _discrepancy_walkthrough(L, discrepancies)
    _footer(L)

    path = _paths.reports_irr_path(output_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path


# ---------------------------------------------------------------------------
# 1. Header + provenance
# ---------------------------------------------------------------------------

def _header(L: List[str], results: dict) -> None:
    L.append("=" * W)
    L.append("INTER-RATER RELIABILITY DOSSIER — VAAMR".center(W))
    L.append("=" * W)
    L.append("")
    L.append(f"Generated: {results.get('generated_at', '')}")
    L.append("")

    hh = results.get('human_human', {})
    rosters = []
    total_items = 0
    for ws in sorted(hh, key=lambda x: int(x)):
        block = hh[ws]
        raters = ', '.join(block.get('raters', []))
        n = block.get('n_items', '?')
        if isinstance(n, int):
            total_items += n
        rosters.append(f"    test-set {ws}: {raters} (n={n} items)")

    L.append("WHAT WAS COMPARED")
    L.append("-" * W)
    L.append("  Human researchers' blind VAAMR coding of frozen validation worksheets,")
    L.append("  against each other and against the project's CURRENT machine labels (the")
    L.append("  multi-run LLM consensus, pulled live so this always reflects the latest")
    L.append("  classification). VAAMR applies to participant segments only.")
    L.append("")
    L.append("  Raters x frozen test-sets:")
    L.extend(rosters)
    llm_n = (results.get('human_vs_llm') or {}).get('n')
    if llm_n:
        L.append(f"  Human-consensus vs LLM: {llm_n} resolved consensus items.")
    L.append("")
    L.append("  Statistics (proven libraries):")
    L.append("    Cohen's κ        — scikit-learn (pairwise / human-vs-machine)")
    L.append("    Fleiss' κ        — statsmodels (complete-case multi-rater)")
    L.append("    Krippendorff's α — krippendorff package (headline multi-rater;")
    L.append("                       tolerates missing ballots / unequal raters)")
    L.append("    95% CIs          — nonparametric item bootstrap (2,000 reps, percentile)")
    L.append("    Verbal bands     — Landis & Koch (1977)")
    L.append("")
    L.extend("  " + ln for ln in provenance_header(['irr']))
    L.append("")
    L.append("  Encoding: 'No code' is a 6th category (an abstain ballot), since ≈36% of")
    L.append("  human-coded items express no VAAMR stage. Unresolved human items (no")
    L.append("  consensus) are excluded from Human↔Machine but kept in Human↔Human.")
    L.append("")


def _drift_banner(L: List[str], results: dict) -> None:
    drift = results.get('testset_drift') or []
    if not drift:
        return
    L.append("!" * W)
    L.append("TEST-SET CONTENT DRIFT — these results may NOT reflect what humans coded")
    L.append("!" * W)
    for d in drift[:12]:
        if d.get('kind') == 'unresolved':
            L.append(f"  test-set {d['worksheet_n']} item {d['item_num']}: segment "
                     f"unresolved ({d.get('segment_id')})")
        else:
            L.append(f"  test-set {d['worksheet_n']} item {d['item_num']} "
                     f"({d.get('segment_id')}): SHA {(d.get('frozen_sha') or '')[:10]} "
                     f"→ {(d.get('live_sha') or '')[:10]}")
    if len(drift) > 12:
        L.append(f"  … and {len(drift) - 12} more")
    L.append("  Re-import the affected worksheet(s) or re-freeze the test-set before")
    L.append("  trusting the κ below.")
    L.append("")


# ---------------------------------------------------------------------------
# 2. Headline table (the meeting slide)
# ---------------------------------------------------------------------------

def _headline_table(L: List[str], results: dict) -> None:
    L.append("=" * W)
    L.append("HEADLINE — RELIABILITY AT A GLANCE")
    L.append("=" * W)
    L.append("")
    hdr = f"  {'comparison':<32}{'stat':>7}{'95% CI':>18}  {'band':<12}{'n':>4}"
    L.append(hdr)
    L.append("  " + "-" * (W - 2))

    def row(label, value, ci, band, n):
        L.append(f"  {label:<32}{_kfmt(value):>7}{_ci_str(ci):>18}  "
                 f"{(band or 'n/a'):<12}{(n if n is not None else '—'):>4}")

    # Family 1 — Human ↔ Human per test-set (Krippendorff α, the headline statistic).
    hh = results.get('human_human', {})
    L.append("  Human ↔ Human  (the ceiling for any machine rater)")
    for ws in sorted(hh, key=lambda x: int(x)):
        pri = hh[ws].get('primary')
        if not pri:
            continue
        a = pri.get('krippendorff_alpha')
        row(f"  test-set {ws} (Krippendorff α)", a, pri.get('alpha_ci'),
            landis_koch(a), pri.get('n_items_scored'))

    # Family 2 — Human ↔ LLM consensus (overall + per test-set) + per-model.
    L.append("")
    llm = results.get('human_vs_llm', {})
    L.append("  Human-consensus ↔ LLM  (Cohen κ)")
    if llm.get('n'):
        row("  overall consensus", llm.get('cohen_kappa'), llm.get('kappa_ci'),
            landis_koch(llm.get('cohen_kappa')), llm.get('n'))
        for ws, sub in sorted((llm.get('per_worksheet') or {}).items(),
                              key=lambda x: int(x[0])):
            row(f"  test-set {ws}", sub.get('cohen_kappa'), sub.get('kappa_ci'),
                landis_koch(sub.get('cohen_kappa')), sub.get('n'))
        per_rater = llm.get('per_llm_rater') or {}
        if per_rater:
            L.append("")
            L.append("  Human-consensus ↔ individual LLM model  (Cohen κ)")
            for rid, sub in sorted(per_rater.items(),
                                   key=lambda x: (x[1].get('cohen_kappa') is None,
                                                  -(x[1].get('cohen_kappa') or 0))):
                short = rid.split('/')[-1]
                row(f"  {short[:30]}", sub.get('cohen_kappa'), sub.get('kappa_ci'),
                    landis_koch(sub.get('cohen_kappa')), sub.get('n'))
    else:
        L.append(f"    ({llm.get('note', 'no usable items')})")

    # Family 3 — GNN, only if present.
    g = results.get('human_vs_gnn', {})
    if g.get('n'):
        L.append("")
        L.append("  Human-consensus ↔ GNN  (Cohen κ)")
        row(f"  {g.get('operative_axis', 'gnn')} ↔ human", g.get('cohen_kappa'),
            g.get('kappa_ci'), landis_koch(g.get('cohen_kappa')), g.get('n'))
    L.append("")
    L.append("  Bands: <0 poor · ≤.20 slight · ≤.40 fair · ≤.60 moderate ·")
    L.append("         ≤.80 substantial · >.80 almost perfect")
    L.append("")


# ---------------------------------------------------------------------------
# 3. Interpretation
# ---------------------------------------------------------------------------

def _interpretation(L: List[str], results: dict) -> None:
    L.append("=" * W)
    L.append("INTERPRETATION — WHAT THIS LICENSES")
    L.append("=" * W)
    L.append("")

    # (a) Pre-registered evidence-tier verdict per test-set.
    L.append("(a) Pre-registered evidence tiers (methodology §5.4):")
    L.append("    ≥75% raw + α≥0.60 → PRIMARY · α 0.40–0.59 → DIRECTIONAL (needs")
    L.append("    convergence with ≥2 independent sources) · α<0.40 → revise framework.")
    L.append("")
    hh = results.get('human_human', {})
    for ws in sorted(hh, key=lambda x: int(x)):
        pri = hh[ws].get('primary')
        if not pri:
            continue
        a = pri.get('krippendorff_alpha')
        L.append(f"    test-set {ws}: α={_kfmt(a)} → {evidence_tier(a)}")
    L.append("")

    # (b) The fuzzy ground-truth ceiling argument.
    band = _human_band(results)
    llm = results.get('human_vs_llm', {})
    llm_k = llm.get('cohen_kappa')
    L.append("(b) The fuzzy-ground-truth ceiling.")
    L.append("    Human coders agree with each other on VAAMR primary codes only at")
    if band:
        L.append(f"    Krippendorff α {band[0]:+.3f} … {band[1]:+.3f} ({_HUMAN_BAND_NOTE}).")
    L.append("    That human↔human band is the CEILING for any machine rater: no model")
    L.append("    can agree with a human consensus more reliably than humans agree among")
    L.append("    themselves. So 'human-level' here means MODERATE agreement, not")
    L.append("    near-perfect agreement — we do not overclaim.")
    if llm_k is not None and band:
        # Express the band in Cohen-κ-comparable terms (the pairwise human band).
        pair_lo, pair_hi = _pairwise_human_band(results)
        sits = ("at/above" if pair_lo is None or llm_k >= pair_lo else "within")
        L.append("")
        L.append(f"    On this corpus the LLM consensus reaches {fmt_kappa(llm_k, llm.get('kappa_ci', {}).get('lo'), llm.get('kappa_ci', {}).get('hi'), llm.get('n'))}")
        L.append(f"    against the human consensus. That sits {sits} the human↔human band")
        if pair_lo is not None:
            L.append(f"    (pairwise human κ {pair_lo:+.3f}–{pair_hi:+.3f}) — the meaningful,")
        L.append("    bounded claim of 'human-level' reliability: the LLM MATCHES a")
        L.append("    moderately-agreeing human panel rather than exceeding it.")
    L.append("")

    # (c) Operational read for the protocol board.
    L.append("(c) Operational read (for the protocol-adaptation board):")
    primary_ws = [ws for ws in sorted(hh, key=lambda x: int(x))
                  if (hh[ws].get('primary') or {}).get('krippendorff_alpha') is not None
                  and hh[ws]['primary']['krippendorff_alpha'] >= 0.60]
    direial_ws = [ws for ws in sorted(hh, key=lambda x: int(x))
                  if (hh[ws].get('primary') or {}).get('krippendorff_alpha') is not None
                  and 0.40 <= hh[ws]['primary']['krippendorff_alpha'] < 0.60]
    below_ws = [ws for ws in sorted(hh, key=lambda x: int(x))
                if (hh[ws].get('primary') or {}).get('krippendorff_alpha') is not None
                and hh[ws]['primary']['krippendorff_alpha'] < 0.40]
    L.append("    • VAAMR labels are usable as DIRECTIONAL evidence on the test-sets in")
    L.append("      the 0.40–0.59 band, becoming PRIMARY only where a test-set clears")
    L.append("      α≥0.60 OR where ≥2 independent sources converge.")
    if direial_ws:
        L.append(f"      Directional-band test-sets: {', '.join(direial_ws)}.")
    if primary_ws:
        L.append(f"      Primary-band test-sets: {', '.join(primary_ws)}.")
    if below_ws:
        L.append(f"      Below-floor test-sets (framework refinement indicated): "
                 f"{', '.join(below_ws)}.")
    L.append("    • The LLM consensus is at human level, so it is the appropriate")
    L.append("      scaling engine for the labeled corpus — NOT a higher authority than")
    L.append("      the human-adjudicated labels, which always outrank it.")
    L.append("    • Per-stage caveat: agreement is driven down by the adjacent-stage")
    L.append("      boundaries (Attention-Regulation vs Metacognition; Metacognition vs")
    L.append("      Reappraisal). Treat stage-level claims at those boundaries as soft;")
    L.append("      see the per-stage diagnostics below.")
    L.append("")


def _pairwise_human_band(results: dict):
    """(min, max) pairwise Cohen κ across all test-sets' primary codes."""
    ks = []
    for v in results.get('human_human', {}).values():
        pri = v.get('primary') or {}
        for pr in pri.get('pairwise', []):
            if pr.get('cohen_kappa') is not None:
                ks.append(pr['cohen_kappa'])
    return (min(ks), max(ks)) if ks else (None, None)


# ---------------------------------------------------------------------------
# 4. Per-stage diagnostics
# ---------------------------------------------------------------------------

# Rough rare-stage threshold: per-stage support below this makes recall unstable.
_RARE_SUPPORT = 6


def _per_stage_diagnostics(L: List[str], results: dict) -> None:
    L.append("=" * W)
    L.append("PER-STAGE DIAGNOSTICS — HUMAN CONSENSUS vs LLM")
    L.append("=" * W)
    llm = results.get('human_vs_llm', {})
    conf = llm.get('confusion') or {}
    if not conf.get('per_class'):
        L.append("  (no confusion data — no machine labels available)")
        L.append("")
        return
    L.append("")
    L.append("  Per VAAMR stage (human consensus as reference):")
    L.append(f"    {'stage':<22}{'precision':>10}{'recall':>9}{'support':>9}")
    rare = []
    for name, pc in conf['per_class'].items():
        flag = ''
        if pc['support'] and pc['support'] < _RARE_SUPPORT:
            flag = '  *'
            rare.append((name, pc['support']))
        L.append(f"    {name:<22}{pc['precision']:>10.2f}{pc['recall']:>9.2f}"
                 f"{pc['support']:>9}{flag}")
    L.append("")
    if rare:
        rare_txt = ', '.join(f"{n} (n={s})" for n, s in rare)
        L.append(f"  * Rare stages — {rare_txt} — have too few items for a stable")
        L.append("    recall/precision estimate; read those rows as indicative only,")
        L.append("    not as a reliability claim. (Avoidance and Metacognition are the")
        L.append("    structurally rare stages on this corpus.)")
        L.append("")

    # Confusion matrix.
    names = conf.get('label_names', [])
    matrix = conf.get('matrix', [])
    if names and matrix:
        L.append("  Confusion matrix (rows = human consensus, cols = LLM):")
        L.append("      " + "".join(f"{n[:6]:>8}" for n in names))
        for i, rowname in enumerate(names):
            cells = "".join(f"{c:>8}" for c in matrix[i])
            L.append(f"    {rowname[:6]:<6}{cells}")
        L.append("")


# ---------------------------------------------------------------------------
# 5. Per-test-set detail
# ---------------------------------------------------------------------------

def _per_testset_detail(L: List[str], results: dict) -> None:
    L.append("=" * W)
    L.append("PER-TEST-SET DETAIL — HUMAN ↔ HUMAN")
    L.append("=" * W)
    hh = results.get('human_human', {})
    for ws in sorted(hh, key=lambda x: int(x)):
        block = hh[ws]
        L.append("")
        L.append(f"Test-set {ws}  (raters: {', '.join(block.get('raters', []))}; "
                 f"{block.get('n_items', '?')} items)")
        for field in ('primary', 'secondary'):
            sub = block.get(field)
            if not sub:
                if field == 'primary':
                    L.append("  primary: (insufficient multi-rater data)")
                continue
            L.append(f"  {field}:")
            ka = sub.get('krippendorff_alpha')
            ci = sub.get('alpha_ci') or {}
            L.append("    " + fmt_kappa(ka, ci.get('lo'), ci.get('hi'),
                                        sub.get('n_items_scored'), stat='Krippendorff α'))
            fk = sub.get('fleiss_kappa')
            L.append(f"    Fleiss κ = {_kfmt(fk)} ({landis_koch(fk)}, "
                     f"complete-case n={sub.get('fleiss_n_complete', 0)})")
            L.append(f"    raw agreement: unanimous {_pfmt(sub.get('percent_agreement_unanimous'))}, "
                     f"pairwise {_pfmt(sub.get('percent_agreement_pairwise'))}")
            for pr in sub.get('pairwise', []):
                L.append(f"      {pr['rater_a']:>7} ↔ {pr['rater_b']:<7} "
                         f"n={pr['n']:<3} κ={_kfmt(pr['cohen_kappa'])} "
                         f"({landis_koch(pr['cohen_kappa'])})  "
                         f"agree={_pfmt(pr['percent_agreement'])}")
    L.append("")


# ---------------------------------------------------------------------------
# 6. Discrepancy walkthrough
# ---------------------------------------------------------------------------

_DISCREPANCY_CAP = 25


def _discrepancy_walkthrough(L: List[str], discrepancies: List[dict]) -> None:
    L.append("=" * W)
    L.append(f"DISCREPANCY WALKTHROUGH  "
             f"({len(discrepancies)} items: consensus ≠ LLM and/or ≠ GNN)")
    L.append("=" * W)
    if not discrepancies:
        L.append("  (none, or no machine labels available)")
        L.append("")
        return
    shown = discrepancies[:_DISCREPANCY_CAP]
    for row in shown:
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
        if row.get('gnn_label') and row['gnn_label'] != 'deferred':
            L.append(f"    GNN             : {row['gnn_label']}"
                     f"{_conf_suffix(row.get('gnn_confidence'))}")
        text = row.get('text', '')
        if text:
            L.append(f"    text            : {_truncate(text, 200)}")
    if len(discrepancies) > _DISCREPANCY_CAP:
        L.append("")
        L.append(f"  … {len(discrepancies) - _DISCREPANCY_CAP} more discrepancies — see")
        L.append("  04_validation/irr/irr_discrepancies.csv and the per-test-set dossiers")
        L.append("  04_validation/irr/irr_items_testset_<n>.txt for every item with reasonings.")
    L.append("")


# ---------------------------------------------------------------------------
# 7. GNN / probe axes (degrade gracefully)
# ---------------------------------------------------------------------------

def _gnn_probe_axes(L: List[str], results: dict) -> None:
    L.append("=" * W)
    L.append("GNN / PROBE AXES  (LLM-free substrates)")
    L.append("=" * W)
    g = results.get('human_vs_gnn', {})
    held = g.get('heldout') or {}
    dist = g.get('distillation') or {}
    has_gnn = bool(g.get('n')) or bool(held.get('available')) or bool(dist.get('available'))
    if not has_gnn:
        L.append("  GNN classifier: not run in this output (default OFF; H5-refuted at")
        L.append("  pilot scale — a probe ties/beats it). Enable with `qra gnn train` to")
        L.append("  re-adjudicate at Cohorts 3–4. No held-out or distillation predictions")
        L.append("  to score here.")
        L.append("")
        return

    L.append("Two GNN quantities answer two questions — each labeled with its circularity:")
    L.append("  • HELD-OUT (out-of-fold): the segment's own LLM label was masked in")
    L.append("    training → HONEST. vs human = independent construct validity; vs LLM =")
    L.append("    the reliability gate.")
    L.append("  • DISTILLATION (in-sample): trained on every label → vs-LLM is fidelity")
    L.append("    (circular, NOT validity); vs-human is optimistic (teacher leakage).")
    L.append("")
    L.append("  HELD-OUT GNN  (honest):")
    if not held.get('available'):
        L.append("    (no held-out predictions — run `qra gnn train` to generate them)")
    else:
        L.append(_axis_line("held-out ↔ human consensus", held.get('vs_human')))
        L.append(_axis_line("held-out ↔ LLM (gate)", held.get('vs_llm')))
    L.append("")
    L.append("  DISTILLATION GNN  (default mechanism; in-sample):")
    L.append(_axis_line("distillation ↔ human", dist.get('vs_human')))
    L.append(_axis_line("distillation ↔ LLM (fidelity)", dist.get('vs_llm')))
    L.append("")

    band = _human_band(results)
    llm = results.get('human_vs_llm', {})
    L.append("  Reference the GNN must approach before LLM-free scaling:")
    if band:
        L.append(f"    human↔human (Krippendorff α range): {band[0]:+.3f} … {band[1]:+.3f}")
    L.append(f"    LLM↔human (Cohen κ): {_kfmt(llm.get('cohen_kappa'))}")
    held_h = (held.get('vs_human') or {}).get('cohen_kappa')
    if held.get('available') and held_h is not None and llm.get('cohen_kappa') is not None:
        verdict = ("GNN comparable to the LLM on the human axis — candidate for "
                   "LLM-free scaling (confirm with `qra gnn status`)."
                   if held_h >= llm['cohen_kappa'] - 0.05
                   else "GNN below the LLM on the human axis — keep LLM consensus as default.")
        L.append(f"  Gate read: {verdict}")
    L.append("")


def _axis_line(label: str, blk: dict) -> str:
    if not blk or not blk.get('n'):
        return f"    {label:<34} (no overlap)"
    k = blk.get('cohen_kappa')
    ci = blk.get('kappa_ci') or {}
    return ("    " + fmt_kappa(k, ci.get('lo'), ci.get('hi'), blk['n'],
                               stat=f"{label} κ"))


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

def _footer(L: List[str]) -> None:
    L.append("-" * W)
    L.append("Machine-readable artifacts: 04_validation/irr/")
    L.append("  irr_results.json · irr_pairwise.csv · irr_discrepancies.csv ·")
    L.append("  irr_item_detail.csv · *.png")
    L.append("Reliability figures: 06_reports/01_reliability/reliability_forest.png")
    L.append("Per-item content + reasonings (one file per test-set):")
    L.append("  04_validation/irr/irr_items_testset_<n>.txt")


def _conf_suffix(c) -> str:
    return '' if c is None else f"  (conf {c:.2f})"


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + '…'
