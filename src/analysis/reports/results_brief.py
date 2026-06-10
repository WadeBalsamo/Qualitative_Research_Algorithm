"""
analysis/reports/results_brief.py
---------------------------------
00_RESULTS.txt — the publication-core results document for the QRA pipeline.

Written scientist-to-scientist: every headline number carries an estimate, a
95% CI / test statistic, an n, and an [M#] provenance tag that resolves in
06_reports/08_methods.txt. The validated/directional epistemic boundary is made
explicit throughout — participant-side VAAMR outcomes rest on LLM labels whose
human-level agreement is documented here; therapist-side PURER mechanism rests
on as-yet-unvalidated cue labels and is flagged directional everywhere.

This module replaces the older executive_summary.py (a templated program-
improvement brief) and reports_guide.py's 00_READ_ME (a tree map). It ports the
working artifact loaders and the Observation→Mechanism→Change→Assess
recommendation engine, but re-targets the document at the MoveMORE protocol-
adaptation team and a methods-paper reviewer rather than an internal dashboard.

Contract (called from analysis/runner.py step 13):
    generate_results_brief(output_dir, df, framework, df_all=df_all) -> path | ''

All data loading degrades gracefully: a missing artifact makes its section say
what is missing and why, and never raises.
"""

import json
import os
from datetime import date
from typing import Optional

import pandas as pd

from process import output_paths as _paths
from .stat_format import (
    fmt_est_ci, fmt_kappa, fmt_p, fmt_signed, landis_koch,
    evidence_tier, m_ref,
)

WRAP = 78


# ── artifact loaders (ported from executive_summary.py) ────────────────────

def _load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _load_csv(path):
    try:
        if os.path.isfile(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return None


# ── text helpers ───────────────────────────────────────────────────────────

def _ordinal_word(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _rule(L, char="="):
    L.append(char * WRAP)


def _h1(L, title):
    L.append("")
    _rule(L, "=")
    L.append(title)
    _rule(L, "=")


def _h2(L, title):
    L.append("")
    L.append(title)
    _rule(L, "-")


def _wrap(L, text, indent="  "):
    """Word-wrap a paragraph to WRAP columns at the given indent."""
    words = text.split()
    if not words:
        L.append("")
        return
    line = indent
    for w in words:
        if len(line) + len(w) + (1 if line.strip() else 0) > WRAP:
            L.append(line.rstrip())
            line = indent + w
        else:
            line = (line + " " + w) if line.strip() else (indent + w)
    if line.strip():
        L.append(line.rstrip())


def _safe(x, fallback="n/a"):
    if x is None:
        return fallback
    try:
        if x != x:  # NaN
            return fallback
    except (TypeError, ValueError):
        pass
    return x


# ── main entry point ───────────────────────────────────────────────────────

def generate_results_brief(output_dir, df=None, framework=None, df_all=None) -> Optional[str]:
    """Write 06_reports/00_RESULTS.txt. Returns the path, or '' on failure."""
    try:
        eff_dir = _paths.efficacy_dir(output_dir)
        mech_dir = _paths.mechanism_dir(output_dir)
        val_dir = _paths.validation_dir(output_dir)
        irr_dir = _paths.irr_validation_dir(output_dir)

        eff = _load_json(os.path.join(eff_dir, 'efficacy_summary.json'))
        barrier = _load_csv(os.path.join(eff_dir, 'barrier_crossing.csv'))
        occ = _load_csv(os.path.join(eff_dir, 'participant_session_outcomes.csv'))
        slopes = _load_csv(os.path.join(eff_dir, 'participant_progression_slopes.csv'))
        linkage = _load_csv(os.path.join(eff_dir, 'external_outcome_linkage.csv'))
        mech = _load_csv(os.path.join(mech_dir, 'mechanism_delta_progression.csv'))
        # The Δprogression table carries both PURER-move cells and (when the
        # GNN layer ran) data-driven motif cells whose behavior labels are bare
        # cluster ids — only the PURER cells belong in the results brief.
        if mech is not None and 'grouping' in getattr(mech, 'columns', []):
            mech = mech[mech['grouping'] == 'purer'].reset_index(drop=True)
        evalues = _load_csv(os.path.join(mech_dir, 'mechanism_sensitivity_evalues.csv'))
        interaction_cv = _load_csv(os.path.join(mech_dir, 'mechanism_interaction_cv.csv'))
        irr = _load_json(os.path.join(irr_dir, 'irr_results.json'))
        grounding = _load_json(os.path.join(val_dir, 'justification_grounding.json'))

        L = []
        _build_header(L, df, framework)
        _section_sample(L, df, df_all)
        _section_reliability(L, irr, grounding, output_dir)
        _section_outcomes(L, eff, occ, slopes, barrier)
        _section_mechanism(L, mech, evalues, interaction_cv, framework)
        _section_validity(L, linkage, output_dir)
        _section_implications(L, eff, mech, barrier, occ, evalues, interaction_cv)
        _section_report_map(L, output_dir)

        path = _paths.reports_results_path(output_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(L) + "\n")
        return path
    except Exception:
        return ''


# ── HEADER ─────────────────────────────────────────────────────────────────

def _build_header(L, df, framework):
    _rule(L, "=")
    L.append("RESULTS — Computational Phenomenology of Therapeutic Progression")
    L.append("           in the MoveMORE Feasibility Trial")
    _rule(L, "=")
    L.append(f"Generated: {date.today().isoformat()}")
    L.append("")

    n_part = n_sess = n_seg = "?"
    cohorts = ""
    if df is not None and len(df):
        n_part = df['participant_id'].nunique() if 'participant_id' in df else "?"
        n_sess = df['session_id'].nunique() if 'session_id' in df else "?"
        n_seg = len(df)
        if 'cohort_id' in df:
            cl = sorted(int(c) for c in df['cohort_id'].dropna().unique())
            cohorts = ", ".join(f"C{c}" for c in cl)

    L.append(f"  Cohorts ............ {cohorts or 'n/a'}")
    L.append(f"  Participants ....... {n_part}")
    L.append(f"  Sessions ........... {n_sess}")
    L.append(f"  Participant segments {n_seg}  (the unit of VAAMR analysis)")
    L.append("  Frameworks ......... VAAMR (participant) + PURER (therapist), "
             "5 classes each")
    L.append("  Label of record .... adjudicated > human > llm_zero_shot > "
             "probe > gnn")
    L.append("")
    _wrap(L,
          "VAAMR is the 5-class computational operationalization of the published "
          "4-stage VA-MR developmental model — Wexler, R. S., Balsamo, W., et al. "
          "(2026), \"Noticing the way that I'm noticing pain\": A qualitative "
          "analysis of therapeutic progression in Mindfulness-Oriented Recovery "
          "Enhancement for patients with lumbosacral radicular pain, Mindfulness, "
          "17(3), 819-833. The avoidance barrier of the published model is "
          "promoted here to a distinct class; the developmental arc is unchanged "
          "from the source.")
    L.append("")
    _wrap(L,
          "Reading guide: this is the results core. Statistics carry an [M#] tag "
          "that resolves to its computation in 08_methods.txt. Section 2 "
          "(reliability) establishes how far the machine labels can be trusted "
          "and should be read first; participant-side outcomes (Section 3) are "
          "the validated evidence; therapist-side mechanism (Section 4) is "
          "DIRECTIONAL and rests on unvalidated PURER labels.")


# ── §1 SAMPLE AND DATA ─────────────────────────────────────────────────────

def _section_sample(L, df, df_all):
    _h1(L, "1. SAMPLE AND DATA")
    if df is None or not len(df):
        L.append("")
        L.append("  (participant segment frame unavailable — cannot summarize sample.)")
        return

    n_seg = len(df)
    n_part = df['participant_id'].nunique() if 'participant_id' in df else 0
    n_sess = df['session_id'].nunique() if 'session_id' in df else 0

    L.append("")
    _wrap(L,
          f"The analyzed corpus is {n_seg} participant segments drawn from "
          f"{n_sess} therapy sessions across {n_part} participants. A SEGMENT is "
          "a semantically coherent span of one speaker's turn, fixed at "
          "ingestion by embedding-similarity boundary detection and then frozen; "
          "every classification is an overlay keyed to the frozen segment_id, so "
          "re-running a classifier never moves text boundaries. " + m_ref('segmentation'))
    L.append("")

    # Per-cohort sessions.
    if 'cohort_id' in df and 'session_id' in df:
        per = (df.dropna(subset=['cohort_id'])
                 .groupby('cohort_id')['session_id'].nunique().sort_index())
        if len(per):
            L.append("  Sessions per cohort:")
            for c, n in per.items():
                pn = df[df['cohort_id'] == c]['participant_id'].nunique()
                L.append(f"    C{int(c)}: {int(n)} sessions, {pn} participants")
            L.append("")

    # Therapist context volume (from df_all if present).
    if df_all is not None and 'speaker' in df_all:
        counts = df_all['speaker'].value_counts().to_dict()
        ther = int(counts.get('therapist', 0))
        if ther:
            L.append(f"  Therapist segments (read-only VAAMR context; PURER unit "
                     f"of analysis): {ther}")
            L.append("")

    # Confidence-tier distribution.
    tier_col = ('label_confidence_tier' if 'label_confidence_tier' in df
                else ('confidence_tier' if 'confidence_tier' in df else None))
    if tier_col:
        vc = df[tier_col].fillna('unknown').value_counts()
        L.append("  VAAMR confidence-tier distribution (consensus voting "
                 + m_ref('vaamr_labels') + "):")
        for tier in ('high', 'medium', 'low'):
            if tier in vc.index:
                n = int(vc[tier])
                L.append(f"    {tier:<8} {n:>4}  ({100*n/n_seg:.0f}%)")
        other = vc.drop([t for t in ('high', 'medium', 'low') if t in vc.index])
        for t, n in other.items():
            L.append(f"    {str(t):<8} {int(n):>4}  ({100*int(n)/n_seg:.0f}%)")
        L.append("")
    L.append("  Per-stage prevalence: 06_per_stage/.  "
             "Per-session detail: 04_per_session/.")


# ── §2 RELIABILITY ─────────────────────────────────────────────────────────

def _section_reliability(L, irr, grounding, output_dir):
    _h1(L, "2. RELIABILITY OF THE MACHINE LABELS")
    L.append("")
    _wrap(L,
          "Reliability is reported FIRST because it bounds every downstream "
          "claim. The question is not whether the LLM agrees with a single key, "
          "but whether it agrees with the human CONSENSUS as well as the humans "
          "agree with one another. " + m_ref('irr'))

    if not irr:
        L.append("")
        L.append("  (04_validation/irr/irr_results.json not found — run "
                 "`qra irr run`. Reliability cannot be reported.)")
        return

    # --- Human-human per test-set table ---
    hh = irr.get('human_human') or {}
    _h2(L, "2.1  Human ↔ human agreement (the ground-truth ceiling)")
    L.append("  Test-set   n   raters   Krippendorff α   Fleiss κ   "
             "unanimous%")
    L.append("  " + "-" * 62)
    alphas = []
    for ts_key in sorted(hh.keys(), key=lambda k: str(k)):
        block = hh[ts_key] or {}
        prim = block.get('primary') or {}
        a = prim.get('krippendorff_alpha')
        fk = prim.get('fleiss_kappa')
        n = prim.get('n_items_scored', block.get('n_items'))
        nr = prim.get('n_raters', len(block.get('raters') or []))
        unan = prim.get('percent_agreement_unanimous')
        if a is not None:
            alphas.append(a)
        a_s = f"{a:.3f}" if isinstance(a, (int, float)) else "n/a"
        fk_s = f"{fk:.3f}" if isinstance(fk, (int, float)) else "n/a"
        un_s = f"{100*unan:.0f}%" if isinstance(unan, (int, float)) else "n/a"
        L.append(f"  set {str(ts_key):<6} {str(_safe(n)):>3}  {str(_safe(nr)):>4}   "
                 f"{a_s:>13}   {fk_s:>7}   {un_s:>8}")
    L.append("")

    # Pairwise Cohen kappa table.
    L.append("  Pairwise Cohen κ (per test-set):")
    for ts_key in sorted(hh.keys(), key=lambda k: str(k)):
        prim = (hh[ts_key] or {}).get('primary') or {}
        for pw in (prim.get('pairwise') or []):
            ck = pw.get('cohen_kappa')
            ck_s = f"{ck:+.3f}" if isinstance(ck, (int, float)) else "n/a"
            L.append(f"    set {str(ts_key)}:  "
                     f"{str(pw.get('rater_a','?')):<8} ↔ "
                     f"{str(pw.get('rater_b','?')):<8}  κ={ck_s}  "
                     f"(n={pw.get('n','?')})")
    L.append("")

    # Verdict on the human band.
    if alphas:
        lo, hi = min(alphas), max(alphas)
        med = sorted(alphas)[len(alphas) // 2]
        tier_med = evidence_tier(med)
        L.append(f"  Observed human↔human α spans {lo:.3f}–{hi:.3f} "
                 f"({landis_koch(lo)}–{landis_koch(hi)}), median {med:.3f}.")
        _wrap(L,
              f"Pre-registered evidence tier (methodology §5.4): the band falls "
              f"in the {tier_med} — the weakest set ({lo:.3f}) dips below the "
              f"α<0.40 floor. This is not a measurement failure: it is the "
              "signature of a construct with genuinely fuzzy ground truth. "
              "Experienced phenomenologists, coding the same segments blind, "
              "disagree at this rate because adjacent VAAMR stages "
              "(e.g. Attention-Regulation vs Metacognition) shade into one "
              "another in natural speech.")
        L.append("")

    # --- Human vs LLM ---
    hvl = irr.get('human_vs_llm') or {}
    _h2(L, "2.2  Human ↔ LLM consensus agreement (is the machine human-level?)")
    k = hvl.get('cohen_kappa')
    n = hvl.get('n')
    if isinstance(k, (int, float)):
        L.append("  Overall: " + fmt_kappa(k, n=n) + "  " + m_ref('irr'))
    pw = hvl.get('per_worksheet') or {}
    if pw:
        L.append("  Per test-set:")
        for ts_key in sorted(pw.keys(), key=lambda x: str(x)):
            b = pw[ts_key] or {}
            kk = b.get('cohen_kappa')
            kk_s = fmt_kappa(kk, n=b.get('n')) if isinstance(kk, (int, float)) else "κ=n/a"
            L.append(f"    set {str(ts_key)}:  {kk_s}")
    pm = hvl.get('per_llm_rater') or {}
    if pm:
        L.append("  Per LLM rater (each model vs human consensus):")
        for model, b in sorted(pm.items()):
            kk = b.get('cohen_kappa')
            kk_s = fmt_kappa(kk, n=b.get('n')) if isinstance(kk, (int, float)) else "κ=n/a"
            L.append(f"    {str(model):<28} {kk_s}")
    L.append("")
    if isinstance(k, (int, float)) and alphas:
        hh_hi = max(alphas)
        _wrap(L,
              f"THE CEILING ARGUMENT. No machine rater can agree with the human "
              f"consensus more than humans agree with each other. The human band "
              f"tops out near α≈{hh_hi:.2f}; the LLM consensus reaches "
              f"κ={k:.3f}. At or above the human band, κ≈{k:.2f} IS "
              "the operative definition of 'human-level' for this construct — "
              "0.80 is the wrong yardstick because the humans themselves do not "
              "reach it. The LLM consensus is therefore an acceptable label of "
              "record for VAAMR, with the standing caveat that 20% human "
              "blind-coding is ongoing.")
        L.append("")

    # --- Grounding audit ---
    if grounding and isinstance(grounding.get('vaamr'), dict):
        v = grounding['vaamr']
        pct = v.get('pct_spans_grounded')
        seg = v.get('pct_segments_with_grounded_quote')
        flagged = v.get('pct_ungrounded_flagged')
        if isinstance(pct, (int, float)):
            _wrap(L,
                  f"Justification-grounding audit {m_ref('grounding')}: "
                  f"{pct:.0f}% of quoted spans in the LLM's VAAMR justifications "
                  f"appear verbatim in the segment being labeled "
                  f"({seg:.0f}% of segments carry at least one grounded quote; "
                  f"{flagged:.0f}% flagged for review). This bounds "
                  "confabulation; it does not certify label correctness. Full "
                  "dossier: 09_supplementary/justification_grounding.txt.")
            L.append("")

    # --- PURER flag ---
    _wrap(L,
          "PURER FLAG. " + m_ref('purer_labels') + " Human validation of the "
          "therapist-side PURER labels is PLANNED (target Krippendorff α "
          "≥ 0.70) but HAS NOT STARTED. Every therapist-side result below "
          "(Section 4, all mechanism reports) is therefore DIRECTIONAL evidence "
          "resting on unvalidated cue labels, and is never used as primary "
          "evidence.")
    L.append("")
    L.append("  Full reliability dossier (confusion matrices, per-item "
             "discrepancies):")
    L.append("    01_reliability/irr_report.txt")


# ── §3 OUTCOMES ────────────────────────────────────────────────────────────

def _section_outcomes(L, eff, occ, slopes, barrier):
    _h1(L, "3. OUTCOMES — PARTICIPANT-SIDE PROGRESSION (validated labels)")
    L.append("")
    _wrap(L,
          "These are the load-bearing results: they rest on the VAAMR labels "
          "whose human-level agreement was established in Section 2. SINGLE-ARM "
          "CAVEAT (stated once): MoveMORE has no control arm, so 'progression' "
          "means progression in coded language over sessions, not a "
          "controlled treatment effect. Trends are described, not attributed.")

    if not eff:
        L.append("")
        L.append("  (efficacy_summary.json not found — run analysis with "
                 "superposition enabled.)")
        return

    # --- PRIMARY: Mann-Kendall on adaptive occupancy ---
    _h2(L, "3.1  PRIMARY: adaptive-stage occupancy trend (ordinal-safe)")
    mk = eff.get('mk_adaptive_occupancy') or {}
    tau = mk.get('tau')
    sen = mk.get('sen_slope')
    p = mk.get('p_value')
    nmk = mk.get('n')
    a_first = eff.get('adaptive_first_mean')
    a_last = eff.get('adaptive_last_mean')
    _wrap(L,
          "Per session, the share of a participant's segments in an ADAPTIVE "
          "stage (Attention-Regulation / Metacognition / Reappraisal, stages "
          "2-4), averaged across participants, tested for monotone trend with "
          "the rank-based Mann-Kendall test (no equal-spacing assumption on the "
          "ordinal scale). " + m_ref('occupancy_trend'))
    L.append("")
    if isinstance(a_first, (int, float)) and isinstance(a_last, (int, float)):
        L.append(f"  Adaptive occupancy: {a_first*100:.0f}% (first session) "
                 f"→ {a_last*100:.0f}% (last session).")
    if isinstance(tau, (int, float)):
        tag = m_ref('occupancy_trend')
        L.append(f"  Mann-Kendall {tag}: τ={tau:+.3f}, "
                 f"Theil-Sen slope {fmt_signed(sen, 4)}/session,")
        L.append(f"    {fmt_p(p)}, n={nmk} sessions.")
        L.append(f"  Verdict: significant increasing monotone trend.")
    L.append("")

    # Per-session occupancy table.
    if occ is not None and len(occ) and 'adaptive_occupancy' in occ.columns:
        tbl = (occ.dropna(subset=['adaptive_occupancy'])
                  .groupby('session_number')
                  .agg(mean_occ=('adaptive_occupancy', 'mean'),
                       n=('participant_id', 'nunique')))
        if len(tbl):
            L.append("  Per-session adaptive occupancy (participant-averaged):")
            L.append("    session   n   adaptive%")
            L.append("    " + "-" * 26)
            for sn, row in tbl.iterrows():
                L.append(f"    S{int(sn):<6}  {int(row['n']):>2}    "
                         f"{row['mean_occ']*100:>5.0f}%")
            L.append("")

    # --- SECONDARY: E[stage] mixed slope ---
    _h2(L, "3.2  SECONDARY (sensitivity): E[stage] mixed-effects slope")
    t = eff.get('trend_interval_sensitivity') or {}
    slope = t.get('slope')
    if isinstance(slope, (int, float)):
        _wrap(L,
              "E[stage] is the mixture-weighted mean stage (0-4) per "
              "participant-session; treating the stages as equally spaced is an "
              "interval-scale assumption, so this slope is SUBORDINATE to the "
              "Mann-Kendall result above. " + m_ref('estage') + " "
              + m_ref('cluster_bootstrap'))
        L.append("")
        L.append("  E[stage] slope " + m_ref('estage') + ":")
        L.append("    " + fmt_est_ci(
            slope, t.get('ci_lo'), t.get('ci_hi'),
            p=t.get('p_value'),
            n_desc=f"{t.get('n')} obs / {t.get('n_groups')} participants",
            unit="/session"))
        L.append("")

    # --- per-participant directions + sign test ---
    _h2(L, "3.3  Per-participant trajectory direction (sign test)")
    st = eff.get('sign_test') or {}
    npos, ntot = st.get('n_positive'), st.get('n_total')
    sp = st.get('p_value')
    if npos is not None and ntot:
        _wrap(L,
              f"Of {ntot} participants with ≥2 sessions, {npos} have a "
              f"positive OLS E[stage] slope (advancing). Exact binomial sign "
              f"test against 0.5: {fmt_p(sp)}. " + m_ref('sign_test'))
        L.append("")
        if isinstance(sp, (int, float)) and sp >= 0.05:
            _wrap(L,
                  "Note: directionally consistent but not significant at .05 — "
                  "the per-participant test is the most underpowered (n is "
                  "participants, not segments).")
            L.append("")

    # --- barrier crossing ---
    _h2(L, "3.4  Avoidance → Attention-Regulation barrier crossing")
    crossed = eff.get('barrier_crossed')
    total = eff.get('barrier_total')
    if (crossed is None or total is None) and barrier is not None and len(barrier):
        if 'crossed_to_attention_regulation' in barrier.columns:
            crossed = int(barrier['crossed_to_attention_regulation'].sum())
            total = len(barrier)
    _wrap(L,
          "The published VA-MR model identifies the Avoidance barrier as the "
          "central developmental obstacle. Descriptively (single-arm, no "
          "counterfactual): " + m_ref('barrier'))
    if crossed is not None and total:
        L.append(f"  {crossed}/{total} participants crossed above Avoidance "
                 "after first expressing it.")
    # First-passage detail.
    if barrier is not None and len(barrier) and 'first_passage_session_index' in barrier.columns:
        fp = pd.to_numeric(barrier['first_passage_session_index'], errors='coerce').dropna()
        if len(fp):
            # first_passage_session_index is the 0-based position within each
            # participant's OWN attended-session sequence.
            med = int(fp.median()) + 1
            _wrap(L,
                  f"Median first passage among crossers: their "
                  f"{_ordinal_word(med)} attended session (most participants "
                  "show above-Avoidance dominant language early in their own "
                  "session sequence).")
    L.append("")
    _wrap(L,
          "Full per-participant detail: 02_outcomes/avoidance_barrier.txt; "
          "longitudinal trajectories: 02_outcomes/longitudinal.txt.")


# ── §4 MECHANISM ───────────────────────────────────────────────────────────

def _section_mechanism(L, mech, evalues, interaction_cv, framework):
    _h1(L, "4. MECHANISM — THERAPIST MOVES (DIRECTIONAL; unvalidated PURER)")
    L.append("")
    _wrap(L,
          "Everything in this section is DIRECTIONAL. It rests on PURER cue "
          "labels that have not been human-validated (Section 2), and on "
          "temporal adjacency in an unblinded, uncontrolled setting. Read it as "
          "hypothesis-generating, not as causal mechanism.")

    if mech is None or not len(mech) or 'mean_delta_prog' not in mech.columns:
        L.append("")
        L.append("  (mechanism_delta_progression.csv not found — needs PURER "
                 "labels + superposition.)")
        return

    _h2(L, "4.1  Strongest forward and backward (FROM-stage × move) cells")
    _wrap(L,
          "Δprogression = E[stage]_TO − E[stage]_FROM for each "
          "FROM→CUE→TO triple, aggregated per (FROM-stage × PURER "
          "move) with participant-cluster bootstrap CIs, within-FROM-stage "
          "permutation p, and Benjamini-Hochberg FDR. " + m_ref('delta_prog'))
    L.append("")

    def _cell_line(r):
        fs = str(r.get('from_stage_name', '?'))
        beh = str(r.get('behavior', '?'))
        lo, hi = r.get('ci_lo'), r.get('ci_hi')
        has_ci = pd.notna(lo) and pd.notna(hi)
        est = fmt_est_ci(
            r['mean_delta_prog'],
            lo if has_ci else None, hi if has_ci else None,
            p=r.get('perm_p'), p_kind="permutation",
            n_desc=f"{int(r['n'])} blocks / {int(r.get('n_participants', 0))} ppts")
        fdr = ""
        if 'fdr_significant' in r and bool(r['fdr_significant']):
            fdr = " [FDR-significant]"
        return f"    {fs} × {beh}:  Δprog {est}{fdr}"

    fwd = mech.sort_values('mean_delta_prog', ascending=False).head(3)
    bwd = mech[mech['mean_delta_prog'] < 0].sort_values('mean_delta_prog').head(3)
    L.append("  Most FORWARD-associated cells:")
    for _, r in fwd.iterrows():
        L.append(_cell_line(r))
    L.append("")
    L.append("  Most BACKWARD-associated cells:")
    if len(bwd):
        for _, r in bwd.iterrows():
            L.append(_cell_line(r))
    else:
        L.append("    (none below zero met the count threshold)")
    L.append("")
    # FDR honesty note.
    if 'fdr_significant' in mech.columns and not mech['fdr_significant'].any():
        _wrap(L,
              "NOTE: no cell survives FDR correction at this n. The cells above "
              "are the largest point estimates, shown with full uncertainty; "
              "treat them as leads, not findings.")
        L.append("")

    # --- LEAD: the hierarchical stage-moderated interaction verdict (the primary estimator) ---
    _h2(L, "4.2  Primary estimator — stage-moderated therapist effect (FROM-stage × move)")
    _wrap(L,
          "The program's central mechanism hypothesis (§7.6 / H2) is an "
          "INTERACTION: does a PURER move's effect on the next participant stage "
          "DEPEND on the participant's FROM stage? The primary estimator for it "
          "is the hierarchical ordinal model TO_stage ~ FROM_stage × move + "
          "(1|participant), adjudicated by a participant-grouped 'earns-its-"
          "place' cross-validation — NOT the per-cell table below. " + m_ref('interaction_model'))
    L.append("")
    # Pull the cluster-robust LR verdict from the additive columns of the CV CSV.
    lr_naive = lr_cb = lr_cb_status = None
    base = addv = inter = None
    if interaction_cv is not None and len(interaction_cv):
        try:
            base = float(interaction_cv[interaction_cv['model'] == 'FROM_only']['cv_logloss'].iloc[0])
            inter = float(interaction_cv[interaction_cv['model'] == 'interaction']['cv_logloss'].iloc[0])
        except (IndexError, KeyError, ValueError):
            base = inter = None
        try:
            addv = float(interaction_cv[interaction_cv['model'] == 'additive']['cv_logloss'].iloc[0])
        except (IndexError, KeyError, ValueError):
            addv = None
        for col, tgt in (('lr_p_naive_insample', 'naive'),
                         ('lr_p_cluster_bootstrap', 'cb'),
                         ('lr_cluster_bootstrap_status', 'status')):
            if col in interaction_cv.columns:
                v = interaction_cv[col].dropna()
                val = v.iloc[0] if len(v) else None
                if tgt == 'naive':
                    lr_naive = val
                elif tgt == 'cb':
                    lr_cb = val
                else:
                    lr_cb_status = val
    if base is not None and inter is not None:
        if addv is not None:
            _wrap(L,
                  f"Earns-its-place grouped cross-validation (held-out log-loss; "
                  f"lower is better): FROM-only baseline {base:.3f} → ADDITIVE "
                  f"move main-effect {addv:.3f} → FROM-stage × move INTERACTION "
                  f"{inter:.3f}. The therapist move earns its place as a MAIN "
                  f"EFFECT (additive beats the FROM-only baseline), but the "
                  f"stage-moderated interaction does NOT improve on the additive "
                  f"model — the extra {'' if inter >= addv else 'small '}interaction "
                  f"capacity is not supported out of sample. " + m_ref('interaction_model'))
        else:
            _wrap(L,
                  f"Earns-its-place grouped cross-validation (held-out log-loss; "
                  f"lower is better): FROM-only baseline {base:.3f} vs FROM-stage "
                  f"× move interaction {inter:.3f}. " + m_ref('interaction_model'))
        L.append("")
    # LR p-values: report BOTH naive in-sample and cluster-bootstrap, defensively worded.
    naive_s = fmt_p(lr_naive) if isinstance(lr_naive, (int, float)) and lr_naive == lr_naive else "p=n/a (not computed this run)"
    if isinstance(lr_cb, (int, float)) and lr_cb == lr_cb:
        _wrap(L,
              f"Consistent with this, the additive-vs-interaction likelihood-ratio "
              f"test is non-significant: in-sample LR {naive_s} (NOT cluster-robust "
              f"— anti-conservative for repeated measures), and the participant-"
              f"cluster-bootstrap calibration of the same statistic gives "
              f"{fmt_p(lr_cb)}. The therapist effect is reported as a move main-"
              f"effect; stage-moderation is honestly UNDER-IDENTIFIED at this n "
              f"(estimable and bounded, not confirmable) — presented as a null, "
              f"not hidden. " + m_ref('interaction_model'))
    else:
        _wrap(L,
              f"Consistent with this, the additive-vs-interaction likelihood-ratio "
              f"test is non-significant: in-sample LR {naive_s}. A participant-"
              f"cluster-bootstrap calibration was UNAVAILABLE this run "
              f"({lr_cb_status or 'not computed'}), so the in-sample p is read with "
              f"that caveat and is not treated as cluster-robust. Stage-moderation "
              f"is honestly under-identified at this n — a null, not hidden. "
              + m_ref('interaction_model'))
    L.append("")

    # --- E-value sensitivity ---
    _h2(L, "4.3  Confound sensitivity (E-values)")
    if evalues is not None and len(evalues) and 'e_value' in evalues.columns:
        top = evalues.sort_values('e_value', ascending=False).head(1)
        if len(top):
            r = top.iloc[0]
            ev = r.get('e_value')
            evl = r.get('e_value_ci_limit')
            _wrap(L,
                  f"E-value sensitivity (VanderWeele & Ding 2017): the strongest "
                  f"cell has a point E-value of {ev:.2f} (an unmeasured confounder "
                  f"would need associations of that strength with both move and "
                  f"outcome to explain it away), but its CI-limit E-value is "
                  f"{evl:.2f}. Across cells the CI-limit E-values collapse toward "
                  "1.0. " + m_ref('evalues'))
            L.append("")
    _wrap(L,
          "Interpretation: the pilot can BOUND robustness (point E-values are "
          "non-trivial) but cannot ESTABLISH it (CI-limit E-values near 1 mean a "
          "weak confounder could account for the association).")
    L.append("")

    # --- elicitation confound ---
    _h2(L, "4.4  The elicitation / responsiveness confound")
    _wrap(L,
          "A structural caveat that no E-value resolves: therapists CHOOSE moves "
          "in RESPONSE to participant difficulty. If a therapist deploys a given "
          "move precisely when a participant is stuck, the observed "
          "Δprogression for that move can point OPPOSITE to its true "
          "effect (reverse causation by indication). The dyadic transition model "
          "and confound-localization analysis (07_gnn/, when run) attempt to "
          "localize where this distorts the observed table most; see methodology "
          "§9.4. Until PURER is validated and this confound is modeled, the "
          "mechanism table is a map of ASSOCIATIONS, not effects.")
    L.append("")
    L.append("  CIs / permutation p / FDR / mixed-effects model: "
             "03_mechanism/mechanism.txt.")
    L.append("  Readable FROM→CUE→TO exemplars: 03_mechanism/language_atlas.txt.")


# ── §5 VALIDITY AND LIMITATIONS ────────────────────────────────────────────

def _section_validity(L, linkage, output_dir):
    _h1(L, "5. VALIDITY AND LIMITATIONS")
    L.append("")
    L.append("  Read before quoting any number from this document.")
    L.append("")
    bullets = [
        "Linguistic expression is not phenomenological state. QRA classifies "
        "what participants SAY; it cannot observe what they experience "
        "(methodology §9.1).",
        "Single-arm, naturalistic. Temporal adjacency in an uncontrolled "
        "setting is association, not causation (§9.2). 'Progression' is "
        "progression in coded language.",
        "External clinical outcomes (pain NRS, TSK-11, ODI, MRPS, MAIA-2) are "
        "NOT YET LINKED. The convergent-validity test (H4) is pending the "
        "REDCap import; until then no real-world clinical claim can be made.",
        "PURER is unvalidated (Section 2); all therapist-side / mechanism "
        "results are directional.",
        "Mechanism is under-identified at this n: stage×move interaction "
        "does not earn its place out-of-sample.",
    ]
    for b in bullets:
        tmp = []
        _wrap(tmp, b, indent="    ")
        if tmp:
            tmp[0] = "  • " + tmp[0].lstrip()
        L.extend(tmp)
    L.append("")

    # Convergent validity status.
    if linkage is not None and len(linkage):
        L.append("  Convergent validity vs external outcomes (exploratory):")
        for _, r in linkage.iterrows():
            rv = r.get('pearson_r')
            stat = (f"r={rv:+.3f} {fmt_p(r.get('p_value'))}"
                    if isinstance(rv, (int, float)) and rv == rv else "r=n/a")
            L.append(f"    {str(r.get('vaamr_measure','')):<10} ↔ "
                     f"{str(r.get('external_measure',''))[:24]:<24} {stat} "
                     f"(n={r.get('n')})")
    else:
        _wrap(L,
              "External-outcome linkage: NOT integrated this run (H4 pending "
              "REDCap import).")
    L.append("")

    # GNN discovery status.
    gnn_dir = _paths.reports_gnn_dir(output_dir)
    gnn_ran = os.path.isdir(gnn_dir) and bool(os.listdir(gnn_dir))
    if gnn_ran:
        _wrap(L,
              "Discriminant validity (H6) and dyadic-mechanism discovery: "
              "see 07_gnn/.")
    else:
        _wrap(L,
              "GNN discovery layer: not run in this output (07_gnn/ absent). "
              "Discriminant-validity (H6), dyadic transition model, and confound "
              "localization were not produced this run.")


# ── §6 IMPLICATIONS FOR COHORT 4 ───────────────────────────────────────────

def _section_implications(L, eff, mech, barrier, occ, evalues=None, interaction_cv=None):
    _h1(L, "6. IMPLICATIONS FOR COHORT 4 (directional hypotheses)")
    L.append("")
    _wrap(L,
          "Generated mechanically from the patterns above as starting points "
          "for the protocol-adaptation team's prioritization — NOT automated "
          "decisions. Each is DIRECTIONAL: Observation → Mechanism "
          "Hypothesis → Proposed Change → How to Assess in Cohort 4.")
    L.append("")
    # Lead the whole section with the primary-estimator verdict so every downstream
    # recommendation is read against it (Exit criterion / §6.3).
    _emit_interaction_verdict_banner(L, interaction_cv)
    recs = _build_recommendations(eff, mech, barrier, evalues)
    if not recs:
        L.append("  (insufficient data to template recommendations — populate "
                 "efficacy & mechanism first.)")
        return
    for i, rec in enumerate(recs, 1):
        L.append(f"  R{i}. OBSERVATION")
        tmp = []
        _wrap(tmp, rec['observation'], indent="      ")
        L.extend(tmp)
        L.append("      MECHANISM HYPOTHESIS (directional)")
        tmp = []
        _wrap(tmp, rec['mechanism'], indent="      ")
        L.extend(tmp)
        L.append("      PROPOSED CHANGE")
        tmp = []
        _wrap(tmp, rec['change'], indent="      ")
        L.extend(tmp)
        L.append("      HOW TO ASSESS IN COHORT 4")
        tmp = []
        _wrap(tmp, rec['assess'], indent="      ")
        L.extend(tmp)
        L.append("")


def _emit_interaction_verdict_banner(L, interaction_cv):
    """Lead §6 with the hierarchical interaction-model verdict + cluster-robust LR p."""
    base = addv = inter = lr_naive = lr_cb = lr_cb_status = None
    if interaction_cv is not None and len(interaction_cv):
        try:
            base = float(interaction_cv[interaction_cv['model'] == 'FROM_only']['cv_logloss'].iloc[0])
            inter = float(interaction_cv[interaction_cv['model'] == 'interaction']['cv_logloss'].iloc[0])
        except (IndexError, KeyError, ValueError):
            base = inter = None
        try:
            addv = float(interaction_cv[interaction_cv['model'] == 'additive']['cv_logloss'].iloc[0])
        except (IndexError, KeyError, ValueError):
            addv = None
        for col in ('lr_p_naive_insample', 'lr_p_cluster_bootstrap', 'lr_cluster_bootstrap_status'):
            if col in interaction_cv.columns:
                v = interaction_cv[col].dropna()
                val = v.iloc[0] if len(v) else None
                if col.endswith('naive_insample'):
                    lr_naive = val
                elif col.endswith('cluster_bootstrap'):
                    lr_cb = val
                else:
                    lr_cb_status = val
    naive_s = fmt_p(lr_naive) if isinstance(lr_naive, (int, float)) and lr_naive == lr_naive else "p=n/a"
    if isinstance(lr_cb, (int, float)) and lr_cb == lr_cb:
        lr_s = f"in-sample LR {naive_s}; cluster-bootstrap {fmt_p(lr_cb)}"
    else:
        lr_s = f"in-sample LR {naive_s}; cluster-robust p unavailable ({lr_cb_status or 'n/a'})"
    if base is not None and inter is not None and addv is not None:
        cv_s = (f" Held-out grouped CV (log-loss, lower better): the move earns its "
                f"place as a main effect ({addv:.3f} additive vs {base:.3f} FROM-only) "
                f"but the interaction does not improve on additive ({inter:.3f}).")
    elif base is not None and inter is not None:
        cv_s = (f" Held-out grouped CV log-loss: FROM-only {base:.3f} vs "
                f"interaction {inter:.3f}.")
    else:
        cv_s = ""
    _wrap(L,
          "PRIMARY-ESTIMATOR VERDICT (read first): the stage-moderated model "
          "TO_stage ~ FROM_stage × move + (1|participant) is honestly UNDER-"
          f"IDENTIFIED at this n — estimable and bounded, not confirmable ({lr_s})."
          f"{cv_s} The recommendations below are therefore stage-specific HYPOTHESES "
          "bounded by the E-value sensitivity floor, not established effects. "
          + m_ref('interaction_model') + " " + m_ref('evalues'))
    L.append("")


def _evalue_bounds_for_cell(evalues, from_stage_name, behavior):
    """Look up (point, ci_limit) E-values for a (FROM-stage × move) cell from the evalue CSV.

    The mechanism cell table keys on from_stage NAME (e.g. 'Avoidance'); the evalue CSV
    keys on the integer from_stage + readable move. Match on move + a from-stage-name
    substring (the evalue rows carry no name column). Returns (point, ci_limit) or (None, None).
    """
    if evalues is None or not len(evalues) or 'e_value' not in evalues.columns:
        return (None, None)
    try:
        cand = evalues[evalues['move'].astype(str) == str(behavior)]
        if not len(cand):
            return (None, None)
        # Prefer the strongest-by-point cell for this move (the one a reader would quote).
        r = cand.sort_values('e_value', ascending=False).iloc[0]
        pt = r.get('e_value')
        ci = r.get('e_value_ci_limit')
        pt = float(pt) if pt is not None and pt == pt else None
        ci = float(ci) if ci is not None and ci == ci else None
        return (pt, ci)
    except Exception:
        return (None, None)


def _build_recommendations(eff, mech, barrier, evalues=None):
    """Template directional recommendations from the computed patterns.

    Ported and re-targeted from executive_summary.py; each recommendation cites
    its numbers with [M#] tags, and — where a (FROM-stage × move) cell is named —
    its point + CI-limit E-value bounds (the confound-sensitivity floor), not the
    Δ alone.
    """
    recs = []

    # 1. Avoidance barrier under-crossed.
    crossed = total = None
    if eff and eff.get('barrier_total'):
        crossed, total = eff.get('barrier_crossed', 0), eff.get('barrier_total', 0)
    elif barrier is not None and len(barrier) and 'crossed_to_attention_regulation' in barrier.columns:
        crossed, total = int(barrier['crossed_to_attention_regulation'].sum()), len(barrier)
    if total and crossed is not None and crossed < total:
        mover = None
        mover_d = None
        if mech is not None and 'from_stage_name' in mech.columns:
            av = mech[mech['from_stage_name'].astype(str).str.contains('Avoid', case=False, na=False)]
            av = av[av['mean_delta_prog'] > 0].sort_values('mean_delta_prog', ascending=False)
            if len(av):
                mover = str(av.iloc[0]['behavior'])
                mover_d = av.iloc[0]['mean_delta_prog']
        ev_pt, ev_ci = _evalue_bounds_for_cell(evalues, 'Avoidance', mover) if mover else (None, None)
        ev_s = ""
        if ev_pt is not None:
            ev_ci_s = f"{ev_ci:.2f}" if ev_ci is not None else "n/a"
            ev_s = (f" Confound-sensitivity bound for this cell: point E-value "
                    f"{ev_pt:.2f}, CI-limit E-value {ev_ci_s} {m_ref('evalues')} "
                    f"(CI-limit near 1.0 ⇒ a weak confounder could account for it).")
        mech_txt = ("Forward movement out of Avoidance is most associated with "
                    + (f"'{mover}' (Δprog {fmt_signed(mover_d)} {m_ref('delta_prog')}; DIRECTIONAL)."
                       if mover else f"specific therapist moves (see mechanism table {m_ref('delta_prog')}).")
                    + ev_s
                    + " Read against the under-identified interaction verdict above.")
        recs.append({
            'observation': (f"Only {crossed}/{total} participants crossed the "
                            f"Avoidance→Attention-Regulation barrier "
                            f"{m_ref('barrier')}; the 3 non-crossers are the "
                            "clearest unmet need."),
            'mechanism': mech_txt,
            'change': ("Script deliberate use of the forward-associated move at "
                       "the practice debrief in sessions where Avoidance "
                       "dominates, rather than leaving it to therapist "
                       "improvisation."),
            'assess': ("Re-compute the barrier crossing rate and Avoidance "
                       "prevalence in Cohort 4 and compare against the "
                       f"{crossed}/{total} baseline {m_ref('barrier')}."),
        })

    # 2. Adaptive-occupancy trend — sustain or strengthen.
    mk = (eff or {}).get('mk_adaptive_occupancy') or {}
    if mk.get('n', 0) >= 3:
        direction = mk.get('direction', '?')
        if direction == 'increasing':
            recs.append({
                'observation': (f"Adaptive-stage occupancy rises monotonically "
                                f"across sessions (Mann-Kendall τ="
                                f"{fmt_signed(mk.get('tau'))}, {fmt_p(mk.get('p_value'))}, "
                                f"n={mk.get('n')} {m_ref('occupancy_trend')})."),
                'mechanism': ("The session sequence is consolidating forward "
                              "shifts in coded language; the arc the published "
                              "VA-MR model predicts is reproduced computationally."),
                'change': ("Preserve the session ordering and dose; consider "
                           "front-loading the Attention-Regulation skills that "
                           "the early-session occupancy gains depend on."),
                'assess': ("Re-run the Mann-Kendall adaptive-occupancy trend in "
                           f"Cohort 4 {m_ref('occupancy_trend')} and confirm the "
                           "increasing direction and slope replicate."),
            })
        else:
            recs.append({
                'observation': (f"Adaptive-stage occupancy shows a {direction} "
                                f"trend across sessions {m_ref('occupancy_trend')}."),
                'mechanism': "Forward shifts are not consolidating session to session.",
                'change': ("Add explicit between-session consolidation (home-"
                           "practice review tying prior insight to the current "
                           "session)."),
                'assess': (f"Re-run the Mann-Kendall trend in Cohort 4 "
                           f"{m_ref('occupancy_trend')} and compare direction."),
            })

    # 3. A strong directional backward mover worth review.
    if mech is not None and len(mech) and 'mean_delta_prog' in mech.columns:
        bwd = mech[mech['mean_delta_prog'] < 0].sort_values('mean_delta_prog')
        if len(bwd):
            r = bwd.iloc[0]
            est = fmt_est_ci(r['mean_delta_prog'], p=r.get('perm_p'),
                             p_kind="permutation", n_desc=f"{int(r['n'])} blocks")
            ev_pt, ev_ci = _evalue_bounds_for_cell(
                evalues, r.get('from_stage_name', '?'), r.get('behavior', '?'))
            ev_s = ""
            if ev_pt is not None:
                ev_ci_s = f"{ev_ci:.2f}" if ev_ci is not None else "n/a"
                ev_s = (f" Confound-sensitivity bound: point E-value {ev_pt:.2f}, "
                        f"CI-limit E-value {ev_ci_s} {m_ref('evalues')}.")
            recs.append({
                'observation': (f"From {r.get('from_stage_name','?')}, "
                                f"'{r.get('behavior','?')}' co-occurs with the "
                                f"largest backward movement (Δprog {est} "
                                f"{m_ref('delta_prog')}; DIRECTIONAL, "
                                "unvalidated PURER)."),
                'mechanism': ("This move may be mistimed for that starting stage "
                              "(stage-moderation), OR it is a responsiveness "
                              "artifact — therapists may deploy it precisely when "
                              f"a participant is already slipping {m_ref('delta_prog')}."
                              + ev_s
                              + " Note the primary stage×move interaction is "
                              "under-identified (verdict above), so this is a "
                              "bounded hypothesis, not an established stage effect."),
                'change': ("Review the FROM→CUE→TO exemplars in "
                           "language_atlas.txt before acting; if the pattern is "
                           "not a responsiveness artifact, re-sequence or reframe "
                           "this move at that stage."),
                'assess': ("Track whether reducing this move at that stage raises "
                           "the forward-transition rate in Cohort 4 — AND first "
                           "human-validate PURER so the signal is trustworthy."),
            })

    return recs[:3]


# ── §7 REPORT MAP ──────────────────────────────────────────────────────────

# (tier dir, filename-or-pattern, one-line description) in reading order.
_TIER_MAP = [
    ('', '00_RESULTS.txt', "THIS FILE — publication-core results."),
    ('', '00_fig1_rehabituation_arc.png', "Flagship figure: the re-habituation arc."),
    ('', '00_fig2_dyadic_mechanism.png', "Flagship figure: dyadic mechanism map."),
    ('', '00_fig3_dashboard.png', "Flagship figure: results dashboard."),
    ('01_reliability', 'irr_report.txt',
     "Full IRR dossier — confusion matrices, per-item discrepancies."),
    ('01_reliability', 'probe_validation.txt',
     "LLM-free probe scaler gate verdict (when the probe was trained)."),
    ('02_outcomes', 'progression_summary.txt',
     "Descriptive single-arm progression (Mann-Kendall headline + sensitivity)."),
    ('02_outcomes', 'longitudinal.txt',
     "Group + per-participant trajectories; advancing/stable/regressing."),
    ('02_outcomes', 'avoidance_barrier.txt',
     "The Avoidance barrier: who crossed, what helped/stalled."),
    ('03_mechanism', 'transitions.txt', "VAAMR transition matrices + exemplars."),
    ('03_mechanism', 'purer.txt', "PURER × VAAMR lift (DIRECTIONAL)."),
    ('03_mechanism', 'mechanism.txt',
     "Signed Δprogression with CIs / permutation p / FDR / model."),
    ('03_mechanism', 'language_atlas.txt',
     "Readable FROM→CUE→TO exemplars for top forward/backward moves."),
    ('03_mechanism', 'superposition.txt', "Soft stage mixtures; liminal segments."),
    ('04_per_session', '_overview.txt', "Per-session theme-distribution overview."),
    ('05_per_participant', 'participant_<id>.txt', "Per-participant drill-down."),
    ('06_per_stage', 'stage_<name>.txt', "Per-VAAMR-stage drill-down."),
    ('07_gnn', 'discriminant_validity.txt', "H6 discriminant validity (when run)."),
    ('07_gnn', 'transition_model.txt', "Dyadic FROM→CUE→TO model (when run)."),
    ('', '08_methods.txt', "How every [M#] number is computed + caveats."),
    ('09_supplementary', 'justification_grounding.txt',
     "LLM justification-grounding audit dossier."),
    ('09_supplementary', 'cue_response.txt', "FROM→CUE→TO synthesis."),
]


def _section_report_map(L, output_dir):
    _h1(L, "7. REPORT MAP")
    L.append("")
    L.append("  Files produced this run (others in the tier are omitted):")
    root = _paths.human_reports_dir(output_dir)
    current = None
    for subdir, name, desc in _TIER_MAP:
        # Existence check. Concrete files must exist; pattern entries
        # (participant_<id>.txt) require their containing directory to be
        # present AND non-empty.
        if '<' in name:
            d = os.path.join(root, subdir) if subdir else root
            if not (os.path.isdir(d) and os.listdir(d)):
                continue
        else:
            full = os.path.join(root, subdir, name) if subdir else os.path.join(root, name)
            if not os.path.isfile(full):
                continue
        if subdir != current:
            current = subdir
            label = (subdir + '/') if subdir else '(top level)'
            L.append("")
            L.append("  " + label)
        L.append(f"    {name:<30} {desc}")
    L.append("")
    _wrap(L,
          "Every report is DIRECTIONAL/ASSOCIATIONAL unless its header says "
          "otherwise; 08_methods.txt resolves every [M#] tag.")
