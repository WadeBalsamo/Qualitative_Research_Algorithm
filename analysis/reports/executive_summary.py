"""
analysis/reports/executive_summary.py
--------------------------------------
Deterministic program-improvement brief — the top-level synthesis that ties the
whole analysis together into a decision-ready document. No LLM, no recomputation:
it reads the machine-readable artifacts other stages already wrote and renders
them in the curriculum-modification structure of methodology §6.3:

    Stage Distribution Summary → Avoidance-Barrier Assessment →
    Therapist Behavior (BOTH directions) → Candidate Recommendations
    (Observation + Mechanism Hypothesis + Proposed Change + How-to-Assess) →
    Validation Caveats.

Every claim is flagged directional/associational and carries the human-validation
caveat verbatim from the methodology, so the brief never overstates what the
computational pipeline can support.
"""

import json
import os
from datetime import date
from typing import Optional

import pandas as pd

from process import output_paths as _paths


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


def _hr(L, title=None):
    L.append("-" * 78)
    if title:
        L.append(title)
        L.append("-" * 78)


def generate_executive_summary(output_dir, df=None, framework=None, df_all=None) -> Optional[str]:
    """Write 06_reports/00_executive_summary.txt. Returns the path."""
    eff_dir = _paths.efficacy_dir(output_dir)
    mech_dir = _paths.mechanism_dir(output_dir)
    data_dir = _paths.analysis_data_dir(output_dir)

    eff = _load_json(os.path.join(eff_dir, 'efficacy_summary.json'))
    longitudinal = _load_json(os.path.join(data_dir, 'longitudinal_summary.json'))
    group_traj = _load_csv(os.path.join(eff_dir, 'group_progression_trajectory.csv'))
    barrier = _load_csv(os.path.join(eff_dir, 'barrier_crossing.csv'))
    linkage = _load_csv(os.path.join(eff_dir, 'external_outcome_linkage.csv'))
    mech = _load_csv(os.path.join(mech_dir, 'mechanism_delta_progression.csv'))
    gnn_ran = os.path.isdir(_paths.gnn_data_dir(output_dir)) and bool(
        os.path.isdir(_paths.gnn_data_dir(output_dir)) and os.listdir(_paths.gnn_data_dir(output_dir)))

    L = []
    L.append("=" * 78)
    L.append("PROGRAM-IMPROVEMENT EXECUTIVE SUMMARY")
    L.append("=" * 78)
    L.append(f"Generated: {date.today().isoformat()}")
    L.append("")
    L.append("A deterministic synthesis of the analysis outputs, organized for between-cohort")
    L.append("curriculum decisions (methodology §6.3). Everything here is DIRECTIONAL and")
    L.append("ASSOCIATIONAL — naturalistic observation of language, not causal proof. See the")
    L.append("Validation Caveats at the end before acting on any recommendation, and the")
    L.append("linked detailed reports for confidence intervals, significance, and exemplars.")
    L.append("")

    # ── 1. Stage distribution / primary outcome ──────────────────────────────
    _hr(L, "1. PROGRESSION OVER SESSIONS (descriptive, single-arm — NOT an efficacy claim)")
    L.append("  Describes how participants' LLM-coded VAAMR language moves across sessions.")
    L.append("  No control arm; the 'outcome' is the coded language itself (also shaped by")
    L.append("  therapist prompting). Hypothesis-generating, not proof of program benefit.")
    if longitudinal and isinstance(longitudinal.get('group_progression'), dict):
        gp = longitudinal['group_progression']
        L.append(f"  Participants: advancing {gp.get('n_advancing', '?')}, "
                 f"stable {gp.get('n_stable', '?')}, regressing {gp.get('n_regressing', '?')}.")
    if eff:
        # PRIMARY: ordinal-safe adaptive-stage occupancy + Mann-Kendall.
        a_first, a_last = eff.get('adaptive_first_mean'), eff.get('adaptive_last_mean')
        if a_first is not None and a_last is not None:
            L.append(f"  Adaptive-stage (2–4) occupancy: {a_first*100:.0f}% → {a_last*100:.0f}% across sessions.")
        mk = eff.get('mk_adaptive_occupancy') or {}
        if mk.get('n', 0) >= 3:
            p = mk.get('p_value')
            p_s = f"p={p:.4f}" if isinstance(p, (int, float)) and p == p else "p=n/a"
            L.append(f"  Monotonic trend (Mann–Kendall, ordinal-safe): {mk.get('direction', '?')}, {p_s}.")
        # SECONDARY: interval-scale slope, explicitly demoted.
        t = eff.get('trend_interval_sensitivity') or {}
        slope = t.get('slope')
        if isinstance(slope, (int, float)) and slope == slope:
            p = t.get('p_value'); p_s = f"p={p:.4f}" if isinstance(p, (int, float)) and p == p else "p=n/a"
            L.append(f"  [sensitivity, equal-spacing assumed] E[stage] slope={slope:+.4f}/session ({p_s}).")
        if eff.get('underpowered'):
            L.append(f"  ⚠ {eff.get('power_note', 'Underpowered sample — inference indicative only.')}")
        if str(eff.get('mixture_source', '')).lower().startswith('gnn'):
            L.append("  ⚠ Stage mixtures are GNN-derived (validated vs LLM, not yet vs human coding).")
    elif group_traj is not None and len(group_traj):
        L.append(f"  Group mean E[stage]: session {int(group_traj.iloc[0]['session_number'])} "
                 f"= {group_traj.iloc[0]['mean']:+.2f} → session {int(group_traj.iloc[-1]['session_number'])} "
                 f"= {group_traj.iloc[-1]['mean']:+.2f} (interval-scale; see progression_summary.txt).")
    else:
        L.append("  (progression outputs not found — run analysis with superposition enabled.)")
    if longitudinal and isinstance(longitudinal.get('feasibility_assessment'), dict):
        fa = longitudinal['feasibility_assessment']
        L.append(f"  Classification feasibility: {fa.get('feasibility_rating', '?')} "
                 f"({100 * fa.get('high_plus_medium_pct', 0):.0f}% high+medium confidence).")
    L.append("  Full detail: 01_outcomes/progression_summary.txt")
    L.append("")

    # ── 2. Avoidance barrier ─────────────────────────────────────────────────
    _hr(L, "2. AVOIDANCE-BARRIER ASSESSMENT (the central developmental challenge)")
    if eff and eff.get('barrier_total'):
        L.append(f"  Crossed Avoidance → Attention-Regulation: {eff.get('barrier_crossed', 0)}/"
                 f"{eff.get('barrier_total', 0)} participants.")
    elif barrier is not None and len(barrier):
        crossed = int(barrier['crossed_to_attention_regulation'].sum())
        L.append(f"  Crossed Avoidance → Attention-Regulation: {crossed}/{len(barrier)} participants.")
    else:
        L.append("  (barrier-crossing data not found.)")
    L.append("  Full per-participant detail + cusp-density trend: 01_outcomes/avoidance_barrier.txt")
    L.append("")

    # ── 3. Therapist behavior, both directions ───────────────────────────────
    _hr(L, "3. THERAPIST BEHAVIOR — what moves participants (forward AND backward)")
    if mech is not None and len(mech) and 'mean_delta_prog' in mech.columns:
        m = mech.copy()
        if 'fdr_significant' in m.columns and m['fdr_significant'].any():
            m = m[m['fdr_significant']]
        fwd = m.sort_values('mean_delta_prog', ascending=False).head(5)
        bwd = m[m['mean_delta_prog'] < 0].sort_values('mean_delta_prog', ascending=True).head(5)
        L.append("  Most FORWARD-associated therapist moves (Δprogression > 0):")
        for _, r in fwd.iterrows():
            L.append(f"    {str(r.get('from_stage_name', '')):<18} {str(r['behavior'])[:24]:<24} "
                     f"Δ={r['mean_delta_prog']:+.3f} (n={int(r['n'])})")
        L.append("  Most BACKWARD/STALLING-associated therapist moves (Δprogression < 0):")
        if len(bwd):
            for _, r in bwd.iterrows():
                L.append(f"    {str(r.get('from_stage_name', '')):<18} {str(r['behavior'])[:24]:<24} "
                         f"Δ={r['mean_delta_prog']:+.3f} (n={int(r['n'])})")
        else:
            L.append("    (none with negative mean Δprogression met the count threshold)")
        L.append("  Concrete FROM→CUE→TO exemplars: 02_mechanism/language_atlas.txt;")
        L.append("  CIs / permutation p / FDR: 02_mechanism/mechanism.txt.")
    else:
        L.append("  (mechanism Δprogression table not found — needs PURER labels + superposition.)")
    L.append("")

    # ── 4. Convergent validity vs external outcomes ──────────────────────────
    _hr(L, "4. CONVERGENT VALIDITY vs EXTERNAL CLINICAL OUTCOMES (exploratory)")
    L.append("  The only route to a real-world claim: does the coded-language trajectory")
    L.append("  CORRELATE with measured outcomes? Correlation is convergent validity, not efficacy.")
    if linkage is not None and len(linkage):
        for _, r in linkage.iterrows():
            rv = r.get('pearson_r')
            stat = (f"r={rv:+.3f} p={r.get('p_value'):.3f}"
                    if isinstance(rv, (int, float)) and rv == rv else "r=n/a")
            L.append(f"    {str(r.get('vaamr_measure', '')):<10} ↔ {str(r.get('external_measure', ''))[:24]:<24} "
                     f"{stat} (n={r.get('n')})")
        L.append("  Positive r ⇒ participants whose language advances more also improve clinically.")
    else:
        L.append("  STATUS: external outcomes NOT yet integrated. No real-world claim can be made.")
        L.append("  Integration plan (REDCap → 02_meta/outcomes.csv): docs/OUTCOME_INTEGRATION_ROADMAP.md")
    L.append("")

    # ── 5. Candidate recommendations ─────────────────────────────────────────
    _hr(L, "5. CANDIDATE RECOMMENDATIONS (directional — require human review)")
    L.append("  Each follows: Observation → Mechanism Hypothesis → Proposed Change → How to Assess.")
    L.append("  These are generated mechanically from the patterns above as starting points for")
    L.append("  the curriculum team's MoSCoW prioritization — NOT automated decisions.")
    L.append("")
    recs = _build_recommendations(eff, mech, barrier, longitudinal)
    if recs:
        for i, rec in enumerate(recs, 1):
            L.append(f"  R{i}. Observation:  {rec['observation']}")
            L.append(f"      Mechanism:    {rec['mechanism']}")
            L.append(f"      Proposed:     {rec['change']}")
            L.append(f"      Assess in C3–4: {rec['assess']}")
            L.append("")
    else:
        L.append("  (insufficient data to template recommendations — populate efficacy & mechanism first)")
        L.append("")

    # ── 6. Validation caveats ────────────────────────────────────────────────
    _hr(L, "6. VALIDATION CAVEATS (read before acting)")
    L.append("  • QRA classifies what participants SAY, not what they experience; linguistic")
    L.append("    expression is not phenomenological state (methodology §9.1).")
    L.append("  • Cue-response evidence is temporal-adjacency association, NOT causal mechanism (§9.2).")
    L.append("  • Production labels are single-model multi-run consensus (stability, not")
    L.append("    independence-based reliability); human validation carries the burden of")
    L.append("    establishing correctness (§5.3–5.4).")
    if longitudinal and isinstance(longitudinal.get('validity_indicators'), dict):
        vn = longitudinal['validity_indicators'].get('validity_narrative')
        if vn:
            L.append(f"  • {vn}")
    L.append("  • Recommendations above are directional hypotheses for Cohort 3–4 testing, not")
    L.append("    conclusions; high-confidence + human-validated findings carry more weight.")
    if not gnn_ran:
        L.append("  • GNN discovery layer did not produce outputs this run — motif/coupling")
        L.append("    findings (06_gnn/, language_atlas §2–3) are absent. Enable/inspect the GNN layer.")
    L.append("")
    L.append("See 00_READ_ME.txt for the full report map and 07_methods_appendix.txt for how")
    L.append("each metric is computed.")

    path = _paths.executive_summary_path(output_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path


def _build_recommendations(eff, mech, barrier, longitudinal):
    """Template directional recommendations from the computed patterns."""
    recs = []

    # Avoidance barrier under-crossed.
    crossed = total = None
    if eff and eff.get('barrier_total'):
        crossed, total = eff.get('barrier_crossed', 0), eff.get('barrier_total', 0)
    elif barrier is not None and len(barrier):
        crossed, total = int(barrier['crossed_to_attention_regulation'].sum()), len(barrier)
    if total and crossed is not None and crossed < total:
        # Find the strongest forward mover OUT of Avoidance to ground the proposal.
        mover = None
        if mech is not None and 'from_stage_name' in mech.columns:
            av = mech[mech['from_stage_name'].astype(str).str.contains('Avoid', case=False, na=False)]
            av = av[av['mean_delta_prog'] > 0].sort_values('mean_delta_prog', ascending=False)
            if len(av):
                mover = str(av.iloc[0]['behavior'])
        recs.append({
            'observation': f"Only {crossed}/{total} participants crossed the Avoidance→Attention-Regulation barrier.",
            'mechanism': ("Forward movement out of Avoidance is most associated with "
                          + (f"'{mover}'." if mover else "specific therapist moves (see mechanism table).")),
            'change': ("Increase deliberate use of the forward-associated move(s) at the practice "
                       "debrief in the sessions where Avoidance dominates."),
            'assess': "Compare Cohort 3–4 Avoidance prevalence and crossing rate against Cohorts 1–2.",
        })

    # Flat / non-increasing group trend (ordinal-safe: use Mann-Kendall direction).
    mk = (eff or {}).get('mk_adaptive_occupancy') or {}
    if mk.get('n', 0) >= 3 and mk.get('direction') in ('flat', 'decreasing'):
        recs.append({
            'observation': f"Adaptive-stage occupancy shows a {mk.get('direction')} monotonic trend across sessions.",
            'mechanism': "Forward shifts in coded language are not consolidating session to session.",
            'change': "Add explicit between-session consolidation (home-practice review tying prior insight to current session).",
            'assess': "Re-run the Mann–Kendall adaptive-occupancy trend in Cohort 3–4 and compare direction.",
        })

    # A strong backward mover worth flagging.
    if mech is not None and len(mech) and 'mean_delta_prog' in mech.columns:
        bwd = mech[mech['mean_delta_prog'] < 0].sort_values('mean_delta_prog', ascending=True)
        if len(bwd):
            r = bwd.iloc[0]
            recs.append({
                'observation': (f"From {r.get('from_stage_name', '?')}, '{str(r['behavior'])[:24]}' co-occurs with "
                                f"backward movement (Δ={r['mean_delta_prog']:+.3f}, n={int(r['n'])})."),
                'mechanism': "This move/pattern may be mistimed for that starting stage (§7.6 stage-moderation).",
                'change': "Review exemplars in language_atlas.txt §1b; consider re-sequencing or reframing this move at that stage.",
                'assess': "Track whether reducing it at that stage raises forward-transition rate in Cohort 3–4.",
            })

    return recs[:5]
