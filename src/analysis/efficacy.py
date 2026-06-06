"""
analysis/efficacy.py
--------------------
Program-efficacy dossier — the "does it work" deliverable.

Answers, with uncertainty, whether participants move toward adaptive VAAMR stages
over the program, and whether that linguistic movement tracks real-world clinical
change. Primary outcome is the continuous progression coordinate (E[stage]);
secondary outcomes are adaptive-stage occupancy (stages 2–4), the
Avoidance→Attention-Regulation barrier crossing + first-passage, and maladaptive
dwell. Every estimate carries a participant-cluster bootstrap CI and a
mixed-effects trend test (random participant effect). External clinical outcomes,
when provided, are merged and correlated against within-program progression.

Observational design ⇒ associations, not causal effects. Hard labels are used for
occupancy (labeler-of-record); the continuous coordinate is used for trajectory.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from process import output_paths as _paths
from .loader import sort_session_ids
from . import stats as S


# ---------------------------------------------------------------------------
# External clinical outcomes (schema-flexible ingestion)
# ---------------------------------------------------------------------------

def load_external_outcomes(output_dir: str, config=None) -> Optional[dict]:
    """Load external clinical outcomes if present. Returns {mode, data, measures,...} or None.

    Auto-detects two layouts (keyed by ``participant_id``):
      - WIDE pre/post: columns ``<measure>_pre`` / ``<measure>_post`` → change scores.
      - LONG per-session: ``participant_id, session_id|session_number|timepoint, <measures…>``.
    Path comes from ``config.outcomes_path`` (absolute, or relative to output_dir),
    defaulting to ``02_meta/outcomes.csv``. Returns None when absent/unreadable.
    """
    default = os.path.join(_paths.meta_dir(output_dir), 'outcomes.csv')
    path = getattr(config, 'outcomes_path', None) if config else None
    if path:
        path = path if os.path.isabs(path) else os.path.join(output_dir, path)
    else:
        path = default
    if not os.path.isfile(path):
        return None
    try:
        odf = pd.read_csv(path)
    except Exception:
        return None
    if 'participant_id' not in odf.columns:
        return None
    odf['participant_id'] = odf['participant_id'].astype(str)

    pre_cols = [c for c in odf.columns if c.endswith('_pre')]
    measures = [c[:-4] for c in pre_cols if f'{c[:-4]}_post' in odf.columns]
    if measures:                                   # WIDE pre/post
        for m in measures:
            odf[f'{m}_change'] = pd.to_numeric(odf[f'{m}_post'], errors='coerce') - \
                                 pd.to_numeric(odf[f'{m}_pre'], errors='coerce')
        return {'mode': 'wide', 'data': odf, 'measures': measures,
                'change_cols': [f'{m}_change' for m in measures]}

    time_col = next((c for c in ('session_id', 'session_number', 'timepoint') if c in odf.columns), None)
    value_cols = [c for c in odf.columns
                  if c not in ('participant_id', time_col) and pd.api.types.is_numeric_dtype(odf[c])]
    return {'mode': 'long', 'data': odf, 'measures': value_cols,
            'time_col': time_col, 'change_cols': []}


# ---------------------------------------------------------------------------
# Internal VAAMR outcomes
# ---------------------------------------------------------------------------

def _adaptive_sets(config):
    adaptive = set(getattr(config, 'adaptive_stages', [2, 3, 4]) if config else [2, 3, 4])
    maladaptive = set(getattr(config, 'maladaptive_stages', [0, 1]) if config else [0, 1])
    barrier_from = getattr(config, 'barrier_from', 1) if config else 1
    barrier_to = getattr(config, 'barrier_to', 2) if config else 2
    return adaptive, maladaptive, barrier_from, barrier_to


def compute_participant_session_outcomes(df: pd.DataFrame, config=None) -> pd.DataFrame:
    """Per (participant, session): progression coordinate + adaptive/maladaptive occupancy."""
    adaptive, maladaptive, _, _ = _adaptive_sets(config)
    has_prog = 'progression_coord' in df.columns
    rows = []
    for (pid, sid), g in df.groupby(['participant_id', 'session_id']):
        n = len(g)
        if n == 0:
            continue
        labels = g['final_label'].dropna().astype(int)
        adapt = float(labels.isin(adaptive).mean()) if len(labels) else float('nan')
        malad = float(labels.isin(maladaptive).mean()) if len(labels) else float('nan')
        prog = float(g['progression_coord'].dropna().mean()) if has_prog and g['progression_coord'].notna().any() \
            else float(labels.mean()) if len(labels) else float('nan')
        rows.append({
            'participant_id': str(pid),
            'session_id': sid,
            'session_number': int(g['session_number'].iloc[0]),
            'n_segments': n,
            'progression_coord': round(prog, 4),
            'adaptive_occupancy': round(adapt, 4),
            'maladaptive_occupancy': round(malad, 4),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(['participant_id', 'session_number']).reset_index(drop=True)
    return out


def compute_barrier_crossing(df: pd.DataFrame, config=None) -> pd.DataFrame:
    """Per participant: first-passage to Attention-Regulation and whether the barrier is crossed.

    A participant 'crosses' when, after expressing the barrier-from stage (Avoidance),
    a later session's dominant/any segment reaches the barrier-to stage (Attention-
    Regulation). Reports the first session index at which barrier-to is reached.
    """
    _, _, b_from, b_to = _adaptive_sets(config)
    rows = []
    for pid, g in df.groupby('participant_id'):
        sids = sort_session_ids(g['session_id'].unique().tolist())
        first_to = None
        ever_from = False
        for idx, sid in enumerate(sids):
            labels = g[g['session_id'] == sid]['final_label'].dropna().astype(int)
            if labels.isin([b_from]).any():
                ever_from = True
            if first_to is None and (labels >= b_to).any():
                first_to = idx
        rows.append({
            'participant_id': str(pid),
            'n_sessions': len(sids),
            'expressed_barrier_from': bool(ever_from),
            'crossed_to_attention_regulation': bool(first_to is not None),
            'first_passage_session_index': first_to if first_to is not None else None,
        })
    return pd.DataFrame(rows)


def compute_group_trajectory(ps_outcomes: pd.DataFrame, outcome: str = 'progression_coord') -> pd.DataFrame:
    """Group mean of an outcome by session_number with participant-cluster bootstrap CI."""
    rows = []
    for snum, g in ps_outcomes.groupby('session_number'):
        vals = g[outcome].to_numpy(dtype=float)
        clusters = g['participant_id'].to_numpy()
        ci = S.cluster_bootstrap_ci(vals, clusters, statistic=np.nanmean, n_boot=1000)
        rows.append({
            'session_number': int(snum),
            'n_participants': int(g['participant_id'].nunique()),
            'mean': round(ci['point'], 4) if ci['point'] == ci['point'] else None,
            'ci_lo': round(ci['lo'], 4) if ci['lo'] == ci['lo'] else None,
            'ci_hi': round(ci['hi'], 4) if ci['hi'] == ci['hi'] else None,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values('session_number').reset_index(drop=True)
    return out


def compute_participant_slopes(ps_outcomes: pd.DataFrame, outcome: str = 'progression_coord') -> pd.DataFrame:
    """Per-participant OLS slope of an outcome over session_number (with CI/p via linregress)."""
    rows = []
    for pid, g in ps_outcomes.groupby('participant_id'):
        g = g.dropna(subset=[outcome]).sort_values('session_number')
        if len(g) < 2:
            rows.append({'participant_id': str(pid), 'slope': None, 'n_sessions': len(g)})
            continue
        res = S.mixedlm_trend(g.assign(_grp='x'), outcome, 'session_number', '_grp')  # OLS path (single grp)
        rows.append({
            'participant_id': str(pid),
            'n_sessions': len(g),
            'slope': round(res['slope'], 4) if res['slope'] == res['slope'] else None,
            'endpoint': round(float(g[outcome].iloc[-1]), 4),
            'baseline': round(float(g[outcome].iloc[0]), 4),
            'change': round(float(g[outcome].iloc[-1] - g[outcome].iloc[0]), 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# External linkage
# ---------------------------------------------------------------------------

def link_to_external(participant_slopes: pd.DataFrame, outcomes: Optional[dict]) -> pd.DataFrame:
    """Correlate within-program progression (slope/change/endpoint) with external change scores."""
    if outcomes is None or outcomes.get('mode') != 'wide' or not outcomes.get('change_cols'):
        return pd.DataFrame()
    from scipy import stats as sps
    ext = outcomes['data'][['participant_id'] + outcomes['change_cols']].copy()
    merged = participant_slopes.merge(ext, on='participant_id', how='inner')
    rows = []
    for vaamr_col in ('slope', 'change', 'endpoint'):
        if vaamr_col not in merged.columns:
            continue
        for ch in outcomes['change_cols']:
            d = merged[[vaamr_col, ch]].dropna()
            if len(d) < 3:
                continue
            r, p = sps.pearsonr(d[vaamr_col], d[ch])
            ci = S.cluster_bootstrap_ci(
                (d[vaamr_col].to_numpy() - d[vaamr_col].mean()) * (d[ch].to_numpy() - d[ch].mean()),
                d.index.to_numpy(), statistic=np.mean, n_boot=1000)
            rows.append({
                'vaamr_measure': vaamr_col,
                'external_measure': ch,
                'n': len(d),
                'pearson_r': round(float(r), 4),
                'p_value': round(float(p), 4),
                'covariance_ci_lo': round(ci['lo'], 4) if ci['lo'] == ci['lo'] else None,
                'covariance_ci_hi': round(ci['hi'], 4) if ci['hi'] == ci['hi'] else None,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestrator + report + figure
# ---------------------------------------------------------------------------

def run_efficacy_analysis(df: pd.DataFrame, framework: dict, output_dir: str, config=None) -> dict:
    """Compute the efficacy dossier, write CSVs + report + figure. Returns {files_written}."""
    files: List[str] = []
    eff_dir = _paths.efficacy_dir(output_dir)
    os.makedirs(eff_dir, exist_ok=True)

    ps = compute_participant_session_outcomes(df, config)
    if ps.empty:
        return {'files_written': files}
    ps.to_csv(os.path.join(eff_dir, 'participant_session_outcomes.csv'), index=False)
    files.append(os.path.join(eff_dir, 'participant_session_outcomes.csv'))

    group_prog = compute_group_trajectory(ps, 'progression_coord')
    group_adapt = compute_group_trajectory(ps, 'adaptive_occupancy')
    group_prog.to_csv(os.path.join(eff_dir, 'group_progression_trajectory.csv'), index=False)
    files.append(os.path.join(eff_dir, 'group_progression_trajectory.csv'))

    slopes = compute_participant_slopes(ps, 'progression_coord')
    slopes.to_csv(os.path.join(eff_dir, 'participant_progression_slopes.csv'), index=False)
    files.append(os.path.join(eff_dir, 'participant_progression_slopes.csv'))

    barrier = compute_barrier_crossing(df, config)
    barrier.to_csv(os.path.join(eff_dir, 'barrier_crossing.csv'), index=False)
    files.append(os.path.join(eff_dir, 'barrier_crossing.csv'))

    # PRIMARY (ordinal-safe): Mann–Kendall monotonic trend on the per-session group
    # series of adaptive-stage (2–4) occupancy. Uses only the sign of pairwise
    # differences, so it does NOT assume the VAAMR stages are equally spaced.
    _adapt_ordered = group_adapt.sort_values('session_number')['mean'].tolist()
    mk_adapt = S.mann_kendall_trend(_adapt_ordered)
    _prog_ordered = group_prog.sort_values('session_number')['mean'].tolist()
    mk_prog = S.mann_kendall_trend(_prog_ordered)

    # SECONDARY (sensitivity, interval-scale): mixed-effects slope on E[stage]. This
    # treats the ordinal coordinate as cardinal — reported as a caveated sensitivity
    # analysis, not the headline.
    trend = S.mixedlm_trend(ps, 'progression_coord', 'session_number', 'participant_id')

    # Sign test on advancing participants (per-participant slope direction).
    valid_slopes = slopes['slope'].dropna()
    n_adv = int((valid_slopes > 0).sum())
    sign = S.sign_test(n_adv, int(len(valid_slopes)))

    # Sample-size guard (flag, do NOT suppress).
    n_part = int(ps['participant_id'].nunique())
    n_sess = int(ps['session_number'].nunique())
    power = S.power_flag(n_part, n_sess)

    # Which substrate produced the stage mixtures feeding all of the above?
    try:
        from .superposition import dominant_source
        mixture_provenance = dominant_source(df)
    except Exception:
        mixture_provenance = 'unknown'

    outcomes = load_external_outcomes(output_dir, config)
    linkage = link_to_external(slopes, outcomes)
    if not linkage.empty:
        linkage.to_csv(os.path.join(eff_dir, 'external_outcome_linkage.csv'), index=False)
        files.append(os.path.join(eff_dir, 'external_outcome_linkage.csv'))

    rep = _write_progression_report(group_prog, group_adapt, slopes, barrier, trend, sign,
                                     mk_adapt, mk_prog, power, mixture_provenance,
                                     outcomes, linkage, framework, output_dir)
    files.append(rep)

    # Machine-readable headline stats so downstream synthesis (the executive
    # summary) reads clean numbers instead of parsing the prose report.
    try:
        import json as _json
        crossed = int(barrier['crossed_to_attention_regulation'].sum()) if len(barrier) else 0
        _clean = lambda d: {k: (None if (isinstance(v, float) and v != v) else v) for k, v in d.items()}
        summary = {
            'mk_adaptive_occupancy': _clean(mk_adapt),   # PRIMARY ordinal trend
            'mk_progression_coord': _clean(mk_prog),
            'trend_interval_sensitivity': _clean(trend),  # SECONDARY (equal-spacing assumed)
            'sign_test': _clean(sign),
            'n_advancing': n_adv,
            'n_participants': n_part,
            'n_sessions': n_sess,
            'underpowered': power['underpowered'],
            'power_note': power['note'],
            'mixture_source': mixture_provenance,
            'n_participants_with_slope': int(len(valid_slopes)),
            'barrier_crossed': crossed,
            'barrier_total': int(len(barrier)),
            'adaptive_first_mean': (float(group_adapt.iloc[0]['mean']) if len(group_adapt) else None),
            'adaptive_last_mean': (float(group_adapt.iloc[-1]['mean']) if len(group_adapt) else None),
            'group_first_mean': (float(group_prog.iloc[0]['mean']) if len(group_prog) else None),
            'group_last_mean': (float(group_prog.iloc[-1]['mean']) if len(group_prog) else None),
        }
        sp = os.path.join(eff_dir, 'efficacy_summary.json')
        with open(sp, 'w', encoding='utf-8') as _f:
            _json.dump(summary, _f, indent=2)
        files.append(sp)
    except Exception as _e:
        print(f"  Warning: efficacy_summary.json failed: {_e}")

    try:
        fig = _plot_efficacy(group_prog, group_adapt, barrier, linkage, output_dir)
        if fig:
            files.append(fig)
    except Exception as e:
        print(f"  Warning: efficacy figure failed: {e}")

    return {'files_written': files, 'trend': trend, 'n_advancing': n_adv}


def _write_progression_report(group_prog, group_adapt, slopes, barrier, trend, sign,
                              mk_adapt, mk_prog, power, mixture_provenance,
                              outcomes, linkage, framework, output_dir) -> str:
    pflag = power.get('note', '')
    L = []
    L.append("=" * 78)
    L.append("PROGRAM PROGRESSION SUMMARY  (descriptive, single-arm)")
    L.append("=" * 78)
    L.append("")
    L.append("WHAT THIS IS — and is NOT. This is a DESCRIPTIVE summary of how participants'")
    L.append("LLM-coded VAAMR language moves across the program. It is NOT an efficacy")
    L.append("estimate: there is no control arm, no randomized comparison here, and the")
    L.append("'outcome' is the same coded language being analyzed — which is also shaped by")
    L.append("therapist prompting (methodology §9.4). Read every number as hypothesis-")
    L.append("generating for human validation and Cohort 3–4 replication, never as proof")
    L.append("that the program caused clinical benefit.")
    L.append("This is an observational, single-arm design: every relationship reported")
    L.append("here is associational, not causal.")
    L.append("")
    L.append("VAAMR is an ORDINAL developmental typology. The PRIMARY trend below uses a")
    L.append("rank-based Mann–Kendall test (no equal-spacing assumption). The E[stage]")
    L.append("'progression coordinate' and its linear slope appear only as a clearly-")
    L.append("labeled interval-scale SENSITIVITY analysis.")
    L.append(f"Stage-mixture substrate: {mixture_provenance}.")
    if str(mixture_provenance).lower().startswith('gnn'):
        L.append("  ⚠ Mixtures are GNN-derived. The GNN is validated against LLM consensus,")
        L.append("    NOT yet against human coding — do not treat as more reliable than the LLM.")
    if pflag:
        L.append(f"  ⚠ {pflag}")
    L.append("")

    L.append("-" * 78)
    L.append("1. ADAPTIVE-STAGE OCCUPANCY OVER SESSIONS (primary, ordinal-safe)")
    L.append("-" * 78)
    L.append("  Proportion of each session's participant segments coded in adaptive stages")
    L.append("  (Attention-Regulation/Metacognition/Reappraisal, 2–4):")
    for _, r in group_adapt.iterrows():
        ci = (f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]" if r['ci_lo'] is not None else "[n/a]")
        L.append(f"  session {int(r['session_number']):<3} {r['mean']*100:5.1f}% {ci}  (n={r['n_participants']})")
    if mk_adapt.get('n', 0) >= 3:
        tau = mk_adapt.get('tau')
        tau_s = f"{tau:+.3f}" if isinstance(tau, (int, float)) and tau == tau else "n/a"
        L.append(f"\n  Monotonic trend (Mann–Kendall): {mk_adapt['direction']}, "
                 f"τ={tau_s}, Sen slope={mk_adapt.get('sen_slope'):+.4f}/session, "
                 f"p={mk_adapt.get('p_value'):.4f} (n={mk_adapt['n']} sessions).")
    else:
        L.append("\n  Monotonic trend: not estimable (need ≥3 sessions with data).")
    if pflag:
        L.append(f"  ⚠ {pflag}")
    L.append("")

    L.append("-" * 78)
    L.append("2. E[stage] PROGRESSION COORDINATE  (SENSITIVITY — interval-scale assumed)")
    L.append("-" * 78)
    L.append("  ⚠ Treats VAAMR 0–4 as equally spaced (Vigilance→Avoidance == Metacog→Reappraisal).")
    L.append("    Provided for comparison only; the ordinal trend above is the headline.")
    for _, r in group_prog.iterrows():
        ci = (f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]" if r['ci_lo'] is not None else "[n/a]")
        L.append(f"  session {int(r['session_number']):<3} mean={r['mean']:+.3f} {ci}  (n={r['n_participants']})")
    if trend['slope'] == trend['slope']:
        L.append(f"\n  Linear trend ({trend['method']}): slope = {trend['slope']:+.4f}/session "
                 f"[{trend['ci_lo']:+.4f}, {trend['ci_hi']:+.4f}], p = "
                 f"{trend['p_value']:.4f} (n={trend['n']} obs, {trend['n_groups']} participants).")
    else:
        L.append("\n  Linear trend: not estimable.")
    mk_p = mk_prog.get('p_value')
    if mk_prog.get('n', 0) >= 3:
        L.append(f"  Mann–Kendall on E[stage] (rank check): {mk_prog['direction']}, "
                 f"p={mk_p:.4f}." if isinstance(mk_p, (int, float)) and mk_p == mk_p else
                 f"  Mann–Kendall on E[stage]: {mk_prog['direction']}.")
    if pflag:
        L.append(f"  ⚠ {pflag}")
    L.append("")

    L.append("-" * 78)
    L.append("3. PER-PARTICIPANT TRAJECTORY DIRECTION")
    L.append("-" * 78)
    for _, r in slopes.iterrows():
        if r.get('slope') is None:
            L.append(f"  {r['participant_id']:<16} (only {int(r['n_sessions'])} session — slope n/a)")
        else:
            L.append(f"  {r['participant_id']:<16} slope={r['slope']:+.3f}  "
                     f"baseline={r.get('baseline', float('nan')):+.2f} → endpoint={r.get('endpoint', float('nan')):+.2f} "
                     f"(Δ={r.get('change', float('nan')):+.2f})")
    L.append(f"\n  Advancing (slope>0): {sign['n_positive']}/{sign['n_total']} participants; "
             f"sign-test p = {sign['p_value']:.4f}." if sign['p_value'] == sign['p_value']
             else f"\n  Advancing: {sign['n_positive']}/{sign['n_total']}.")
    L.append("  (Per-participant slopes inherit the interval-scale caveat from §2.)")
    L.append("")

    L.append("-" * 78)
    L.append("4. AVOIDANCE → ATTENTION-REGULATION BARRIER (language-internal)")
    L.append("-" * 78)
    L.append("  'Crossing' = first session in which Avoidance is no longer the participant's")
    L.append("  dominant coded stage. A within-coding-scheme transition of LANGUAGE, shaped")
    L.append("  by therapist prompting — not a verified clinical milestone.")
    crossed = int(barrier['crossed_to_attention_regulation'].sum())
    L.append(f"  Dominant stage left Avoidance: {crossed}/{len(barrier)} participants.")
    for _, r in barrier.iterrows():
        fp = r['first_passage_session_index']
        fp_s = f"session #{int(fp) + 1}" if fp is not None and fp == fp else "—"
        L.append(f"    {r['participant_id']:<16} crossed={str(bool(r['crossed_to_attention_regulation'])):<5} first-passage={fp_s}")
    L.append("")

    L.append("-" * 78)
    L.append("5. CONVERGENT VALIDITY vs EXTERNAL CLINICAL OUTCOMES (exploratory)")
    L.append("-" * 78)
    L.append("  The ONLY place this analysis can speak to real-world change: does the coded-")
    L.append("  language trajectory CORRELATE with measured clinical outcomes? Correlation is")
    L.append("  convergent-validity evidence, still NOT efficacy.")
    if outcomes is None:
        L.append("  STATUS: no external outcomes integrated yet (expected at 02_meta/outcomes.csv).")
        L.append("  See docs/OUTCOME_INTEGRATION_ROADMAP.md for the REDCap → outcomes.csv plan.")
    elif linkage is None or linkage.empty:
        L.append(f"  External outcomes loaded ({outcomes['mode']}, measures: "
                 f"{', '.join(outcomes.get('measures', [])) or 'none'}), but too few matched")
        L.append("  participants for correlation (need ≥3 with both VAAMR slope and outcome change).")
    else:
        L.append("  Pearson correlation of within-program VAAMR progression vs external change:")
        for _, r in linkage.iterrows():
            rv = r['pearson_r']
            stat = (f"r={rv:+.3f}  p={r['p_value']:.3f}" if isinstance(rv, (int, float)) and rv == rv
                    else "r=n/a (no variance among matched participants)")
            L.append(f"    {r['vaamr_measure']:<10} ↔ {str(r['external_measure'])[:24]:<24} "
                     f"{stat}  (n={r['n']})")
        L.append("  Positive r ⇒ participants whose language advances more also improve clinically.")
        L.append("  Exploratory/associational; small N — convergent-validity evidence, not efficacy.")
    L.append("")

    rep_dir = _paths.reports_outcomes_dir(output_dir)
    os.makedirs(rep_dir, exist_ok=True)
    path = os.path.join(rep_dir, 'progression_summary.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path


def _plot_efficacy(group_prog, group_adapt, barrier, linkage, output_dir):
    """Multipanel efficacy figure. Returns PNG path or None."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Panel 1: group progression w/ CI band.
    ax = axes[0]
    if not group_prog.empty:
        x = group_prog['session_number'].to_numpy()
        m = group_prog['mean'].to_numpy(dtype=float)
        lo = group_prog['ci_lo'].to_numpy(dtype=float)
        hi = group_prog['ci_hi'].to_numpy(dtype=float)
        ax.fill_between(x, lo, hi, color='#648FFF', alpha=0.2)
        ax.plot(x, m, '-o', color='#648FFF', lw=2)
    ax.set_title('Group progression (E[stage]) ± 95% CI')
    ax.set_xlabel('Session number'); ax.set_ylabel('Progression coordinate')

    # Panel 2: adaptive occupancy w/ CI.
    ax = axes[1]
    if not group_adapt.empty:
        x = group_adapt['session_number'].to_numpy()
        m = group_adapt['mean'].to_numpy(dtype=float)
        lo = group_adapt['ci_lo'].to_numpy(dtype=float)
        hi = group_adapt['ci_hi'].to_numpy(dtype=float)
        ax.fill_between(x, lo, hi, color='#3AB75D', alpha=0.2)
        ax.plot(x, m, '-o', color='#3AB75D', lw=2)
    ax.set_ylim(0, 1)
    ax.set_title('Adaptive-stage (2–4) occupancy ± 95% CI')
    ax.set_xlabel('Session number'); ax.set_ylabel('Proportion adaptive')

    # Panel 3: external linkage scatter or barrier-crossing bar.
    ax = axes[2]
    if not linkage.empty:
        top = linkage.iloc[0]
        ax.bar([str(r['external_measure'])[:10] for _, r in linkage.iterrows()],
               [r['pearson_r'] for _, r in linkage.iterrows()], color='#785EF0')
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_ylim(-1, 1)
        ax.set_title('VAAMR progression ↔ external change (r)')
        ax.tick_params(axis='x', rotation=45, labelsize=7)
    else:
        crossed = int(barrier['crossed_to_attention_regulation'].sum())
        ax.bar(['crossed', 'not'], [crossed, len(barrier) - crossed],
               color=['#3AB75D', '#FE6100'])
        ax.set_title('Avoidance→AttnReg barrier crossing')
        ax.set_ylabel('participants')

    fig.tight_layout()
    out = _paths.figures_dir(output_dir)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'program_efficacy.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path
