"""
analysis/reports/case_studies.py
--------------------------------
Per-cohort preliminary case studies — deterministic selection + grant-doc report.

For every cohort in the corpus, deterministically selects the participant whose
LLM-consensus VA-MR language shows the clearest within-program shift, then writes
a single grant-ready document (06_reports/02_outcomes/case_studies.txt) with a
companion figure quantifying the number of utterances per stage for each case
(case_studies_fig.png).  Intended as the qualitative-side half of the H4
convergent-validity linkage: each case carries fill-in-the-blank slots for the
trial's REDCap patient-reported outcomes.

Selection rule (pre-specified; [M17] in 08_methods.txt):
  Tier A "climber":      >=4 attended sessions with >=1 coded utterance,
                         >=10 coded utterances, Kendall tau(session, stage) > 0.
                         Rank: tau desc, early->late shift desc, n_coded desc,
                         participant_id asc.
  Tier B "consolidator": same eligibility, no tau requirement, early->late
                         shift > 0 (shift = drop in Vigilance/Avoidance share
                         plus rise in Metacognition/Reappraisal share between
                         the first two and last two attended sessions).
  Tier C "insufficient": largest n_coded (reported with an explicit caveat).

Everything is computed from the label of record (final_label); no random state.
Analyst-curated background/confounds are read from the optional sidecar
02_meta/case_study_annotations.json so curated content stays data, and the
generated document remains deterministic.
"""

import json
import os
import re
from datetime import date

import numpy as np
import pandas as pd

from ..figures import _VAAMR_COLORS
from ..loader import _derive_cohort_id
from .stat_format import fmt_p, provenance_header

STAGE_NAMES = {0: 'Vigilance', 1: 'Avoidance', 2: 'Attention Regulation',
               3: 'Metacognition', 4: 'Reappraisal'}
STAGE_SHORT = {0: 'VIG', 1: 'AVD', 2: 'ATT', 3: 'META', 4: 'REAP'}

# The trial's PRO battery (Move-MORE protocol) with pre-registered directions
# under H4/H2a (greater VA-MR progression <-> greater improvement).
PRO_CHECKLIST = [
    ("Weekly pain NRS (0-10)", "down"),
    ("PEG-3 (pain, enjoyment, activity)", "down"),
    ("Modified Oswestry Disability Index", "down"),
    ("TSK-11 kinesiophobia", "down"),
    ("Pain Catastrophizing Scale (+ subscales)", "down"),
    ("Mindful Reappraisal of Pain Scale (MRPS)", "up"),
    ("MAIA-2 interoception (subscales)", "up"),
    ("PROMIS Sleep Disturbance 6a (+ duration)", "down"),
    ("PROMIS Self-Efficacy Managing Symptoms 8a", "up"),
    ("PROMIS Physical Function 6b", "up"),
    ("PROMIS Ability to Participate in Social Roles", "up"),
    ("Daily EMA pain (REDCap diary)", "down"),
    ("Home-practice adherence / minutes", "context"),
    ("Adverse events / co-interventions log", "context"),
]

WRAP_RULE = "=" * 78
SUB_RULE = "-" * 78


# ── cohort + candidate table ────────────────────────────────────────────────

def _with_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a per-row integer session_cohort (from session_id) and
    a per-participant modal cohort column (ties -> lowest; dual_cohort flagged)."""
    out = df.copy()
    out['session_cohort'] = out['session_id'].apply(_derive_cohort_id)
    if 'cohort_id' in out.columns:
        out['session_cohort'] = out['session_cohort'].fillna(
            pd.to_numeric(out['cohort_id'], errors='coerce'))
    return out


def _participant_cohort(sub: pd.DataFrame):
    """Modal session cohort for one participant; (cohort, dual_cohort_flag)."""
    vals = sub['session_cohort'].dropna().astype(int)
    if vals.empty:
        return None, False
    mode = vals.mode()
    return int(mode.min()), vals.nunique() > 1


def compute_candidate_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per participant with the deterministic selection statistics.

    Expects the analysis-loader frame: participant-only rows, final_label int.
    """
    try:
        from scipy import stats as _st
    except Exception:      # pragma: no cover - scipy is a pipeline dependency
        _st = None
    d = _with_cohort(df)
    rows = []
    for pid in sorted(d['participant_id'].dropna().unique()):
        g = d[d['participant_id'] == pid].dropna(subset=['final_label'])
        if g.empty:
            continue
        cohort, dual = _participant_cohort(g)
        sessions = sorted(g['session_number'].dropna().unique())
        tau = tau_p = np.nan
        if _st is not None and len(sessions) >= 2 and g['final_label'].nunique() > 1:
            tau, tau_p = _st.kendalltau(g['session_number'], g['final_label'])
        early = g[g['session_number'].isin(sessions[:2])]
        late = g[g['session_number'].isin(sessions[-2:])]
        va_e, va_l = (early['final_label'] <= 1).mean(), (late['final_label'] <= 1).mean()
        mr_e, mr_l = (early['final_label'] >= 3).mean(), (late['final_label'] >= 3).mean()
        rows.append({
            'participant_id': pid, 'cohort': cohort, 'dual_cohort': dual,
            'n_sessions': len(sessions), 'n_coded': int(len(g)),
            'sessions': sessions,
            'tau': float(tau) if pd.notna(tau) else np.nan,
            'tau_p': float(tau_p) if pd.notna(tau_p) else np.nan,
            'E_first': float(g[g['session_number'] == sessions[0]]['final_label'].mean()),
            'E_last': float(g[g['session_number'] == sessions[-1]]['final_label'].mean()),
            'va_early': float(va_e), 'va_late': float(va_l),
            'mr_early': float(mr_e), 'mr_late': float(mr_l),
            'shift': float((va_e - va_l) + (mr_l - mr_e)),
            'n_early': int(len(early)), 'n_late': int(len(late)),
        })
    return pd.DataFrame(rows)


def select_case_studies(df: pd.DataFrame) -> list:
    """Deterministically select one case study per cohort. Returns a list of
    dicts (candidate-table row + 'archetype' + 'tier')."""
    table = compute_candidate_table(df)
    if table.empty:
        return []
    picks = []
    for cohort in sorted(table['cohort'].dropna().unique()):
        cand = table[table['cohort'] == cohort]
        eligible = cand[(cand['n_sessions'] >= 4) & (cand['n_coded'] >= 10)]
        tier_a = eligible[eligible['tau'] > 0].sort_values(
            ['tau', 'shift', 'n_coded', 'participant_id'],
            ascending=[False, False, False, True])
        if len(tier_a):
            row, arch, tier = tier_a.iloc[0], 'climber', 'A'
        else:
            tier_b = eligible[eligible['shift'] > 0].sort_values(
                ['shift', 'n_coded', 'participant_id'],
                ascending=[False, False, True])
            if len(tier_b):
                row, arch, tier = tier_b.iloc[0], 'consolidator', 'B'
            else:
                tier_c = cand.sort_values(['n_coded', 'participant_id'],
                                          ascending=[False, True])
                row, arch, tier = tier_c.iloc[0], 'insufficient', 'C'
        pick = row.to_dict()
        pick['archetype'], pick['tier'] = arch, tier
        picks.append(pick)
    return picks


# ── per-case statistics + quotes ────────────────────────────────────────────

def _fisher(early, late, pred):
    try:
        from scipy import stats as _st
    except Exception:      # pragma: no cover
        return np.nan
    a, c = int(pred(early).sum()), int(pred(late).sum())
    try:
        return float(_st.fisher_exact([[a, len(early) - a], [c, len(late) - c]])[1])
    except Exception:
        return np.nan


def _clean_quote(text: str, limit: int = 420, own_pid: str = '') -> str:
    """Keep the participant's own words: drop everything before the LAST
    participant speaker tag (removes embedded preceding therapist turns),
    strip the tag itself, drop leading cross-talk sentences that address a
    DIFFERENT participant placeholder, and truncate at a word boundary."""
    s = str(text)
    matches = list(re.finditer(r'\[Participant_[^\]]*\]:\s*', s))
    if matches:
        s = s[matches[-1].end():]
    s = re.split(r'\[[A-Za-z_{][^\]]{0,40}\]:', s)[0]   # drop any trailing other-speaker turn
    s = re.sub(r'\s+', ' ', s).strip()
    if own_pid:                                          # drop a SHORT leading run of cross-talk
        sents = re.split(r'(?<=[.?!])\s+', s)
        cut = -1
        for i, sent in enumerate(sents[:5]):
            if '{Participant_' in sent and own_pid not in sent:
                cut = i
        if 0 <= cut < len(sents) - 1:
            head = ' '.join(sents[:cut + 1])
            if len(head) <= min(180, 0.4 * len(s)):      # never eat the substance
                s = ' '.join(sents[cut + 1:])
    for _ in range(4):                                   # strip leading conversational filler
        s2 = re.sub(r'^(?:Yeah\.?,?|Okay\.?,?|Right\.?,?|\(NAME\)\.?,?|Um,?|Uh,?|I mean,?|you know,?)\s+',
                    '', s, count=1, flags=re.IGNORECASE)
        if s2 == s:
            break
        s = s2
    if s and s[0].islower():
        s = '[...] ' + s
    if len(s) > limit:
        s = s[:limit].rsplit(' ', 1)[0].rstrip(',;: ') + ' [...]'
    return s


def _pick_quotes(g: pd.DataFrame, sessions, phase: str, n: int = 2) -> list:
    """Deterministic quote picker. phase='early' -> maladaptive (or lowest)
    stages in the first two attended sessions; phase='late' -> the most
    advanced stages in the last two attended sessions."""
    sub = g[g['session_number'].isin(sessions[:2] if phase == 'early' else sessions[-2:])]
    sub = sub.dropna(subset=['final_label'])
    if sub.empty:
        return []
    if phase == 'early':
        pool = sub[sub['final_label'] <= 1]
        if pool.empty:
            pool = sub[sub['final_label'] == sub['final_label'].min()]
    else:
        best = sub['final_label'].max()
        pool = sub[sub['final_label'] >= max(2, best - 1)]
        if pool.empty:
            pool = sub[sub['final_label'] == best]
    pool = pool.copy()
    pool['_pain_kw'] = pool['text'].astype(str).str.contains('pain', case=False, na=False)
    cols = [c for c in ('llm_run_consistency', 'llm_confidence_primary') if c in pool.columns]
    sort_cols = (['session_number'] if phase == 'late' else []) + ['_pain_kw'] + cols + ['segment_id']
    ascending = ([False] if phase == 'late' else []) + [False] + [False] * len(cols) + [True]
    pool = pool.sort_values(sort_cols, ascending=ascending, na_position='last')
    out, seen = [], []
    for _, r in pool.iterrows():
        txt = _clean_quote(r.get('text', ''), own_pid=str(r.get('participant_id', '')))
        if any(txt[:120] == t[:120] for t in seen):      # overlapping segment windows
            continue
        seen.append(txt)
        out.append({'segment_id': r['segment_id'],
                    'session_number': int(r['session_number']),
                    'stage': int(r['final_label']),
                    'text': txt})
        if len(out) >= n:
            break
    return out


def _case_detail(df: pd.DataFrame, df_all, pick: dict) -> dict:
    """Assemble everything the report/figure needs for one selected case."""
    pid = pick['participant_id']
    g = df[df['participant_id'] == pid].dropna(subset=['final_label'])
    sessions = pick['sessions']
    early = g[g['session_number'].isin(sessions[:2])]
    late = g[g['session_number'].isin(sessions[-2:])]
    per_session, totals = {}, {s: 0 for s in STAGE_NAMES}
    n_uncoded = {}
    for s in sessions:
        gs = g[g['session_number'] == s]
        per_session[int(s)] = {int(k): int(v) for k, v in
                               gs['final_label'].value_counts().sort_index().items()}
        for k, v in per_session[int(s)].items():
            totals[k] += v
    if df_all is not None and 'speaker' in df_all.columns:
        alls = df_all[(df_all['participant_id'] == pid)
                      & (df_all['speaker'] == 'participant')]
        for s in sessions:
            tot = int((alls['session_number'] == s).sum())
            n_uncoded[int(s)] = max(0, tot - sum(per_session[int(s)].values()))
    return {
        **pick,
        'per_session': per_session,
        'totals': totals,
        'n_uncoded': n_uncoded,
        'n_uncoded_total': int(sum(n_uncoded.values())) if n_uncoded else None,
        'fisher_va': _fisher(early['final_label'], late['final_label'], lambda s: s <= 1),
        'fisher_mr': _fisher(early['final_label'], late['final_label'], lambda s: s >= 3),
        'fisher_adaptive': _fisher(early['final_label'], late['final_label'], lambda s: s >= 2),
        'quotes_early': _pick_quotes(g, sessions, 'early'),
        'quotes_late': _pick_quotes(g, sessions, 'late'),
        'seq': {int(s): [STAGE_SHORT[int(v)] for v in
                         g[g['session_number'] == s].sort_values('segment_index')['final_label']]
                for s in sessions},
    }


def _load_annotations(output_dir: str) -> dict:
    path = os.path.join(output_dir, '02_meta', 'case_study_annotations.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith('_')}
    except Exception:
        return {}


# ── document ────────────────────────────────────────────────────────────────

def _pct(x) -> str:
    return 'n/a' if pd.isna(x) else f"{x:.0%}"


def _grant_paragraph(c: dict, ann: dict) -> str:
    """One flowing grant-ready paragraph for a case (single line, no hard wrap)."""
    pid_short = str(c['participant_id']).replace('Participant_', '')
    qe = c['quotes_early'][0]['text'] if c['quotes_early'] else ''
    ql = c['quotes_late'][0]['text'] if c['quotes_late'] else ''
    stats_bit = (
        f"the share of codable utterances coded Vigilance/Avoidance moved from "
        f"{_pct(c['va_early'])} ({int(round(c['va_early'] * c['n_early']))}/{c['n_early']}) in the first two attended sessions to "
        f"{_pct(c['va_late'])} ({int(round(c['va_late'] * c['n_late']))}/{c['n_late']}) in the final two (Fisher exact {fmt_p(c['fisher_va'])}), "
        f"while Metacognition/Reappraisal moved from {_pct(c['mr_early'])} to {_pct(c['mr_late'])} ({fmt_p(c['fisher_mr'])}); "
        f"the segment-level stage ordinal trends with session at Kendall tau={c['tau']:+.2f} ({fmt_p(c['tau_p'])}, n={c['n_coded']} coded utterances), "
        f"and mean stage moved from {c['E_first']:.2f} (first session) to {c['E_last']:.2f} (last)"
    )
    if c['archetype'] == 'climber':
        reaches_mr = c['mr_late'] > 0 and c['mr_late'] >= c['mr_early']
        dest = ("toward Metacognition/Reappraisal" if reaches_mr
                else "across the Avoidance barrier into sustained Attention Regulation")
        lead = (f"Across {c['n_sessions']} attended sessions, QRA-coded VA-MR language for Participant {pid_short} "
                f"moved from the maladaptive pole {dest}: ")
        quote_bit = (f" Early-program language exemplifies the maladaptive pole (\"{qe}\"), "
                     f"while final-session language exemplifies the adaptive pole (\"{ql}\").")
    elif c['archetype'] == 'consolidator':
        lead = (f"Participant {pid_short} entered the program already expressing adaptive-stage language and consolidated it: ")
        quote_bit = (f" Early-program language (\"{qe}\") gives way to consolidated adaptive-stage language (\"{ql}\").")
    else:
        lead = (f"Participant {pid_short} is the best-documented participant in this cohort, though the corpus "
                f"does not support a trajectory claim (see caveats): ")
        quote_bit = f" Representative language: \"{ql or qe}\"."
    pro_bit = (" On the trial's patient-reported outcomes, baseline-to-Week-8 change was ___ on PEG-3 (___ -> ___), "
               "___ on weekly pain NRS (___ -> ___), ___ on the Oswestry Disability Index (___ -> ___), "
               "___ on TSK-11 kinesiophobia (___ -> ___), ___ on the Mindful Reappraisal of Pain Scale (___ -> ___), "
               "and ___ on the Pain Catastrophizing Scale (___ -> ___), with 3-month follow-up values of ___.")
    caveats = ann.get('confounds') or []
    caveat_bit = (" [Caveats to carry into any outcome claim: " + " ".join(caveats) + "]") if caveats else ""
    return lead + stats_bit + "." + quote_bit + pro_bit + caveat_bit


def _write_case_section(L: list, idx: int, c: dict, ann: dict) -> None:
    pid_short = str(c['participant_id']).replace('Participant_', '')
    arch = {'climber': 'clear forward VA-MR shift',
            'consolidator': 'adaptive-stage consolidation',
            'insufficient': 'best available (insufficient for a trajectory claim)'}[c['archetype']]
    L.append("")
    L.append(WRAP_RULE)
    L.append(f"CASE {idx} - PARTICIPANT {pid_short}  (Cohort {int(c['cohort'])}; "
             f"selection tier {c['tier']}: {arch})")
    L.append(WRAP_RULE)
    if c.get('dual_cohort'):
        L.append("NOTE: dual-cohort participant - began in one cohort and completed the program "
                 "with another; the session axis below is program week pooled across cohorts.")
        L.append("")
    if ann.get('profile'):
        L.append(f"Profile: {ann['profile']}")
        L.append("")
    L.append("Grant-ready paragraph (fill ___ from REDCap):")
    L.append("")
    L.append(_grant_paragraph(c, ann))
    L.append("")
    L.append("Utterance counts by session (LLM-consensus label of record; uncoded = abstained/unlabeled):")
    hdr = "  session | " + " | ".join(f"{STAGE_SHORT[s]:>4}" for s in STAGE_NAMES) + " | uncoded | coded sequence"
    L.append(hdr)
    for s, counts in c['per_session'].items():
        cells = " | ".join(f"{counts.get(st, 0):>4}" for st in STAGE_NAMES)
        unc = c['n_uncoded'].get(s, '')
        L.append(f"  {s:>7} | {cells} | {str(unc):>7} | " + " -> ".join(c['seq'][s]))
    tot_cells = " | ".join(f"{c['totals'].get(st, 0):>4}" for st in STAGE_NAMES)
    unc_tot = c['n_uncoded_total'] if c['n_uncoded_total'] is not None else ''
    L.append(f"  {'TOTAL':>7} | {tot_cells} | {str(unc_tot):>7} |")
    L.append("")
    L.append(f"Trend: Kendall tau={c['tau']:+.3f} ({fmt_p(c['tau_p'])}); mean stage first->last session "
             f"{c['E_first']:.2f} -> {c['E_last']:.2f}; adaptive-share early->late "
             f"{_pct(1 - c['va_early'])} -> {_pct(1 - c['va_late'])} (Fisher {fmt_p(c['fisher_adaptive'])}).")
    L.append("")
    L.append("Supporting quotes (verbatim from frozen segments; deterministic top-confidence picks):")
    for q in c['quotes_early']:
        L.append(f"  S{q['session_number']} [{STAGE_SHORT[q['stage']]}] \"{q['text']}\"  ({q['segment_id']})")
    for q in c['quotes_late']:
        L.append(f"  S{q['session_number']} [{STAGE_SHORT[q['stage']]}] \"{q['text']}\"  ({q['segment_id']})")
    if ann.get('outcome_signals'):
        L.append("")
        L.append("Unprompted outcome signals embedded in session speech (check against REDCap):")
        for s in ann['outcome_signals']:
            L.append(f"  - {s}")
    if ann.get('confounds'):
        L.append("")
        L.append("Confounds / caveats:")
        for s in ann['confounds']:
            L.append(f"  - {s}")
    if ann.get('notes'):
        L.append("")
        L.append(f"Analyst note: {ann['notes']}")


def _write_document(path: str, cases: list, table: pd.DataFrame,
                    annotations: dict, fig_name: str) -> None:
    L = []
    L.append(WRAP_RULE)
    L.append("PRELIMINARY CASE STUDIES - ONE PER COHORT (deterministically selected)")
    L.append("VA-MR language trajectories as candidates for REDCap outcome linkage")
    L.append(WRAP_RULE)
    L.append(f"Generated: {date.today().isoformat()}")
    L.extend(provenance_header(['vaamr_labels', 'case_selection']))
    L.append("")
    L.append("Purpose: for each cohort, surface the participant whose QRA-coded VA-MR language shows the clearest within-program shift, so their patient-reported outcomes (PROs) can be pulled from REDCap and examined as PRELIMINARY convergent-validity evidence (H4/H2a: greater VA-MR progression <-> greater PRO improvement). Selection is rule-based and re-derived on every analysis run; it contains no efficacy claim (single-arm; the selected cases are the upper tail of the cohort distribution, chosen for illustration - say so wherever they are quoted). EXPLORATORY, small n.")
    L.append("")
    L.append("Companion figure: " + fig_name + " (utterance counts per stage, per case).")
    for i, c in enumerate(cases, start=1):
        ann = annotations.get(c['participant_id'], {})
        _write_case_section(L, i, c, ann)
    L.append("")
    L.append(WRAP_RULE)
    L.append("REDCAP LOOKUP CHECKLIST  (expected direction under H2a; B=Baseline, W4, W8, 3m)")
    L.append(WRAP_RULE)
    name_w = max(len(n) for n, _ in PRO_CHECKLIST)
    case_hdr = "   ".join(f"{str(c['participant_id']).replace('Participant_', ''):>18}" for c in cases)
    L.append(f"  {'instrument':<{name_w}} | {'dir':<7} | " + case_hdr)
    for name, direction in PRO_CHECKLIST:
        cells = "   ".join(f"{'___/___/___/___':>18}" for _ in cases)
        L.append(f"  {name:<{name_w}} | {direction:<7} | " + cells)
    L.append("")
    L.append(SUB_RULE)
    L.append("SELECTION AUDIT  (every participant, deterministic statistics)")
    L.append(SUB_RULE)
    L.append("  Rule: Tier A 'climber' = >=4 sessions, >=10 coded utterances, Kendall tau(session, stage)>0, ranked by tau then early->late shift; Tier B 'consolidator' = same eligibility, shift>0; Tier C = best-documented fallback. Cohort derived per participant from modal session-id prefix (robust to missing cohort_id).")
    L.append("")
    L.append("  participant  cohort  sessions  coded    tau    tau_p   E first->last   VA early->late   MR early->late   shift")
    for _, r in table.sort_values(['cohort', 'tau'], ascending=[True, False]).iterrows():
        pid_short = str(r['participant_id']).replace('Participant_', '')
        tau = 'n/a  ' if pd.isna(r['tau']) else f"{r['tau']:+.3f}"
        tp = 'n/a ' if pd.isna(r['tau_p']) else f"{r['tau_p']:.3f}"
        L.append(f"  {pid_short:<11}  {int(r['cohort']) if pd.notna(r['cohort']) else '?':>6}  {r['n_sessions']:>8}  {r['n_coded']:>5}  {tau}  {tp}   "
                 f"{r['E_first']:.2f} -> {r['E_last']:.2f}     {_pct(r['va_early']):>4} -> {_pct(r['va_late']):<4}   "
                 f"{_pct(r['mr_early']):>4} -> {_pct(r['mr_late']):<4}   {r['shift']:+.2f}")
    L.append("")
    L.append("Reproducibility: statistics recompute from the frozen label of record (final_label) via analysis.reports.case_studies; quotes are verbatim frozen-segment text addressed by segment_id. Curated profile/confound lines come from 02_meta/case_study_annotations.json (analyst-maintained data).")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L) + "\n")


# ── figure ──────────────────────────────────────────────────────────────────

def _render_figure(path: str, cases: list) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    _BG, _GRID, _INK, _MUTED = 'white', '#D8D8D8', '#1A1A1A', '#555555'
    ncol = max(1, len(cases))
    fig, axes = plt.subplots(2, ncol, figsize=(4.6 * ncol, 8.2), squeeze=False)
    fig.patch.set_facecolor(_BG)

    def style(ax):
        ax.set_facecolor(_BG)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color('#888888')
        ax.tick_params(colors=_MUTED, labelsize=9)
        ax.set_axisbelow(True)

    for j, c in enumerate(cases):
        pid_short = str(c['participant_id']).replace('Participant_', '')
        arch = {'climber': 'forward shift', 'consolidator': 'consolidation',
                'insufficient': 'best available'}[c['archetype']]
        # top: per-session stacked counts
        ax = axes[0][j]
        style(ax)
        sessions = sorted(c['per_session'].keys())
        bottoms = np.zeros(len(sessions))
        for st in STAGE_NAMES:
            vals = np.array([c['per_session'][s].get(st, 0) for s in sessions], dtype=float)
            bars = ax.bar([str(s) for s in sessions], vals, bottom=bottoms,
                          color=_VAAMR_COLORS[st], width=0.72,
                          edgecolor='white', linewidth=1.2)
            for b, v in zip(bars, vals):
                if v >= 1:
                    ax.text(b.get_x() + b.get_width() / 2, b.get_y() + v / 2, str(int(v)),
                            ha='center', va='center', fontsize=8, color='white',
                            fontweight='bold')
            bottoms += vals
        n_unc = c['n_uncoded_total']
        sub = f"n={c['n_coded']} coded" + (f" (+{n_unc} uncoded)" if n_unc else "")
        ax.set_title(f"{pid_short} - Cohort {int(c['cohort'])} ({arch})\n{sub}",
                     loc='left', fontsize=10.5, color=_INK)
        ax.set_xlabel('Program session', color=_MUTED, fontsize=9)
        if j == 0:
            ax.set_ylabel('Utterances (count)', color=_MUTED, fontsize=9)
        ax.yaxis.grid(True, color=_GRID, linewidth=0.8)
        ax.yaxis.get_major_locator().set_params(integer=True)
        # bottom: per-stage totals
        ax = axes[1][j]
        style(ax)
        ys = list(STAGE_NAMES)[::-1]
        vals = [c['totals'].get(st, 0) for st in ys]
        bars = ax.barh([STAGE_NAMES[st] for st in ys], vals,
                       color=[_VAAMR_COLORS[st] for st in ys],
                       height=0.62, edgecolor='white', linewidth=1.0)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + max(vals + [1]) * 0.02, b.get_y() + b.get_height() / 2,
                    str(int(v)), va='center', ha='left', fontsize=9, color=_INK)
        ax.set_xlim(0, max(vals + [1]) * 1.15)
        ax.set_xlabel('Total utterances (count)', color=_MUTED, fontsize=9)
        ax.xaxis.grid(True, color=_GRID, linewidth=0.8)
        ax.xaxis.get_major_locator().set_params(integer=True)

    handles = [Patch(facecolor=_VAAMR_COLORS[st], label=STAGE_NAMES[st]) for st in STAGE_NAMES]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=9)
    fig.suptitle('Per-cohort case studies - VA-MR utterance counts by stage '
                 '(LLM-consensus label of record; abstentions excluded from stage counts)',
                 x=0.02, ha='left', fontsize=11, color=_INK)
    fig.text(0.5, 0.005,
             'Deterministic per-cohort selection - illustration, not efficacy. Methods: 08_methods.txt',
             ha='center', fontsize=8.5, color=_MUTED, style='italic')
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=220, facecolor=_BG)
    plt.close(fig)


# ── entry point ─────────────────────────────────────────────────────────────

def generate_case_studies(df: pd.DataFrame, df_all, framework, output_dir: str) -> list:
    """Write 06_reports/02_outcomes/case_studies.txt (+ _fig.png). Returns paths."""
    try:
        from process import output_paths as _paths
        if df is None or df.empty:
            return []
        picks = select_case_studies(df)
        if not picks:
            return []
        d = _with_cohort(df)
        cases = [_case_detail(d, df_all, p) for p in picks]
        table = compute_candidate_table(df)
        annotations = _load_annotations(output_dir)
        txt_path = _paths.reports_case_studies_path(output_dir)
        fig_path = _paths.reports_case_studies_figure_path(output_dir)
        _write_document(txt_path, cases, table, annotations, os.path.basename(fig_path))
        try:
            _render_figure(fig_path, cases)
        except Exception as e:      # figure failure must not lose the document
            print(f"  Warning: case-study figure failed: {e}")
            return [txt_path]
        return [txt_path, fig_path]
    except Exception as e:
        print(f"  Warning: case studies failed: {e}")
        return []
