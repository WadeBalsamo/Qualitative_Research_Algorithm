"""Per-participant human-readable text report generator.

Restructured to LEAD with the quantitative trajectory: a per-session E[stage]
table, an OLS slope (estimator-labeled), barrier status, stage profile, and the
participant's own dominant-stage movement — before any LLM narrative prose. This
is the canonical home for participant quotes (one best expression per stage).
"""

import os
import re
from datetime import date

import numpy as np
import pandas as pd

from process import output_paths as _paths
from ._formatting import _bar, _pct, _wrap_quote, _wrap_text, _summarize_participant_text
from .stat_format import fmt_signed, provenance_header


# Direction thresholds mirror analysis/participant.py:_compute_progression_trend
# interpretation (slope > 0.1 advancing, < -0.1 regressing, else stable). Keep in
# sync with that module — it is the source of the per-participant slope semantics.
_ADV_SLOPE = 0.1

_ADAPTIVE_STAGES = (2, 3, 4)   # Attention Regulation / Metacognition / Reappraisal
_AVOIDANCE_STAGE = 1           # the published barrier


def _direction_word(slope: float) -> str:
    if slope is None or (isinstance(slope, float) and np.isnan(slope)):
        return 'undetermined'
    if slope > _ADV_SLOPE:
        return 'advancing'
    if slope < -_ADV_SLOPE:
        return 'regressing'
    return 'stable'


def _ols_slope(xs, ys) -> float:
    """Least-squares slope of ys over xs (positional). None if <2 points."""
    if len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    xm, ym = x.mean(), y.mean()
    den = float(((x - xm) ** 2).sum())
    if den == 0:
        return None
    return float(((x - xm) * (y - ym)).sum() / den)


def _session_estage(sdf: pd.DataFrame) -> float:
    """E[stage] for one session's participant segments.

    Prefers the mixture-weighted coordinate (progression_coord); falls back to
    the mean hard label when no mixture is attached.
    """
    if sdf.empty:
        return None
    if 'progression_coord' in sdf.columns and sdf['progression_coord'].notna().any():
        return float(sdf['progression_coord'].dropna().mean())
    if 'final_label' in sdf.columns and sdf['final_label'].notna().any():
        return float(sdf['final_label'].dropna().astype(float).mean())
    return None


def generate_participant_txt_report(
    df: pd.DataFrame,
    participant_id: str,
    participant_json: dict,
    framework: dict,
    output_dir: str,
    llm_client=None,
    participant_summaries_config=None,
) -> str:
    """Generate a human-readable .txt report for a single participant.

    Structure (quantitative-first):
      1. Title + provenance header
      2. TRAJECTORY TABLE  (per attended session: E[stage], adaptive %, ΔE) + OLS slope
      3. BARRIER STATUS
      4. STAGE PROFILE     (distribution + confidence-tier mix)
      5. WITHIN-PERSON MOVEMENT  (dominant-stage arrows + forward/backward count)
      6. BEST EXPRESSIONS  (1 per stage, highest confidence)
      7. NARRATIVE CONTEXT (LLM, optional, LAST)

    participant_summaries_config: optional config with .enabled and
    .max_words_per_session. When enabled, an LLM narrative is appended LAST under
    an explicit "not analysis" rule.

    Returns path to written file.
    """
    stage_ids = sorted(framework.keys())
    stage_names = {sid: framework[sid].get('short_name', f'Stage {sid}') for sid in stage_ids}

    cohort_id = participant_json.get('cohort_id')
    session_ids = participant_json.get('session_ids', [])
    sessions_detail = participant_json.get('sessions', {})
    stage_exemplars_overall = participant_json.get('stage_exemplars_overall', {})
    n_sessions = participant_json.get('n_sessions', len(session_ids))

    cohort_str = f'Cohort {cohort_id}' if cohort_id is not None else ''

    pdf_all = df[df['participant_id'] == participant_id]
    n_total_segs = len(pdf_all)
    stage_counts_overall = (
        pdf_all['final_label'].value_counts().to_dict() if 'final_label' in pdf_all.columns else {}
    )

    # Participant summary config
    summaries_enabled = (
        participant_summaries_config is not None
        and getattr(participant_summaries_config, 'enabled', False)
    )
    max_summary_words = (
        getattr(participant_summaries_config, 'max_words_per_session', 300)
        if participant_summaries_config is not None else 300
    )

    lines = []

    # ── Title + provenance ────────────────────────────────────────────────────
    header = f'PARTICIPANT {participant_id}'
    if cohort_str:
        header += f'  [{cohort_str}]'
    lines.append(header)
    lines.append('=' * 68)
    lines.append(f'Generated: {date.today().isoformat()}   |   {n_sessions} session(s) attended')
    lines.extend(provenance_header(['vaamr_labels', 'estage', 'barrier']))
    lines.append('')

    # ── 1. TRAJECTORY TABLE (the centerpiece) ─────────────────────────────────
    # Build per-session rows in chronological order.
    rows = []  # (snum, sid, n_seg, dom_stage, estage, adaptive_pct)
    for sid in session_ids:
        sdet = sessions_detail.get(sid, {})
        sdf = pdf_all[pdf_all['session_id'] == sid] if not pdf_all.empty else pdf_all
        n_seg = sdet.get('n_segments', len(sdf))
        snum = sdet.get('session_number')
        if snum is None and not sdf.empty and 'session_number' in sdf.columns:
            try:
                snum = int(sdf['session_number'].iloc[0])
            except Exception:
                snum = None
        dom_stage = sdet.get('dominant_stage')
        if dom_stage is None and not sdf.empty and 'final_label' in sdf.columns and sdf['final_label'].notna().any():
            dom_stage = int(sdf['final_label'].mode().iloc[0])
        # E[stage]: prefer JSON continuous_progression, else compute from df.
        estage = sdet.get('continuous_progression')
        if estage is None:
            estage = _session_estage(sdf)
        # adaptive occupancy (stages 2-4)
        adaptive_pct = None
        if not sdf.empty and 'final_label' in sdf.columns and sdf['final_label'].notna().any():
            fl = sdf['final_label'].dropna().astype(int)
            adaptive_pct = float(fl.isin(_ADAPTIVE_STAGES).mean())
        rows.append({
            'snum': snum, 'sid': sid, 'n_seg': n_seg,
            'dom_stage': dom_stage, 'estage': estage, 'adaptive_pct': adaptive_pct,
        })

    lines.append('TRAJECTORY  [one row per attended session, chronological]')
    lines.append('─' * 72)
    lines.append(
        f'{"S#":>3}  {"Session":<8} {"Seg":>4}  {"Dominant Stage":<14} '
        f'{"E[stage]":>8}  {"Adapt%":>7}  {"ΔE[stage]":>9}'
    )
    lines.append('─' * 72)
    prev_estage = None
    for r in rows:
        snum_s = str(r['snum']) if r['snum'] is not None else '?'
        dom_name = stage_names.get(r['dom_stage'], '?') if r['dom_stage'] is not None else '?'
        est = r['estage']
        est_s = f'{est:.2f}' if est is not None else '  ─'
        adapt_s = _pct(r['adaptive_pct']) if r['adaptive_pct'] is not None else '   ─'
        if est is not None and prev_estage is not None:
            delta_s = fmt_signed(est - prev_estage)
        else:
            delta_s = '   ─'
        lines.append(
            f'{snum_s:>3}  {r["sid"]:<8} {r["n_seg"]:>4}  {dom_name:<14} '
            f'{est_s:>8}  {adapt_s:>7}  {delta_s:>9}'
        )
        if est is not None:
            prev_estage = est
    lines.append('─' * 72)

    # OLS slope over per-session E[stage] means (≥3 sessions only).
    est_series = [(i, r['estage']) for i, r in enumerate(rows) if r['estage'] is not None]
    if len(est_series) >= 3:
        slope = _ols_slope([i for i, _ in est_series], [e for _, e in est_series])
        if slope is not None:
            word = _direction_word(slope)
            lines.append(
                f'E[stage] OLS slope: {fmt_signed(slope)}/session  ({word})'
            )
            lines.append(
                '  (E[stage] = mixture-weighted mean stage; OLS over per-session means, '
                'interval-scale sensitivity estimator)'
            )
    elif len(est_series) >= 2:
        lines.append(
            f'E[stage] OLS slope: not reported (only {len(est_series)} sessions with E[stage]; ≥3 required)'
        )
    else:
        lines.append('E[stage] OLS slope: n/a (single session)')
    lines.append('')

    # ── 2. BARRIER STATUS ─────────────────────────────────────────────────────
    lines.append('BARRIER STATUS  [Avoidance → Attention-Regulation, the published obstacle]')
    lines.append('─' * 72)
    dom_seq = [(r['sid'], r['dom_stage']) for r in rows if r['dom_stage'] is not None]
    first_avoid_idx = next((i for i, (_, st) in enumerate(dom_seq) if st == _AVOIDANCE_STAGE), None)
    if first_avoid_idx is not None:
        avoid_sid = dom_seq[first_avoid_idx][0]
        # First subsequent session whose dominant stage is above Avoidance.
        crossed = next(
            ((sid, st) for sid, st in dom_seq[first_avoid_idx + 1:] if st > _AVOIDANCE_STAGE),
            None,
        )
        lines.append(f'  Ever Avoidance-dominant: yes (first at {avoid_sid})')
        if crossed:
            lines.append(
                f'  Crossed the barrier: yes — first reached '
                f'{stage_names.get(crossed[1], crossed[1])} (dominant) at {crossed[0]}'
            )
        else:
            lines.append('  Crossed the barrier: no session after first Avoidance reaches a higher dominant stage')
    else:
        # Never Avoidance-dominant; note whether they were ever above the barrier.
        ever_adaptive = any(st in _ADAPTIVE_STAGES for _, st in dom_seq)
        lines.append('  Ever Avoidance-dominant: no')
        if ever_adaptive:
            lines.append('  (was adaptive-dominant in ≥1 session without an Avoidance-dominant barrier session)')
    lines.append('')

    # ── 3. STAGE PROFILE ──────────────────────────────────────────────────────
    lines.append('STAGE PROFILE  [all sessions combined]')
    lines.append('─' * 72)
    for st in stage_ids:
        name = stage_names[st]
        cnt = int(stage_counts_overall.get(st, 0))
        prop = cnt / n_total_segs if n_total_segs > 0 else 0.0
        lines.append(f'  {name:<20} {cnt:>3} segs  ({_pct(prop):>6})  {_bar(prop, width=24)}')
    # Confidence-tier mix one-liner.
    if 'label_confidence_tier' in pdf_all.columns and not pdf_all.empty:
        tier_counts = pdf_all['label_confidence_tier'].fillna('low').str.lower().value_counts().to_dict()
        tier_parts = [
            f'{t}={int(tier_counts.get(t, 0))}'
            for t in ('high', 'medium', 'low') if tier_counts.get(t, 0)
        ]
        if tier_parts:
            lines.append(f'  Confidence tiers: {", ".join(tier_parts)}  (of {n_total_segs} segments)')
    lines.append('')

    # ── 4. WITHIN-PERSON MOVEMENT ─────────────────────────────────────────────
    lines.append('WITHIN-PERSON MOVEMENT  [dominant stage, session to session]')
    lines.append('─' * 72)
    if dom_seq:
        arrow_parts = []
        for idx, (sid, st) in enumerate(dom_seq, 1):
            arrow_parts.append(f'S{idx} {stage_names.get(st, st)}')
        lines.append('  ' + _wrap_text(' → '.join(arrow_parts), indent=2).lstrip())
        fwd = bwd = lat = 0
        for (_, a), (_, b) in zip(dom_seq, dom_seq[1:]):
            if b > a:
                fwd += 1
            elif b < a:
                bwd += 1
            else:
                lat += 1
        lines.append(
            f'  Between-session moves: {fwd} forward, {bwd} backward, {lat} stable'
        )
    else:
        lines.append('  [no dominant-stage sequence available]')
    lines.append('')

    # ── 5. BEST EXPRESSIONS BY STAGE (canonical participant quotes) ───────────
    lines.append('BEST EXPRESSIONS BY STAGE  [≤1 per stage, highest confidence]')
    lines.append('─' * 72)
    any_exemplar = False
    for st in stage_ids:
        ex = stage_exemplars_overall.get(str(st))
        if not ex:
            continue
        any_exemplar = True
        conf = ex.get('confidence') or 0
        text = ex.get('text', '').strip()
        sid_ex = ex.get('session_id', '')
        sid_str = f'{sid_ex}, ' if sid_ex else ''
        conf_str = f'conf={conf:.2f}' if conf else ''
        lines.append(f'  {stage_names[st]} [{sid_str}{conf_str}]:')
        lines.append(_wrap_quote(text, indent=4))
    if not any_exemplar:
        lines.append('  [no exemplars available]')
    lines.append('')

    # ── 6. NARRATIVE CONTEXT (LLM, LAST, optional) ────────────────────────────
    if summaries_enabled:
        lines.append('NARRATIVE CONTEXT (LLM-generated, not analysis)')
        lines.append('─' * 72)
        wrote_any = False
        for sid in session_ids:
            pdf_s = df[(df['participant_id'] == participant_id) & (df['session_id'] == sid)]
            if pdf_s.empty:
                continue
            sort_col = 'segment_index' if 'segment_index' in pdf_s.columns else pdf_s.columns[0]
            all_text = ' '.join(
                str(t).strip() for t in pdf_s.sort_values(sort_col)['text'] if str(t).strip()
            )
            if not all_text:
                continue
            p_summary, _ = _summarize_participant_text(all_text, llm_client, max_summary_words)
            lines.append(f'  ── {sid} ──')
            lines.append(_wrap_text(p_summary, indent=4))
            lines.append('')
            wrote_any = True
        if not wrote_any:
            lines.append('  [no participant text available to summarize]')
            lines.append('')

    content = '\n'.join(lines)

    # Strip leading "Participant_" (case-insensitive) to avoid participant_Participant_MM.txt
    clean_id = re.sub(r'^[Pp]articipant_', '', participant_id)
    out_dir = _paths.reports_per_participant_dir(output_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'participant_{clean_id}.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


def generate_all_participant_txt_reports(
    df: pd.DataFrame,
    participant_reports: list,
    framework: dict,
    output_dir: str,
    llm_client=None,
    participant_summaries_config=None,
) -> list:
    """Generate per-participant .txt files for every participant in participant_reports.

    Returns list of paths written.
    """
    paths = []
    for report in participant_reports:
        pid = report.get('participant_id', '')
        if not pid:
            continue
        try:
            path = generate_participant_txt_report(
                df, pid, report, framework, output_dir,
                llm_client=llm_client,
                participant_summaries_config=participant_summaries_config,
            )
            paths.append(path)
        except Exception as e:
            print(f'  Warning: participant txt report failed for {pid}: {e}')
    return paths
