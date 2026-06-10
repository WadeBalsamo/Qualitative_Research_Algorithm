"""Per-session human-readable text report generator.

Restructured to LEAD with a quantitative SESSION DASHBOARD (stage distribution,
mean E[stage], liminal-segment count, PURER move mix) before any narrative. Then
who-was-where per participant, within-session transitions, capped expressions by
stage, and the LLM session summary LAST under an explicit "not analysis" rule.
"""

import os
from collections import defaultdict
from datetime import date

import pandas as pd

from process import output_paths as _paths
from ._formatting import (
    _bar, _pct, _wrap_quote, _wrap_text,
    _PURER_SHORT, _PURER_NAME,
)
from .stat_format import provenance_header


_PURER_DISPLAY_ORDER = [3, 0, 4, 2, 1]  # E, P, R2, R, U — typical frequency

_PURER_DIRECTIONAL_NOTE = (
    'PURER therapist-move labels are DIRECTIONAL (human validation not yet '
    'complete); read therapist-side counts as provisional.'
)


def _session_estage(sdf: pd.DataFrame) -> float:
    """Mean E[stage] for a session's participant segments (mixture-weighted; falls back to label mean)."""
    if sdf is None or sdf.empty:
        return None
    if 'progression_coord' in sdf.columns and sdf['progression_coord'].notna().any():
        return float(sdf['progression_coord'].dropna().mean())
    if 'final_label' in sdf.columns and sdf['final_label'].notna().any():
        return float(sdf['final_label'].dropna().astype(float).mean())
    return None


def generate_session_txt_report(
    df: pd.DataFrame,
    session_id: str,
    session_json: dict,
    framework: dict,
    output_dir: str,
    llm_client=None,
    therapist_cue_config=None,
    df_all: pd.DataFrame = None,
    session_summaries: dict = None,
) -> str:
    """Generate a human-readable .txt report for a single session.

    Structure (quantitative-first):
      1. Title + provenance header
      2. SESSION DASHBOARD  (participants, segments, stage dist, mean E[stage],
         liminal count, PURER move mix + directional note)
      3. WHO WAS WHERE      (per participant: dominant stage, E[stage], n segs)
      4. WITHIN-SESSION TRANSITIONS  (compact)
      5. EXPRESSIONS BY STAGE  (≤2 per stage)
      6. NARRATIVE CONTEXT (LLM session summary, LAST)

    session_summaries: pre-computed {session_id: summary_text}.

    Returns path to written file.
    """
    stage_ids = sorted(framework.keys())
    stage_names = {sid: framework[sid].get('short_name', f'Stage {sid}') for sid in stage_ids}

    snum = session_json.get('session_number', '?')
    cohort_id = session_json.get('cohort_id')
    n_segments = session_json.get('n_segments', 0)
    n_participants = session_json.get('n_participants', 0)
    group_props = session_json.get('group_stage_proportions', {})
    exemplars = session_json.get('stage_exemplars', {})
    trans_mat = session_json.get('stage_transition_matrix', {})
    participants_detail = session_json.get('participants', {})

    cohort_str = f'Cohort {cohort_id}' if cohort_id is not None else ''

    # Participant-segment slice for this session.
    sdf = df[df['session_id'] == session_id] if df is not None else pd.DataFrame()

    # ── Dominant / runner-up stage ────────────────────────────────────────────
    stage_props_sorted = sorted(
        [(float(group_props.get(str(st), 0.0)), st) for st in stage_ids],
        reverse=True,
    )
    dominant_stage = stage_props_sorted[0][1] if stage_props_sorted else None

    # ── PURER therapist stats for this session ────────────────────────────────
    purer_session_counts = {}
    total_t_purer = 0
    purer_dominant_id = None
    if df_all is not None and 'purer_primary' in df_all.columns:
        t_segs = df_all[
            (df_all['session_id'] == session_id)
            & (df_all['speaker'] == 'therapist')
            & df_all['purer_primary'].notna()
        ]
        if not t_segs.empty:
            total_t_purer = len(t_segs)
            vc = t_segs['purer_primary'].value_counts()
            purer_session_counts = vc.to_dict()
            purer_dominant_id = int(vc.idxmax())

    lines = []

    # ── Title + provenance ────────────────────────────────────────────────────
    header = f'SESSION {session_id} — Session {snum}'
    if cohort_str:
        header += f'  [{cohort_str}]'
    lines.append(header)
    lines.append('=' * 68)
    lines.append(f'Generated: {date.today().isoformat()}')
    lines.extend(provenance_header(['vaamr_labels', 'purer_labels']))
    lines.append('')

    # ── 1. SESSION DASHBOARD ──────────────────────────────────────────────────
    lines.append('SESSION DASHBOARD')
    lines.append('─' * 72)
    lines.append(f'  Participants: {n_participants}    Participant segments: {n_segments}')
    mean_est = _session_estage(sdf)
    if mean_est is not None:
        lines.append(f'  Mean E[stage]: {mean_est:.2f}'
                     + (f'   (dominant: {stage_names.get(dominant_stage, "?")})'
                        if dominant_stage is not None else ''))
    elif dominant_stage is not None:
        lines.append(f'  Dominant stage: {stage_names.get(dominant_stage, "?")} '
                     f'({_pct(float(group_props.get(str(dominant_stage), 0.0)))})')

    # Liminal-segment count (mixture entropy above the session median).
    if 'mixture_entropy' in sdf.columns and sdf['mixture_entropy'].notna().any():
        ent = sdf['mixture_entropy'].dropna()
        if len(ent) >= 2:
            med = float(ent.median())
            n_liminal = int((ent > med).sum())
            lines.append(f'  Liminal segments (mixture entropy > session median): {n_liminal} / {len(ent)}')
    lines.append('')

    # Stage distribution.
    lines.append('  Stage distribution (group, equal-weighted):')
    for st in stage_ids:
        name = stage_names[st]
        prop = float(group_props.get(str(st), 0.0))
        cnt = round(prop * n_segments)
        lines.append(f'    {name:<20} {cnt:>3} segs  ({_pct(prop):>6})  {_bar(prop, width=24)}')

    # PURER move distribution for this session's cue blocks.
    if purer_session_counts:
        lines.append('')
        lines.append(f'  PURER therapist-move distribution ({total_t_purer} turns):')
        for pid in _PURER_DISPLAY_ORDER:
            cnt = int(purer_session_counts.get(pid, 0))
            if cnt == 0:
                continue
            short = _PURER_SHORT.get(pid, str(pid))
            name = _PURER_NAME.get(pid, str(pid))
            prop = cnt / total_t_purer
            lines.append(
                f'    {short:<3} {name:<16} {cnt:>3} turns  ({_pct(prop):>6})  {_bar(prop, width=20)}'
            )
        lines.append('  ' + _PURER_DIRECTIONAL_NOTE)
    lines.append('')

    # ── 2. WHO WAS WHERE ──────────────────────────────────────────────────────
    lines.append('WHO WAS WHERE  [per participant this session]')
    lines.append('─' * 72)
    lines.append(f'  {"Participant":<22} {"Dominant Stage":<14} {"E[stage]":>8}  {"Seg":>4}')
    if participants_detail:
        for pid in sorted(participants_detail.keys()):
            pdet = participants_detail[pid]
            n_pseg = pdet.get('n_segments', 0)
            dom_name = pdet.get('dominant_stage_name', '?')
            pdf_p = sdf[sdf['participant_id'] == pid] if not sdf.empty else sdf
            p_est = _session_estage(pdf_p)
            p_est_s = f'{p_est:.2f}' if p_est is not None else '  ─'
            lines.append(f'  {pid:<22} {dom_name:<14} {p_est_s:>8}  {n_pseg:>4}')
    else:
        lines.append('  [no per-participant data]')
    lines.append('')

    # ── 3. WITHIN-SESSION TRANSITIONS ─────────────────────────────────────────
    lines.append('WITHIN-SESSION TRANSITIONS')
    lines.append('─' * 72)
    forward = sum(
        trans_mat.get(str(a), {}).get(str(b), 0)
        for a in stage_ids for b in stage_ids if b > a
    )
    backward = sum(
        trans_mat.get(str(a), {}).get(str(b), 0)
        for a in stage_ids for b in stage_ids if b < a
    )
    lateral = sum(
        trans_mat.get(str(a), {}).get(str(a), 0)
        for a in stage_ids
    )
    lines.append(f'  Forward: {forward}   Backward: {backward}   Lateral: {lateral}')
    all_trans = sorted(
        [
            (trans_mat.get(str(a), {}).get(str(b), 0), a, b)
            for a in stage_ids for b in stage_ids
            if trans_mat.get(str(a), {}).get(str(b), 0) > 0
        ],
        reverse=True,
    )
    non_self = [(cnt, fr, to) for cnt, fr, to in all_trans if fr != to]
    if non_self:
        lines.append('  Most common stage changes:')
        for cnt, fr, to in non_self[:5]:
            direction = 'forward' if to > fr else 'backward'
            lines.append(f'    {stage_names[fr]} → {stage_names[to]}  {cnt}x  [{direction}]')
    lines.append('')

    # ── 4. EXPRESSIONS BY STAGE (capped at top-2 per stage) ───────────────────
    lines.append('EXPRESSIONS BY STAGE  [≤2 per stage]')
    lines.append('─' * 72)
    any_ex = False
    for st in stage_ids:
        ex_list = exemplars.get(str(st), [])
        if not ex_list:
            continue
        any_ex = True
        lines.append(f'  {stage_names[st]}:')
        for ex in ex_list[:2]:
            conf = ex.get('confidence') or 0
            text = ex.get('text', '').strip()
            pid_ex = ex.get('participant_id', '')
            conf_str = f'conf={conf:.2f}' if conf else ''
            head = ', '.join(p for p in (pid_ex, conf_str) if p)
            lines.append(f'    [{head}]:' if head else '    :')
            lines.append(_wrap_quote(text, indent=6))
    if not any_ex:
        lines.append('  [no exemplars available]')
    lines.append('')

    # ── 5. NARRATIVE CONTEXT (LLM session summary, LAST) ──────────────────────
    lines.append('NARRATIVE CONTEXT (LLM-generated, not analysis)')
    lines.append('─' * 72)
    summary = (session_summaries or {}).get(session_id)
    if summary:
        lines.append(_wrap_text(summary, indent=2))
    else:
        lines.append('  [session summaries not generated]')
    lines.append('')

    lines.append(f'For mechanism (PURER × transition) detail see 09_supplementary/cue_response.txt;')
    lines.append(f'for the cross-session matrix see 04_per_session/_overview.txt.')

    content = '\n'.join(lines)
    out_dir = _paths.reports_per_session_dir(output_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'session_{session_id}.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


def generate_all_session_txt_reports(
    df: pd.DataFrame,
    session_reports: list,
    framework: dict,
    output_dir: str,
    llm_client=None,
    therapist_cue_config=None,
    df_all: pd.DataFrame = None,
    session_summaries: dict = None,
) -> list:
    """Generate per-session .txt files for every session in session_reports.

    Returns list of paths written.
    """
    paths = []
    for report in session_reports:
        sid = report.get('session_id', '')
        if not sid:
            continue
        try:
            path = generate_session_txt_report(
                df, sid, report, framework, output_dir,
                llm_client=llm_client,
                therapist_cue_config=therapist_cue_config,
                df_all=df_all,
                session_summaries=session_summaries,
            )
            paths.append(path)
        except Exception as e:
            print(f'  Warning: session txt report failed for {sid}: {e}')
    return paths
