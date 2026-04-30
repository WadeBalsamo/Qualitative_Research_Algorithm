"""Per-session human-readable text report generator."""

import os
from datetime import date

import pandas as pd

from process import output_paths as _paths
from ._formatting import (
    _bar, _pct, _wrap_quote, _collect_therapist_cue, _summarize_cue,
)
from .transition_report import _find_transition_examples_by_cohort_session


def _wrap_text(text: str, indent: int = 0, max_width: int = 80) -> str:
    """Word-wrap plain text (no quotes) to max_width."""
    if not text:
        return ''
    prefix = ' ' * indent
    out_lines = []
    for raw_line in text.replace('\r\n', '\n').split('\n'):
        if not raw_line.strip():
            out_lines.append('')
            continue
        current = prefix
        for word in raw_line.split():
            if len(current) + len(word) + 1 > max_width:
                out_lines.append(current)
                current = prefix + word
            else:
                current = current + word if current == prefix else current + ' ' + word
        out_lines.append(current)
    return '\n'.join(out_lines)


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

    session_summaries: pre-computed {session_id: summary_text} from session_summaries.py.
    If not provided or session not present, shows a placeholder.

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
    _cue_df = df_all if df_all is not None else df

    max_cue_words = 150
    if therapist_cue_config is not None:
        max_cue_words = getattr(therapist_cue_config, 'max_length_per_cue', 150)

    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    header = f'SESSION {session_id} — Session {snum}'
    if cohort_str:
        header += f'  [{cohort_str}]'
    lines.append(header)
    lines.append('=' * 68)
    lines.append(f'Generated: {date.today().isoformat()}')
    lines.append('')

    # ── Session instruction summary (from pre-computed session_summaries) ─────
    lines.append('SESSION INSTRUCTION SUMMARY')
    lines.append('─' * 35)
    summary = (session_summaries or {}).get(session_id)
    if summary:
        lines.append(_wrap_text(summary, indent=2))
    else:
        lines.append('  [session summaries not generated]')
    lines.append('')

    # ── Overview ──────────────────────────────────────────────────────────────
    lines.append('OVERVIEW')
    lines.append('─' * 10)
    lines.append(f'Participants: {n_participants}   Segments: {n_segments}')
    lines.append('')

    # ── Stage distribution ────────────────────────────────────────────────────
    lines.append('STAGE DISTRIBUTION (group, equal-weighted)')
    lines.append('─' * 45)
    for st in stage_ids:
        name = stage_names[st]
        prop = float(group_props.get(str(st), 0.0))
        cnt = round(prop * n_segments)
        lines.append(f'  {name:<20} {cnt:>3} segs  ({_pct(prop):>6})  {_bar(prop)}')
    lines.append('')

    # ── Best exemplar per stage ───────────────────────────────────────────────
    if any(exemplars.get(str(st)) for st in stage_ids):
        lines.append('Best exemplar per stage (highest confidence):')
        for st in stage_ids:
            ex_list = exemplars.get(str(st), [])
            if not ex_list:
                continue
            ex = ex_list[0]
            conf = ex.get('confidence') or 0
            text = ex.get('text', '').strip()
            pid_ex = ex.get('participant_id', '')
            lines.append(f'  {stage_names[st]} [{pid_ex}, conf={conf:.2f}]:')
            lines.append(_wrap_quote(text, indent=4))
        lines.append('')

    # ── Transition counts ─────────────────────────────────────────────────────
    lines.append('TRANSITIONS THIS SESSION')
    lines.append('─' * 28)
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
    lines.append(f'Forward: {forward}   Backward: {backward}   Lateral: {lateral}')
    lines.append('')

    all_trans = sorted(
        [
            (trans_mat.get(str(a), {}).get(str(b), 0), a, b)
            for a in stage_ids for b in stage_ids
            if trans_mat.get(str(a), {}).get(str(b), 0) > 0
        ],
        reverse=True,
    )

    if all_trans:
        lines.append('Most common transitions:')
        for cnt, fr, to in all_trans[:5]:
            direction = 'forward' if to > fr else ('backward' if to < fr else 'lateral')
            lines.append(f'  {stage_names[fr]} → {stage_names[to]}  {cnt}x  [{direction}]')
        lines.append('')

    # ── Transition examples with therapist cue ────────────────────────────────
    non_self = [(cnt, fr, to) for cnt, fr, to in all_trans if fr != to]
    if non_self:
        lines.append('Transition examples (with therapist cue):')
        sdf = df[df['session_id'] == session_id]
        for cnt, fr, to in non_self[:3]:
            examples = _find_transition_examples_by_cohort_session(sdf, fr, to)
            if not examples:
                continue
            lines.append(f'  ── {stage_names[fr]} → {stage_names[to]} ──')
            for ex in examples[:2]:
                lines.append(
                    f'  [{ex["participant_id"]}  {ex["session_id"]}  '
                    f'seg{ex["from_seg_idx"]:04d}→seg{ex["to_seg_idx"]:04d}]'
                )
                lines.append(
                    f'    FROM: [{stage_names[fr]}={ex["from_conf"]:.2f}] '
                    + _wrap_quote(ex['from_text'].strip(), indent=12).lstrip()
                )
                cue_raw = _collect_therapist_cue(
                    _cue_df, ex['session_id'],
                    ex.get('from_end_ms', 0), ex.get('to_start_ms', 0),
                )
                if cue_raw:
                    cue_text, was_summarized = _summarize_cue(cue_raw, llm_client, max_cue_words)
                    cue_words = len(cue_text.split())
                    marker = ', summarized' if was_summarized else ''
                    lines.append(
                        f'     CUE: [therapist, {cue_words} words{marker}] '
                        + _wrap_quote(cue_text.strip(), indent=12).lstrip()
                    )
                lines.append(
                    f'      TO: [{stage_names[to]}={ex["to_conf"]:.2f}] '
                    + _wrap_quote(ex['to_text'].strip(), indent=12).lstrip()
                )
        lines.append('')

    # ── Per-participant breakdown ──────────────────────────────────────────────
    lines.append('PER-PARTICIPANT BREAKDOWN')
    lines.append('─' * 28)
    for pid in sorted(participants_detail.keys()):
        pdet = participants_detail[pid]
        n_pseg = pdet.get('n_segments', 0)
        dom_name = pdet.get('dominant_stage_name', '?')
        props = pdet.get('stage_proportions', {})
        lines.append(f'{pid}  ({n_pseg} segments)')
        lines.append(f'  Dominant stage: {dom_name}')
        pct_parts = ' | '.join(
            f'{stage_names[st]} {_pct(float(props.get(str(st), 0.0)))}'
            for st in stage_ids
        )
        lines.append(f'  {pct_parts}')
        pdf = df[(df['participant_id'] == pid) & (df['session_id'] == session_id)]
        if not pdf.empty:
            has_conf = 'llm_confidence_primary' in pdf.columns
            if has_conf and pdf['llm_confidence_primary'].notna().any():
                best_row = pdf.sort_values('llm_confidence_primary', ascending=False).iloc[0]
            else:
                best_row = pdf.iloc[0]
            best_text = str(best_row.get('text', '')).strip()
            best_stage = int(best_row.get('final_label', 0))
            best_conf = float(best_row.get('llm_confidence_primary', 0) or 0)
            lines.append(
                f'  Best expression [{stage_names.get(best_stage, str(best_stage))}={best_conf:.2f}]:'
            )
            lines.append(_wrap_quote(best_text, indent=4))
        lines.append('')

    content = '\n'.join(lines)
    out_dir = os.path.join(_paths.human_reports_dir(output_dir), 'per_session')
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
