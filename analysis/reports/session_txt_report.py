"""Per-session human-readable text report generator."""

import os
from collections import defaultdict
from datetime import date

import pandas as pd

from process import output_paths as _paths
from ._formatting import (
    _bar, _pct, _wrap_quote, _collect_therapist_cue, _summarize_cue,
    _collect_cue_block_purer_profile, _format_purer_profile,
    _PURER_SHORT, _PURER_NAME,
)
from .transition_report import _find_transition_examples_by_cohort_session


_PURER_DISPLAY_ORDER = [3, 0, 4, 2, 1]  # E, P, R2, R, U — typical frequency


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


def _compute_session_purer_by_transition(df_all: pd.DataFrame, session_id: str) -> dict:
    """Compute PURER distribution per (from_stage, to_stage) for one session.

    Uses timestamp-window logic identical to _collect_therapist_cue: therapist
    segments that overlap the window between two consecutive participant segments.

    Returns {(from_stage, to_stage): {'counts': {purer_id: n}, 'total_mediated': n}}.
    """
    required = {'session_id', 'speaker', 'final_label', 'start_time_ms', 'end_time_ms'}
    if not required.issubset(df_all.columns):
        return {}
    if 'purer_primary' not in df_all.columns:
        return {}

    sdf = df_all[df_all['session_id'] == session_id]

    participant_rows = (
        sdf[
            (sdf['speaker'] == 'participant')
            & sdf['final_label'].notna()
        ]
        .sort_values('start_time_ms')
        .reset_index(drop=True)
    )
    if len(participant_rows) < 2:
        return {}

    therapist_rows = sdf[sdf['speaker'] == 'therapist']

    result: dict = defaultdict(lambda: {'counts': defaultdict(int), 'total_mediated': 0})

    for i in range(len(participant_rows) - 1):
        from_row = participant_rows.iloc[i]
        to_row = participant_rows.iloc[i + 1]
        from_end_ms = int(from_row['end_time_ms'])
        to_start_ms = int(to_row['start_time_ms'])
        from_stage = int(from_row['final_label'])
        to_stage = int(to_row['final_label'])

        if to_start_ms <= from_end_ms:
            continue

        between = therapist_rows[
            (therapist_rows['start_time_ms'] < to_start_ms)
            & (therapist_rows['end_time_ms'] > from_end_ms)
        ]
        if between.empty:
            continue

        # Only count mediated blocks that have at least one PURER label
        labeled = between[between['purer_primary'].notna()]
        if labeled.empty:
            continue

        key = (from_stage, to_stage)
        result[key]['total_mediated'] += 1
        dominant = labeled['purer_primary'].value_counts().idxmax()
        result[key]['counts'][int(dominant)] += 1

    # Convert defaultdicts to plain dicts for cleaner access
    return {k: {'counts': dict(v['counts']), 'total_mediated': v['total_mediated']}
            for k, v in result.items()}


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

    # ── Precompute dominant / secondary stage ─────────────────────────────────
    stage_props_sorted = sorted(
        [(float(group_props.get(str(st), 0.0)), st) for st in stage_ids],
        reverse=True,
    )
    dominant_stage = stage_props_sorted[0][1] if stage_props_sorted else None
    secondary_stage = (
        stage_props_sorted[1][1]
        if len(stage_props_sorted) > 1 and stage_props_sorted[1][0] > 0
        else None
    )

    # ── Precompute PURER therapist stats for this session ────────────────────
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

    # ── Header ────────────────────────────────────────────────────────────────
    header = f'SESSION {session_id} — Session {snum}'
    if cohort_str:
        header += f'  [{cohort_str}]'
    lines.append(header)
    lines.append('=' * 68)
    lines.append(f'Generated: {date.today().isoformat()}')
    lines.append('')

    # ── Session instruction summary ───────────────────────────────────────────
    lines.append('SESSION INSTRUCTION SUMMARY')
    lines.append('─' * 35)
    summary = (session_summaries or {}).get(session_id)
    if summary:
        lines.append(_wrap_text(summary, indent=2))
    else:
        lines.append('  [session summaries not generated]')
    lines.append('')

    # ── Overview with inline VAAMR + PURER dominant ───────────────────────────
    lines.append('OVERVIEW')
    lines.append('─' * 10)
    lines.append(f'Participants: {n_participants}   Segments: {n_segments}')
    vaamr_str = (
        f'VAAMR dominant: {stage_names[dominant_stage]} '
        f'({_pct(float(group_props.get(str(dominant_stage), 0.0)))})'
        if dominant_stage is not None else ''
    )
    if purer_dominant_id is not None:
        purer_dom_name = _PURER_NAME.get(purer_dominant_id, str(purer_dominant_id))
        purer_dom_pct = _pct(purer_session_counts[purer_dominant_id] / total_t_purer)
        purer_str = f'PURER dominant: {purer_dom_name} ({purer_dom_pct}, n={total_t_purer} turns)'
        if vaamr_str:
            lines.append(f'{vaamr_str}   |   {purer_str}')
        else:
            lines.append(purer_str)
    elif vaamr_str:
        lines.append(vaamr_str)
    lines.append('')

    # ── Stage distribution ────────────────────────────────────────────────────
    lines.append('STAGE DISTRIBUTION (group, equal-weighted)')
    lines.append('─' * 45)
    for st in stage_ids:
        name = stage_names[st]
        prop = float(group_props.get(str(st), 0.0))
        cnt = round(prop * n_segments)
        lines.append(f'  {name:<20} {cnt:>3} segs  ({_pct(prop):>6})  {_bar(prop)}')
    lines.append(f'  [see session_{session_id}_stage_timeline.png for temporal view]')
    lines.append('')

    # ── PURER therapist move distribution ─────────────────────────────────────
    if purer_session_counts:
        lines.append('PURER THERAPIST MOVE DISTRIBUTION (this session)')
        lines.append('─' * 50)
        for pid in _PURER_DISPLAY_ORDER:
            cnt = int(purer_session_counts.get(pid, 0))
            if cnt == 0:
                continue
            short = _PURER_SHORT.get(pid, str(pid))
            name  = _PURER_NAME.get(pid, str(pid))
            prop  = cnt / total_t_purer
            lines.append(
                f'  {short:<3} {name:<18} {cnt:>3} turns  ({_pct(prop):>6})  {_bar(prop)}'
            )
        lines.append(f'  [{total_t_purer} therapist turns classified]')
        lines.append('')

    # ── Expressions of dominant stage + secondary stage ───────────────────────
    for rank, (rank_label, st) in enumerate([
        ('dominant stage', dominant_stage),
        ('secondary stage', secondary_stage),
    ]):
        if st is None:
            continue
        ex_list = exemplars.get(str(st), [])
        if not ex_list:
            continue
        ex = ex_list[0]
        prop = float(group_props.get(str(st), 0.0))
        conf = ex.get('confidence') or 0
        text = ex.get('text', '').strip()
        pid_ex = ex.get('participant_id', '')
        lines.append(f'EXPRESSIONS OF {stage_names[st].upper()}')
        lines.append(f'  [{rank_label} — {_pct(prop)}]')
        lines.append(f'  [{pid_ex}, conf={conf:.2f}]:')
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

    # ── PURER by transition type (this session) ───────────────────────────────
    if df_all is not None and 'purer_primary' in df_all.columns:
        purer_by_trans = _compute_session_purer_by_transition(df_all, session_id)
        if purer_by_trans:
            lines.append('PURER BY TRANSITION TYPE (this session)')
            lines.append('─' * 43)
            # Sort by total mediated descending, matching all_trans order
            trans_sorted = sorted(
                purer_by_trans.items(),
                key=lambda kv: kv[1]['total_mediated'],
                reverse=True,
            )
            for (fr, to), stats in trans_sorted:
                total_med = stats['total_mediated']
                counts = stats['counts']
                direction = '→' if to > fr else ('←' if to < fr else '↔')
                lines.append(
                    f'  {stage_names[fr]} {direction} {stage_names[to]}'
                    f'  (n={total_med} mediated cue blocks)'
                )
                for pid in _PURER_DISPLAY_ORDER:
                    cnt = counts.get(pid, 0)
                    if cnt == 0:
                        continue
                    short = _PURER_SHORT.get(pid, str(pid))
                    name  = _PURER_NAME.get(pid, str(pid))
                    frac  = cnt / total_med
                    bar   = '█' * int(frac * 20)
                    lines.append(
                        f'    {short:<2} {name:<18}  {bar:<20}  {_pct(frac):>6}  (n={cnt})'
                    )
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
                    annotate_purer=True,
                )
                if cue_raw:
                    cue_text, was_summarized = _summarize_cue(cue_raw, llm_client, max_cue_words)
                    cue_words = len(cue_text.split())
                    marker = ', summarized' if was_summarized else ''
                    purer_profile = _collect_cue_block_purer_profile(
                        _cue_df, ex['session_id'],
                        ex.get('from_end_ms', 0), ex.get('to_start_ms', 0),
                        include_secondary=True,
                    )
                    purer_str = _format_purer_profile(purer_profile)
                    purer_part = f' | PURER: {purer_str}' if purer_str else ''
                    lines.append(
                        f'     CUE: [therapist, {cue_words} words{marker}{purer_part}] '
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

        # Participant-level stage transitions for this session
        pdf = df[(df['participant_id'] == pid) & (df['session_id'] == session_id)]
        if not pdf.empty and 'final_label' in pdf.columns:
            sort_col = 'segment_index' if 'segment_index' in pdf.columns else 'start_time_ms'
            pseq = [int(v) for v in pdf.sort_values(sort_col)['final_label'].tolist() if pd.notna(v)]
            if len(pseq) >= 2:
                trans_counts: dict = defaultdict(int)
                p_fwd = p_bwd = p_lat = 0
                for a, b in zip(pseq, pseq[1:]):
                    trans_counts[(a, b)] += 1
                    if b > a:
                        p_fwd += 1
                    elif b < a:
                        p_bwd += 1
                    else:
                        p_lat += 1
                lines.append(
                    f'  Transitions: {p_fwd} forward, {p_bwd} backward, {p_lat} lateral'
                )
                trans_sorted = sorted(trans_counts.items(), key=lambda kv: -kv[1])
                for (fa, fb), tc in trans_sorted:
                    direction = 'forward' if fb > fa else ('backward' if fb < fa else 'lateral')
                    lines.append(
                        f'    {stage_names[fa]} → {stage_names[fb]}  {tc}x  [{direction}]'
                    )

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
