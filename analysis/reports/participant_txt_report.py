"""Per-participant human-readable text report generator."""

import os
import re
from datetime import date

import pandas as pd

from process import output_paths as _paths
from ._formatting import _bar, _pct, _wrap_quote, _summarize_participant_text


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

    participant_summaries_config: optional config with .enabled and .max_words_per_session.
    When enabled, each session block includes an LLM summary of the participant's own language.

    Returns path to written file.
    """
    stage_ids = sorted(framework.keys())
    stage_names = {sid: framework[sid].get('short_name', f'Stage {sid}') for sid in stage_ids}

    cohort_id = participant_json.get('cohort_id')
    session_ids = participant_json.get('session_ids', [])
    sessions_detail = participant_json.get('sessions', {})
    trend = participant_json.get('progression_trend', 0.0)
    stage_exemplars_overall = participant_json.get('stage_exemplars_overall', {})
    n_sessions = participant_json.get('n_sessions', len(session_ids))

    cohort_str = f'Cohort {cohort_id}' if cohort_id is not None else ''

    pdf_all = df[df['participant_id'] == participant_id]
    n_total_segs = len(pdf_all)
    stage_counts_overall = pdf_all['final_label'].value_counts().to_dict()

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

    # ── Header ────────────────────────────────────────────────────────────────
    header = f'PARTICIPANT {participant_id}'
    if cohort_str:
        header += f'  [{cohort_str}]'
    lines.append(header)
    lines.append('=' * 68)
    lines.append(f'Generated: {date.today().isoformat()}')
    lines.append('')

    # ── Overview ──────────────────────────────────────────────────────────────
    lines.append('OVERVIEW')
    lines.append('─' * 10)
    progression_word = (
        'advancing' if trend > 0.1 else
        'regressing' if trend < -0.1 else
        'stable'
    )
    lines.append(f'Sessions attended:     {n_sessions}')
    lines.append(f'Session IDs:           {", ".join(session_ids)}')
    lines.append(f'Overall progression:   {progression_word}  (slope={trend:+.2f}/session)')
    lines.append('')

    # ── Participation statistics table ────────────────────────────────────────
    lines.append('PARTICIPATION STATISTICS')
    lines.append('─' * 58)
    lines.append(f'{"Session":<10} {"Seg":>4}  {"Dominant Stage":<22} {"Conf":>5}   {"Prog Score":>10}')
    lines.append('─' * 58)
    for sid in session_ids:
        sdet = sessions_detail.get(sid, {})
        n_seg = sdet.get('n_segments', 0)
        dom_name = sdet.get('dominant_stage_name', '?')
        mean_conf = sdet.get('mean_confidence')
        conf_str = f'{mean_conf:.2f}' if mean_conf is not None else '  ─'
        prog = sdet.get('progression_score')
        prog_str = f'{prog:.2f}' if prog is not None else '  ─'
        lines.append(f'{sid:<10} {n_seg:>4}  {dom_name:<22} {conf_str:>5}   {prog_str:>10}')
    lines.append('')

    # ── Overall stage distribution ────────────────────────────────────────────
    lines.append('STAGE DISTRIBUTION (all sessions combined)')
    lines.append('─' * 45)
    for st in stage_ids:
        name = stage_names[st]
        cnt = int(stage_counts_overall.get(st, 0))
        prop = cnt / n_total_segs if n_total_segs > 0 else 0.0
        lines.append(f'  {name:<20} {cnt:>3} segs  ({_pct(prop):>6})  {_bar(prop)}')
    lines.append('')

    # ── Longitudinal trajectory ───────────────────────────────────────────────
    lines.append('LONGITUDINAL TRAJECTORY  [session-by-session participant expression]')
    lines.append('─' * 66)
    lines.append('')

    prev_dom = None
    for sid in session_ids:
        sdet = sessions_detail.get(sid, {})
        dom_stage = sdet.get('dominant_stage')
        dom_name = stage_names.get(dom_stage, '?') if dom_stage is not None else '?'

        if prev_dom is None:
            direction_label = ''
        elif dom_stage is None:
            direction_label = ''
        elif dom_stage > prev_dom:
            direction_label = f' — ADVANCE from {stage_names.get(prev_dom, str(prev_dom))}'
        elif dom_stage < prev_dom:
            direction_label = f' — REGRESS from {stage_names.get(prev_dom, str(prev_dom))}'
        else:
            direction_label = ' — STABLE'

        label_block = f'[{dom_name}{direction_label}]'
        bar_len = max(0, 56 - len(sid) - len(label_block))
        lines.append(f'━━ SESSION {sid}  {label_block} {"━" * bar_len}')

        # Optional LLM summary of all participant text in this session
        pdf_s = df[(df['participant_id'] == participant_id) & (df['session_id'] == sid)]
        if summaries_enabled and not pdf_s.empty:
            all_text = ' '.join(
                str(t).strip() for t in pdf_s.sort_values(
                    'segment_index' if 'segment_index' in pdf_s.columns else pdf_s.columns[0]
                )['text'] if str(t).strip()
            )
            if all_text:
                p_summary, _ = _summarize_participant_text(all_text, llm_client, max_summary_words)
                lines.append('  Participant summary:')
                lines.append(_wrap_text(p_summary, indent=4))
                lines.append('')

        # Best expression (highest-confidence segment) in this session
        if not pdf_s.empty:
            has_conf = 'llm_confidence_primary' in pdf_s.columns
            if has_conf and pdf_s['llm_confidence_primary'].notna().any():
                best_row = pdf_s.sort_values('llm_confidence_primary', ascending=False).iloc[0]
            else:
                best_row = pdf_s.iloc[0]
            best_text = str(best_row.get('text', '')).strip()
            best_stage = int(best_row.get('final_label', 0))
            best_conf = float(best_row.get('llm_confidence_primary', 0) or 0)
            lines.append(
                f'  Best expression [{stage_names.get(best_stage, str(best_stage))}={best_conf:.2f}]:'
            )
            lines.append(_wrap_quote(best_text, indent=4))
        else:
            lines.append('  [no data for this session]')
        lines.append('')

        if dom_stage is not None:
            prev_dom = dom_stage

    # ── Between-session transitions ───────────────────────────────────────────
    if len(session_ids) > 1:
        lines.append('BETWEEN-SESSION TRANSITIONS')
        lines.append('─' * 30)
        for i in range(len(session_ids) - 1):
            from_sid = session_ids[i]
            to_sid = session_ids[i + 1]
            from_dom = sessions_detail.get(from_sid, {}).get('dominant_stage')
            to_dom = sessions_detail.get(to_sid, {}).get('dominant_stage')
            from_name = stage_names.get(from_dom, '?') if from_dom is not None else '?'
            to_name = stage_names.get(to_dom, '?') if to_dom is not None else '?'
            if from_dom is not None and to_dom is not None:
                direction = (
                    'advance' if to_dom > from_dom else
                    'regress' if to_dom < from_dom else
                    'stable'
                )
            else:
                direction = '?'
            lines.append(
                f'  {from_sid} → {to_sid}:  {from_name} → {to_name}   [{direction}]'
            )
        lines.append('')

    # ── Best expressions by stage ─────────────────────────────────────────────
    lines.append('BEST EXPRESSIONS BY STAGE')
    lines.append('─' * 28)
    for st in stage_ids:
        ex = stage_exemplars_overall.get(str(st))
        if not ex:
            continue
        conf = ex.get('confidence') or 0
        text = ex.get('text', '').strip()
        sid_ex = ex.get('session_id', '')
        sid_str = f'{sid_ex}, ' if sid_ex else ''
        conf_str = f'conf={conf:.2f}' if conf else ''
        lines.append(f'  {stage_names[st]} [{sid_str}{conf_str}]:')
        lines.append(_wrap_quote(text, indent=4))
    lines.append('')

    content = '\n'.join(lines)

    # Strip leading "Participant_" (case-insensitive) to avoid participant_Participant_MM.txt
    clean_id = re.sub(r'^[Pp]articipant_', '', participant_id)
    out_dir = os.path.join(_paths.human_reports_dir(output_dir), 'per_participant')
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
