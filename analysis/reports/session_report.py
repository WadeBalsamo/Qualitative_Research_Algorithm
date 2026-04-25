"""Comprehensive per-session theme analysis report generator."""

import os
from collections import defaultdict
from datetime import date

import pandas as pd

from ..loader import sort_session_ids
from process import output_paths as _paths
from ._formatting import _bar, _pct, _wrap_quote


def generate_comprehensive_session_report(
    df: pd.DataFrame,
    framework: dict,
    session_reports: list,
    output_dir: str,
) -> str:
    """Generate comprehensive_theme_analysis.txt in the output root.

    Integrates the per-session theme distribution, exemplars, transitions,
    and overall consistency analysis from the pipeline's session JSON reports.

    Returns path to written file.
    """
    stage_ids = sorted(framework.keys())
    stage_names = {sid: framework[sid].get('short_name', f'Stage {sid}') for sid in stage_ids}

    # Index session reports by session_id
    sessions_by_id = {r['session_id']: r for r in session_reports if r}
    session_id_order = sort_session_ids(list(sessions_by_id.keys()))

    total_segments = sum(r.get('n_segments', 0) for r in session_reports if r)
    total_participants = df['participant_id'].nunique()

    lines = []
    lines.append('QRA COMPREHENSIVE THEME ANALYSIS')
    lines.append('=' * 60)
    lines.append(f'Generated:    {date.today().isoformat()}')
    lines.append(f'Sessions:     {len(session_id_order)}')
    lines.append(f'Participants: {total_participants}')
    lines.append(f'Segments:     {total_segments}')
    lines.append('')

    # Track theme presence across sessions for consistency analysis
    theme_session_presence = defaultdict(set)

    for sid in session_id_order:
        report = sessions_by_id.get(sid, {})
        if not report:
            continue

        snum = report.get('session_number', '?')
        n_seg = report.get('n_segments', 0)
        n_part = report.get('n_participants', 0)
        group_props = report.get('group_stage_proportions', {})
        exemplars = report.get('stage_exemplars', {})
        trans_mat = report.get('stage_transition_matrix', {})

        lines.append(f'── SESSION {sid} ' + '─' * max(1, 46 - len(sid)))
        lines.append(f'Session Number: {snum}  |  Participants: {n_part}  |  Segments: {n_seg}')
        lines.append('')
        lines.append('  Theme Distribution:')

        for st in stage_ids:
            name = stage_names[st]
            prop = float(group_props.get(str(st), 0.0))
            cnt = round(prop * n_seg)
            bar = _bar(prop, width=25)
            lines.append(f'    {name:<20} {cnt:>3} segs  ({_pct(prop):>6})  {bar}')
            if prop > 0:
                theme_session_presence[name].add(sid)

        lines.append('')

        # Exemplar quotes per theme
        has_any_exemplar = any(exemplars.get(str(st)) for st in stage_ids)
        if has_any_exemplar:
            lines.append('  Best Exemplars by Theme:')
            for st in stage_ids:
                ex_list = exemplars.get(str(st), [])
                if not ex_list:
                    continue
                lines.append(f'    {stage_names[st]}:')
                for i, ex in enumerate(ex_list[:2], 1):
                    conf = ex.get('confidence', 0)
                    text = ex.get('text', '').strip()
                    pid_ex = ex.get('participant_id', '')
                    lines.append(f'      {i}. [{pid_ex}] (confidence: {conf:.2f})')
                    lines.append(_wrap_quote(text, indent=9))
            lines.append('')

        # Transition summary
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
        lines.append(f'  Transitions: {forward} forward, {backward} backward, {lateral} lateral')
        lines.append('')

    # Theme consistency analysis
    lines.append('')
    lines.append('THEME CONSISTENCY ANALYSIS')
    lines.append('-' * 40)
    n_sessions_total = len(session_id_order)

    themes_all = [t for t, sids in theme_session_presence.items()
                  if len(sids) == n_sessions_total]
    themes_one = [t for t, sids in theme_session_presence.items() if len(sids) == 1]
    themes_most = [t for t, sids in theme_session_presence.items()
                   if 2 <= len(sids) < n_sessions_total]

    lines.append(f'Themes present in all {n_sessions_total} sessions:')
    if themes_all:
        for t in sorted(themes_all):
            lines.append(f'  - {t}')
    else:
        lines.append('  (none)')

    lines.append('')
    lines.append('Themes present in only one session:')
    if themes_one:
        for t in sorted(themes_one):
            sid_val = next(iter(theme_session_presence[t]))
            lines.append(f'  - {t} (session {sid_val})')
    else:
        lines.append('  (none)')

    lines.append('')
    if themes_most:
        lines.append('Themes present in some but not all sessions:')
        for t in sorted(themes_most, key=lambda x: -len(theme_session_presence[x])):
            cnt_s = len(theme_session_presence[t])
            lines.append(f'  - {t}: {cnt_s}/{n_sessions_total} sessions')
        lines.append('')

    content = '\n'.join(lines)
    os.makedirs(_paths.human_reports_dir(output_dir), exist_ok=True)
    path = os.path.join(_paths.human_reports_dir(output_dir), 'comprehensive_theme_analysis.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path
