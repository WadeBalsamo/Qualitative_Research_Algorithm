"""
analysis/text_reports.py
------------------------
Human-readable text report generators for the QRA analysis module.

Writes .txt files to the output root (easy access) and to constructs/.
"""

import os
import re
from collections import defaultdict
from datetime import date

import pandas as pd

from .loader import sort_session_ids
from .stage_progression import compute_cross_session_transitions


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _bar(value: float, width: int = 30) -> str:
    """ASCII progress bar for a proportion (0.0–1.0)."""
    filled = int(round(value * width))
    return '█' * filled + '░' * (width - filled)


def _pct(value: float) -> str:
    return f'{value * 100:.1f}%'


def _wrap_quote(text: str, indent: int = 9, max_width: int = 80) -> str:
    """Wrap a quoted text to max_width, indenting continuation lines."""
    words = text.split()
    lines = []
    current = ' ' * indent + '"'
    prefix_len = indent + 1
    for word in words:
        if len(current) + len(word) + 1 > max_width:
            lines.append(current)
            current = ' ' * (prefix_len) + word
        else:
            if current == ' ' * indent + '"':
                current += word
            else:
                current += ' ' + word
    current += '"'
    lines.append(current)
    return '\n'.join(lines)


def _find_transition_examples(df: pd.DataFrame, from_stage: int, to_stage: int,
                               framework: dict, n: int = 2) -> list:
    """Find example pairs of consecutive segments going from_stage → to_stage."""
    examples = []
    for (pid, sid), group in df.groupby(['participant_id', 'session_id']):
        group = group.sort_values('segment_index')
        labels = group['final_label'].astype(int).tolist()
        texts = group['text'].tolist()
        for i in range(len(labels) - 1):
            if labels[i] == from_stage and labels[i + 1] == to_stage:
                examples.append({
                    'participant_id': pid,
                    'session_id': sid,
                    'from_text': texts[i],
                    'to_text': texts[i + 1],
                })
                if len(examples) >= n:
                    return examples
    return examples


def _cross_session_example(df: pd.DataFrame, pid: str, sessions: list,
                            from_idx: int) -> dict:
    """Find a representative quote from the session at from_idx in sessions."""
    sid = sessions[from_idx]
    sdf = df[(df['participant_id'] == pid) & (df['session_id'] == sid)]
    if sdf.empty:
        return {}
    # Pick highest-confidence segment
    if 'llm_confidence_primary' in sdf.columns and sdf['llm_confidence_primary'].notna().any():
        row = sdf.sort_values('llm_confidence_primary', ascending=False).iloc[0]
    else:
        row = sdf.iloc[0]
    return {'session_id': sid, 'text': str(row.get('text', '')),
            'confidence': float(row.get('llm_confidence_primary', 0) or 0)}


# -----------------------------------------------------------------------
# 1. Comprehensive session analysis report
# -----------------------------------------------------------------------

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
    path = os.path.join(output_dir, 'comprehensive_theme_analysis.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


# -----------------------------------------------------------------------
# 2. Per-stage construct text reports
# -----------------------------------------------------------------------

def generate_stage_text_report(
    df: pd.DataFrame,
    stage_id: int,
    framework: dict,
    stage_report: dict,
    output_dir: str,
) -> str:
    """Generate a human-readable text report for a single VA-MR stage.

    Saved to reports/analysis/constructs/stage_{name}.txt.
    Returns path to written file.
    """
    stage_info = framework.get(stage_id, {})
    stage_name = stage_info.get('short_name', f'Stage {stage_id}')
    full_name = stage_info.get('name', stage_name)
    definition = stage_info.get('definition', '(no definition available)')
    slug = re.sub(r'[^a-z0-9]+', '_', stage_name.lower()).strip('_')

    overall_prev = stage_report.get('overall_prevalence', 0.0)
    n_total = stage_report.get('n_segments_total', 0)
    n_stage = stage_report.get('n_segments_this_stage', 0)
    trend = stage_report.get('longitudinal_trend', 0.0)
    prev_by_session = stage_report.get('prevalence_by_session_number', {})
    prev_by_participant = stage_report.get('prevalence_by_participant', {})
    top_exemplars = stage_report.get('top_exemplars', [])
    co_codes = stage_report.get('co_occurring_codes', [])

    trend_desc = (
        'increasing (advancing)' if trend > 0.02 else
        'decreasing' if trend < -0.02 else
        'stable'
    )

    lines = []
    lines.append(f'STAGE: {full_name} ({stage_name})')
    lines.append('=' * (len(full_name) + len(stage_name) + 10))
    lines.append('')
    lines.append('Definition:')
    # Word-wrap definition
    words = definition.split()
    line_buf = '  '
    for word in words:
        if len(line_buf) + len(word) + 1 > 78:
            lines.append(line_buf)
            line_buf = '  ' + word
        else:
            line_buf += (' ' if line_buf.strip() else '') + word
    lines.append(line_buf)
    lines.append('')

    lines.append(f'Overall prevalence: {_pct(overall_prev)} ({n_stage}/{n_total} segments)')
    lines.append(f'Longitudinal trend: {trend:+.4f}/session ({trend_desc})')
    lines.append(f'[See figure: stage_{slug}_prevalence.png]')
    lines.append('')

    if prev_by_session:
        lines.append('Prevalence by Session:')
        for snum in sorted(prev_by_session.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
            v = float(prev_by_session[snum])
            lines.append(f'  Session {snum}: {_pct(v):>7}  {_bar(v, 20)}')
        lines.append('')

    if prev_by_participant:
        lines.append('Prevalence by Participant (top 10):')
        sorted_parts = sorted(prev_by_participant.items(), key=lambda kv: -kv[1])
        for pid, v in sorted_parts[:10]:
            lines.append(f'  {pid:<25} {_pct(float(v)):>7}  {_bar(float(v), 20)}')
        lines.append('')

    if top_exemplars:
        lines.append('Top Exemplar Quotes:')
        for i, ex in enumerate(top_exemplars[:5], 1):
            pid_ex = ex.get('participant_id', '')
            sid_ex = ex.get('session_id', '')
            conf = ex.get('confidence', 0)
            text = ex.get('text', '').strip()
            justification = ex.get('justification', '').strip()
            lines.append(f'  {i}. [{pid_ex} / {sid_ex}]  confidence={conf:.2f}')
            lines.append(_wrap_quote(text, indent=5))
            if justification:
                lines.append(f'       Justification: {justification[:120]}')
            lines.append('')

    if co_codes:
        lines.append('Co-occurring Codebook Codes (by lift):')
        for c in co_codes[:8]:
            lines.append(f'  - {c["code_id"]:<30} lift={c["lift"]:.2f}  count={c["count"]}')
        lines.append('')

    content = '\n'.join(lines)
    out_dir = os.path.join(output_dir, 'reports', 'analysis', 'constructs')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'stage_{slug}.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


def generate_all_stage_text_reports(
    df: pd.DataFrame,
    framework: dict,
    stage_reports: list,
    output_dir: str,
) -> list:
    """Generate text reports for all stages. Returns list of paths."""
    paths = []
    for stage_report in stage_reports:
        stage_id = stage_report.get('stage_id')
        if stage_id is None:
            continue
        try:
            path = generate_stage_text_report(df, stage_id, framework, stage_report, output_dir)
            if path:
                paths.append(path)
        except Exception as e:
            print(f"  Warning: stage text report failed for stage {stage_id}: {e}")
    return paths


# -----------------------------------------------------------------------
# 3. State transition explanation
# -----------------------------------------------------------------------

def generate_transition_explanation(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
) -> str:
    """Generate state_transition_explanation.txt in the output root.

    Explains both within-session and between-session transition heatmaps,
    with example quotes for the most common transitions.
    Returns path to written file.
    """
    from .stage_progression import compute_state_transition_matrix

    stage_ids = sorted(framework.keys())
    stage_names = {sid: framework[sid].get('short_name', f'Stage {sid}') for sid in stage_ids}

    within_matrix = compute_state_transition_matrix(df, framework)
    cross_matrix, participant_sequences = compute_cross_session_transitions(df, framework)

    lines = []
    lines.append('STATE TRANSITION ANALYSIS — EXPLANATION')
    lines.append('=' * 60)
    lines.append(f'Generated: {date.today().isoformat()}')
    lines.append('')
    lines.append(
        'This document explains the two state transition heatmaps produced\n'
        'by the QRA analysis pipeline.\n'
    )

    # ── Within-session ──
    lines.append('WITHIN-SESSION TRANSITIONS  (state_transition_heatmap.png)')
    lines.append('-' * 60)
    lines.append(
        'Each cell [row → col] counts how many times a classified segment\n'
        'with theme ROW was immediately followed by a segment with theme COL,\n'
        'within the same session. Rows = "from", columns = "to".\n'
    )

    # Summarize top transitions
    pairs = []
    for i, fr in enumerate(stage_ids):
        for j, to in enumerate(stage_ids):
            cnt = int(within_matrix.iloc[i, j])
            if cnt > 0:
                pairs.append((cnt, fr, to))
    pairs.sort(reverse=True)

    lines.append('Most common within-session transitions:')
    for cnt, fr, to in pairs[:8]:
        direction = 'stay' if fr == to else ('forward' if to > fr else 'backward')
        lines.append(f'  {stage_names[fr]:<20} → {stage_names[to]:<20} {cnt:>4}x  [{direction}]')
    lines.append('')

    # Example quotes for top 3 non-self transitions
    non_self = [(cnt, fr, to) for cnt, fr, to in pairs if fr != to]
    if non_self:
        lines.append('Example quotes for top within-session transitions:')
        for cnt, fr, to in non_self[:3]:
            examples = _find_transition_examples(df, fr, to, framework, n=1)
            if not examples:
                continue
            ex = examples[0]
            lines.append(f'\n  {stage_names[fr]} → {stage_names[to]}'
                         f'  ({ex["participant_id"]}, session {ex["session_id"]}):')
            from_text = ex['from_text'].strip()
            to_text = ex['to_text'].strip()
            lines.append(f'  FROM: {_wrap_quote(from_text, indent=8).lstrip()}')
            lines.append(f'    TO: {_wrap_quote(to_text, indent=8).lstrip()}')
        lines.append('')

    # ── Between-session ──
    lines.append('')
    lines.append('BETWEEN-SESSION TRANSITIONS  (cross_session_transition_heatmap.png)')
    lines.append('-' * 60)
    lines.append(
        'Each cell [row → col] counts how many times a participant\'s\n'
        'DOMINANT theme in session N was ROW and their dominant theme in\n'
        'session N+1 was COL. This captures program-level progression.\n'
    )

    cross_pairs = []
    for i, fr in enumerate(stage_ids):
        for j, to in enumerate(stage_ids):
            cnt = int(cross_matrix.iloc[i, j])
            if cnt > 0:
                cross_pairs.append((cnt, fr, to))
    cross_pairs.sort(reverse=True)

    lines.append('Most common between-session transitions:')
    for cnt, fr, to in cross_pairs[:8]:
        direction = 'stay' if fr == to else ('advance' if to > fr else 'regress')
        lines.append(f'  {stage_names[fr]:<20} → {stage_names[to]:<20} {cnt:>4}x  [{direction}]')
    lines.append('')

    # Per-participant sequences
    lines.append('Individual participant trajectories (dominant stage per session):')
    for pid, seq in sorted(participant_sequences.items()):
        if not seq:
            continue
        traj = ' → '.join(s[2] for s in seq)
        lines.append(f'  {pid}: {traj}')
    lines.append('')

    content = '\n'.join(lines)
    path = os.path.join(output_dir, 'state_transition_explanation.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


# -----------------------------------------------------------------------
# 4. Comprehensive longitudinal analysis report
# -----------------------------------------------------------------------

def generate_longitudinal_text_report(
    df: pd.DataFrame,
    participant_reports: list,
    framework: dict,
    output_dir: str,
) -> str:
    """Generate longitudinal_analysis.txt in the output root.

    Covers group trajectory, group transition patterns, and per-participant
    trajectories with key transition quotes.
    Returns path to written file.
    """
    stage_ids = sorted(framework.keys())
    stage_names = {sid: framework[sid].get('short_name', f'Stage {sid}') for sid in stage_ids}

    n_participants = df['participant_id'].nunique()
    session_ids_all = sort_session_ids(df['session_id'].unique().tolist())
    n_sessions = len(session_ids_all)
    session_numbers = sorted(df['session_number'].unique().tolist())

    # Group mean proportions by session_number (combined cohorts)
    group_props = {}
    for snum in session_numbers:
        sdf = df[df['session_number'] == snum]
        n = len(sdf)
        group_props[snum] = {
            st: round(int((sdf['final_label'] == st).sum()) / n, 3) if n > 0 else 0.0
            for st in stage_ids
        }

    # Participant trend stats
    n_advancing = sum(1 for r in participant_reports if r.get('progression_trend', 0) > 0.1)
    n_stable = sum(1 for r in participant_reports if abs(r.get('progression_trend', 0)) <= 0.1)
    n_regressing = sum(1 for r in participant_reports if r.get('progression_trend', 0) < -0.1)
    mean_trend = (
        sum(r.get('progression_trend', 0) for r in participant_reports) / len(participant_reports)
        if participant_reports else 0.0
    )

    cross_matrix, participant_sequences = compute_cross_session_transitions(df, framework)

    lines = []
    lines.append('QRA LONGITUDINAL ANALYSIS REPORT')
    lines.append('=' * 60)
    lines.append(f'Generated:    {date.today().isoformat()}')
    lines.append(f'Participants: {n_participants}  |  Sessions: {n_sessions}')
    lines.append('')

    # ── Group trajectory ──
    lines.append('GROUP TRAJECTORY')
    lines.append('-' * 40)
    trend_dir = 'advancing' if mean_trend > 0.02 else ('regressing' if mean_trend < -0.02 else 'stable')
    lines.append(f'{n_advancing} advancing / {n_stable} stable / {n_regressing} regressing participants')
    lines.append(f'Mean group progression trend: {mean_trend:+.4f}/session ({trend_dir})')
    lines.append('[See figure: group_longitudinal_trajectory.png]')
    lines.append('')
    lines.append('Stage proportions by session (combined cohorts):')

    col_w = 14
    header = f'  {"Session":>9}  ' + ''.join(f'{stage_names[st][:col_w-1]:<{col_w}}' for st in stage_ids)
    lines.append(header)
    lines.append('  ' + '-' * (len(header) - 2))
    for snum in session_numbers:
        props = group_props[snum]
        row = f'  {str(snum):>9}  ' + ''.join(
            f'{_pct(props[st]):>{col_w-1}} ' for st in stage_ids
        )
        lines.append(row)
    lines.append('')

    # ── Group transitions ──
    lines.append('GROUP TRANSITION PATTERNS (between sessions)')
    lines.append('-' * 40)
    lines.append('[See figure: reports/analysis/figures/cross_session_transition_heatmap.png]')
    lines.append('')

    cross_pairs = []
    for i, fr in enumerate(stage_ids):
        for j, to in enumerate(stage_ids):
            cnt = int(cross_matrix.iloc[i, j])
            if cnt > 0:
                cross_pairs.append((cnt, fr, to))
    cross_pairs.sort(reverse=True)

    lines.append('Most frequent between-session transitions:')
    for cnt, fr, to in cross_pairs[:8]:
        direction = 'stay' if fr == to else ('advance' if to > fr else 'regress')
        lines.append(f'  {stage_names[fr]:<20} → {stage_names[to]:<20} {cnt:>3}x  [{direction}]')
    lines.append('')

    # ── Per-participant trajectories ──
    lines.append('PER-PARTICIPANT TRAJECTORIES')
    lines.append('-' * 40)
    lines.append('[See figures: reports/analysis/figures/participant_*_trajectory.png]')
    lines.append('')

    reports_by_pid = {r['participant_id']: r for r in participant_reports if r}

    for pid, seq in sorted(participant_sequences.items()):
        if not seq:
            continue
        report = reports_by_pid.get(pid, {})
        cohort = report.get('cohort_id', '?')
        n_sess = len(seq)
        trend = report.get('progression_trend', 0.0)
        trend_interp = report.get('progression_trend_interpretation', '')

        traj_str = ' → '.join(s[2] for s in seq)
        lines.append(f'{pid}  (Cohort {cohort}, {n_sess} session{"s" if n_sess != 1 else ""}):')
        lines.append(f'  Trajectory: {traj_str}')
        lines.append(f'  Trend: {trend:+.3f}/session — {trend_interp}')

        # Highlight key transitions (stage changes between consecutive sessions)
        key_transitions = [
            (seq[i], seq[i + 1])
            for i in range(len(seq) - 1)
            if seq[i][1] != seq[i + 1][1]
        ]
        if key_transitions:
            lines.append('  Key transitions:')
            for from_entry, to_entry in key_transitions[:3]:
                from_sid, from_stage, from_name = from_entry
                to_sid, to_stage, to_name = to_entry
                direction = 'advance' if to_stage > from_stage else 'regress'
                lines.append(
                    f'    {from_sid} → {to_sid}: {from_name} → {to_name}  [{direction}]'
                )
                # Find a representative quote from the "from" session
                sdf = df[(df['participant_id'] == pid) & (df['session_id'] == from_sid)
                         & (df['final_label'] == from_stage)]
                if not sdf.empty:
                    if 'llm_confidence_primary' in sdf.columns and sdf['llm_confidence_primary'].notna().any():
                        best = sdf.sort_values('llm_confidence_primary', ascending=False).iloc[0]
                    else:
                        best = sdf.iloc[0]
                    quote = str(best.get('text', '')).strip()
                    if quote:
                        lines.append(f'      Quote from {from_sid} ({from_name}):')
                        lines.append(_wrap_quote(quote, indent=8))
        lines.append('')

    content = '\n'.join(lines)
    path = os.path.join(output_dir, 'longitudinal_analysis.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path
