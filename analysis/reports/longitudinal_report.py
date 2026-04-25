"""Comprehensive longitudinal analysis report generator."""

import os
from datetime import date

import pandas as pd

from ..loader import sort_session_ids
from ..stage_progression import compute_cross_session_transitions
from process import output_paths as _paths
from ._formatting import _bar, _pct, _wrap_quote


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

        # Highlight key transitions (stage changes between consecutive coded sessions)
        key_transitions = [
            (seq[i], seq[i + 1])
            for i in range(len(seq) - 1)
            if seq[i][1] != seq[i + 1][1]
        ]
        if key_transitions:
            lines.append('  Key transitions:')
            for from_entry, to_entry in key_transitions[:3]:
                from_sid, from_stage, from_name = from_entry[:3]
                to_sid, to_stage, to_name = to_entry[:3]
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
    os.makedirs(_paths.human_reports_dir(output_dir), exist_ok=True)
    path = os.path.join(_paths.human_reports_dir(output_dir), 'longitudinal_analysis.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path
