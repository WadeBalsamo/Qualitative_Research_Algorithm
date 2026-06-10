"""Cross-session overview report generator (_overview.txt).

Compressed from the former ~1,600-line per-session dump into two compact
tables — a session × stage occupancy matrix and a per-session summary table —
plus pointers into 04_per_session/ for detail.
"""

import os
from datetime import date

import pandas as pd

from ..loader import sort_session_ids
from process import output_paths as _paths
from ._formatting import _pct
from .stat_format import provenance_header


def _session_mean_estage(df: pd.DataFrame, session_id: str) -> float:
    """Mean E[stage] for a session's participant segments (mixture-weighted; falls back to label mean)."""
    if df is None or len(df) == 0 or 'session_id' not in df.columns:
        return None
    sdf = df[df['session_id'] == session_id]
    if sdf.empty:
        return None
    if 'progression_coord' in sdf.columns and sdf['progression_coord'].notna().any():
        return float(sdf['progression_coord'].dropna().mean())
    if 'final_label' in sdf.columns and sdf['final_label'].notna().any():
        return float(sdf['final_label'].dropna().astype(float).mean())
    return None


def generate_comprehensive_session_report(
    df: pd.DataFrame,
    framework: dict,
    session_reports: list,
    output_dir: str,
) -> str:
    """Generate 04_per_session/_overview.txt — a compact cross-session view.

    Sections:
      1. Session × stage occupancy matrix (rows = sessions chronological,
         columns = 5 stages, cells = %).
      2. Per-session summary table (session_id, n participants, n segments,
         mean E[stage], dominant stage).
      3. Pointers to 04_per_session/ for detail.

    Returns path to written file.
    """
    stage_ids = sorted(framework.keys())
    stage_names = {sid: framework[sid].get('short_name', f'Stage {sid}') for sid in stage_ids}
    # Short column abbreviations for the matrix header (first 5 chars).
    stage_abbr = {sid: stage_names[sid][:5] for sid in stage_ids}

    sessions_by_id = {r['session_id']: r for r in session_reports if r}
    session_id_order = sort_session_ids(list(sessions_by_id.keys()))

    total_segments = sum(r.get('n_segments', 0) for r in session_reports if r)
    total_participants = df['participant_id'].nunique() if df is not None and 'participant_id' in df.columns else 0

    lines = []
    lines.append('CROSS-SESSION OVERVIEW')
    lines.append('=' * 68)
    lines.append(f'Generated:    {date.today().isoformat()}')
    lines.append(f'Sessions:     {len(session_id_order)}    '
                 f'Participants: {total_participants}    Segments: {total_segments}')
    lines.extend(provenance_header(['vaamr_labels', 'estage']))
    lines.append('')

    # ── 1. Session × stage occupancy matrix ───────────────────────────────────
    lines.append('SESSION × STAGE OCCUPANCY MATRIX  [cells = % of session segments]')
    lines.append('─' * 72)
    hdr = f'{"Session":<10}' + ''.join(f'{stage_abbr[st]:>9}' for st in stage_ids)
    lines.append(hdr)
    lines.append('─' * 72)
    for sid in session_id_order:
        report = sessions_by_id.get(sid, {})
        group_props = report.get('group_stage_proportions', {})
        cells = ''.join(
            f'{_pct(float(group_props.get(str(st), 0.0))):>9}' for st in stage_ids
        )
        lines.append(f'{sid:<10}{cells}')
    lines.append('─' * 72)
    lines.append('  Columns: ' + ' | '.join(f'{stage_abbr[st]}={stage_names[st]}' for st in stage_ids))
    lines.append('')

    # ── 2. Per-session summary table ──────────────────────────────────────────
    lines.append('PER-SESSION SUMMARY')
    lines.append('─' * 72)
    lines.append(
        f'{"Session":<10} {"S#":>3} {"Part":>5} {"Seg":>5}  {"E[stage]":>8}  {"Dominant Stage":<16}'
    )
    lines.append('─' * 72)
    for sid in session_id_order:
        report = sessions_by_id.get(sid, {})
        snum = report.get('session_number', '?')
        n_seg = report.get('n_segments', 0)
        n_part = report.get('n_participants', 0)
        group_props = report.get('group_stage_proportions', {})
        # Dominant stage by group proportion.
        dom_stage = None
        if group_props:
            dom_stage = max(stage_ids, key=lambda st: float(group_props.get(str(st), 0.0)))
            if float(group_props.get(str(dom_stage), 0.0)) <= 0:
                dom_stage = None
        dom_name = stage_names.get(dom_stage, '─') if dom_stage is not None else '─'
        mean_est = _session_mean_estage(df, sid)
        est_s = f'{mean_est:.2f}' if mean_est is not None else '   ─'
        lines.append(
            f'{sid:<10} {str(snum):>3} {n_part:>5} {n_seg:>5}  {est_s:>8}  {dom_name:<16}'
        )
    lines.append('─' * 72)
    lines.append('')

    # ── 3. Pointers ───────────────────────────────────────────────────────────
    lines.append('FOR DETAIL')
    lines.append('─' * 72)
    lines.append('  Per-session drill-down (dashboard, who-was-where, transitions, expressions):')
    lines.append('    04_per_session/session_<id>.txt')
    lines.append('  Per-participant trajectories: 05_per_participant/participant_<id>.txt')
    lines.append('  Outcomes (did adaptive occupancy rise?): 02_outcomes/')
    lines.append('  Mechanism (PURER × transition): 03_mechanism/ and 09_supplementary/cue_response.txt')
    lines.append('')

    content = '\n'.join(lines)
    os.makedirs(_paths.reports_per_session_dir(output_dir), exist_ok=True)
    path = os.path.join(_paths.reports_per_session_dir(output_dir), '_overview.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path
