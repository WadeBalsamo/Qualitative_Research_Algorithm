"""Per-stage construct text report generators."""

import os
import re

import pandas as pd

from process import output_paths as _paths
from ._formatting import _bar, _pct, _wrap_quote


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
        lines.append('Prevalence by Participant:')
        sorted_parts = sorted(prev_by_participant.items(), key=lambda kv: -kv[1])
        for pid, v in sorted_parts:#[:10]:
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
                lines.append(f'       Justification: {justification}')
            lines.append('')

    if co_codes:
        lines.append('Co-occurring Codebook Codes (by lift):')
        for c in co_codes[:8]:
            lines.append(f'  - {c["code_id"]:<30} lift={c["lift"]:.2f}  count={c["count"]}')
        lines.append('')

    content = '\n'.join(lines)
    out_dir = _paths.themes_dir(output_dir)
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
