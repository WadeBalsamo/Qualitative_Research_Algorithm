"""Session-level therapist language summaries — generated once, stored for reuse."""

import json
import os
from datetime import date

import pandas as pd

from process import output_paths as _paths
from ._formatting import _summarize_cue


def _collect_session_therapist_text(df: pd.DataFrame, session_id: str) -> str:
    """Concatenate all therapist text for a session in temporal order."""
    if 'speaker' not in df.columns:
        return ''
    mask = (df['session_id'] == session_id) & (df['speaker'] == 'therapist')
    rows = df[mask]
    if rows.empty:
        return ''
    sort_col = 'start_time_ms' if 'start_time_ms' in rows.columns else rows.columns[0]
    rows = rows.sort_values(sort_col)
    return '\n'.join(str(t).strip() for t in rows['text'] if str(t).strip())


def generate_session_summaries(
    df_all: pd.DataFrame,
    session_ids: list,
    output_dir: str,
    llm_client=None,
    max_words: int = 500,
) -> dict:
    """Generate LLM summaries of therapist language per session.

    Each session's therapist text is summarized to max_words (default 500).
    When text is already under max_words, it is used verbatim.
    When llm_client is None, text is used verbatim regardless of length.

    Writes:
      06_reports/session_summaries.json  — {session_id: summary_text}
      06_reports/session_summaries.txt   — human-readable, one block per session

    Returns {session_id: summary_text}.
    """
    summaries = {}

    for sid in session_ids:
        raw = _collect_session_therapist_text(df_all, sid)
        if not raw:
            summaries[sid] = '[no therapist segments]'
        else:
            text, _ = _summarize_cue(raw, llm_client, max_words)
            summaries[sid] = text

    # ── Write JSON ────────────────────────────────────────────────────────────
    json_dir = _paths.analysis_data_dir(output_dir)
    os.makedirs(json_dir, exist_ok=True)

    json_path = os.path.join(json_dir, 'session_summaries.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    # ── Write human-readable txt ──────────────────────────────────────────────
    out_dir = _paths.human_reports_dir(output_dir)
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    lines.append('SESSION THERAPIST SUMMARIES')
    lines.append('=' * 60)
    lines.append(f'Generated: {date.today().isoformat()}')
    lines.append(f'Sessions: {len(session_ids)}   Max words per summary: {max_words}')
    lines.append('')

    for sid in session_ids:
        summary = summaries.get(sid, '[no summary]')
        lines.append(f'── SESSION {sid} ' + '─' * max(1, 48 - len(sid)))
        lines.append(summary)
        lines.append('')

    txt_path = os.path.join(out_dir, 'session_summaries.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return summaries
