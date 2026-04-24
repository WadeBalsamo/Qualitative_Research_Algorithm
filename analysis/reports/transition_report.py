"""State transition explanation report generator."""

import os
from datetime import date

import pandas as pd

from ..stage_progression import compute_state_transition_matrix, compute_cross_session_transitions
from process import output_paths as _paths
from ._common import _bar, _pct, _wrap_quote, _collect_therapist_cue, _summarize_cue
from ._transition_helpers import (
    _find_transition_examples_by_cohort_session,
    _find_cross_transition_examples_by_cohort_session,
)


def generate_transition_explanation(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
    therapist_cue_config=None,
    llm_client=None,
) -> str:
    """Generate state_transition_explanation.txt in the output root.

    Explains both within-session and between-session transition heatmaps,
    with example quotes for the most common transitions.
    Returns path to written file.
    """
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

    # Example quotes for top non-self transitions — one per (cohort, session)
    non_self = [(cnt, fr, to) for cnt, fr, to in pairs if fr != to]
    if non_self:
        lines.append('Exemplar quotes by cohort and session (within-session transitions):')
        for cnt, fr, to in non_self[:5]:
            examples = _find_transition_examples_by_cohort_session(df, fr, to)
            if not examples:
                continue
            direction = 'forward' if to > fr else 'backward'
            lines.append(
                f'\n  ── {stage_names[fr]} → {stage_names[to]}  ({cnt}x, [{direction}]) ──'
            )
            for ex in examples:
                lines.append(
                    f'  [{ex["participant_id"]}  {ex["session_id"]}  '
                    f'seg{ex["from_seg_idx"]:04d}→seg{ex["to_seg_idx"]:04d}]'
                )
                lines.append(
                    f'    FROM: [{stage_names[fr]}={ex["from_conf"]:.2f}] '
                    + _wrap_quote(ex['from_text'].strip(), indent=12).lstrip()
                )
                _show_cue = (
                    therapist_cue_config is not None
                    and getattr(therapist_cue_config, 'enabled', False)
                )
                if _show_cue:
                    _cue_raw = _collect_therapist_cue(
                        df, ex['session_id'], ex['from_seg_idx'], ex['to_seg_idx']
                    )
                    if not _cue_raw:
                        lines.append('     CUE: [none]')
                    else:
                        _cue_text, _was_summarized = _summarize_cue(
                            _cue_raw, llm_client,
                            therapist_cue_config.max_length_per_cue,
                        )
                        _cue_words = len(_cue_text.split())
                        _marker = ', summarized' if _was_summarized else ''
                        lines.append(
                            f'     CUE: [therapist, {_cue_words} words{_marker}] '
                            + _wrap_quote(_cue_text.strip(), indent=12).lstrip()
                        )
                lines.append(
                    f'      TO: [{stage_names[to]}={ex["to_conf"]:.2f}] '
                    + _wrap_quote(ex['to_text'].strip(), indent=12).lstrip()
                )
        lines.append('')

    # ── Between-session ──
    lines.append('')
    lines.append('BETWEEN-SESSION TRANSITIONS  (cross_session_transition_heatmap.png)')
    lines.append('-' * 60)
    lines.append(
        'Each cell [row → col] counts how many times a participant\'s dominant\n'
        'theme in one session was ROW and their dominant theme in their next coded\n'
        'session was COL. Pairs where a session was skipped (no coded segments for\n'
        'that participant) are included but flagged in the exemplar quotes below.\n'
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

    # Exemplar quotes for top between-session transitions — one per (cohort, from_session)
    non_self_cross = [(cnt, fr, to) for cnt, fr, to in cross_pairs if fr != to]
    if non_self_cross:
        lines.append('Exemplar quotes by cohort and session (between-session transitions):')
        for cnt, fr, to in non_self_cross[:5]:
            examples = _find_cross_transition_examples_by_cohort_session(
                df, fr, to, participant_sequences
            )
            if not examples:
                continue
            direction = 'advance' if to > fr else 'regress'
            lines.append(
                f'\n  ── {stage_names[fr]} → {stage_names[to]}  ({cnt}x, [{direction}]) ──'
            )
            for ex in examples:
                gap = ex.get('session_gap', 0)
                if gap > 0:
                    skipped = list(range(ex['from_snum'] + 1, ex['to_snum']))
                    skip_note = (
                        f'  ← session{"s" if len(skipped) > 1 else ""} '
                        + ', '.join(str(s) for s in skipped)
                        + ' not coded for this participant'
                    )
                else:
                    skip_note = ''
                lines.append(
                    f'  [{ex["participant_id"]}  '
                    f'{ex["from_session_id"]} → {ex["to_session_id"]}{skip_note}]'
                )
                lines.append(
                    f'    FROM: [{ex["from_session_id"]}_seg{ex["from_seg_idx"]:04d}  '
                    f'{stage_names[fr]}={ex["from_conf"]:.2f}] '
                    + _wrap_quote(ex['from_text'].strip(), indent=12).lstrip()
                )
                if ex.get('to_text'):
                    lines.append(
                        f'      TO: [{ex["to_session_id"]}_seg{ex["to_seg_idx"]:04d}  '
                        f'{stage_names[to]}={ex["to_conf"]:.2f}] '
                        + _wrap_quote(ex['to_text'].strip(), indent=12).lstrip()
                    )
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
    os.makedirs(_paths.human_reports_dir(output_dir), exist_ok=True)
    path = os.path.join(_paths.human_reports_dir(output_dir), 'state_transition_explanation.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path
