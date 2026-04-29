"""State transition explanation report generator."""

import os
from collections import defaultdict
from datetime import date

import pandas as pd

from ..stage_progression import compute_state_transition_matrix, compute_cross_session_transitions
from process import output_paths as _paths
from ._formatting import (
    _bar, _pct, _wrap_quote, _collect_therapist_cue, _summarize_cue,
    _summarize_participant_text,
)


# ── Private helpers (formerly _transition_helpers.py) ─────────────────────────

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


def _find_transition_examples_by_cohort_session(
    df: pd.DataFrame, from_stage: int, to_stage: int,
) -> list:
    """Find one example per (cohort_id, session_id) for from_stage → to_stage.

    Returns list of dicts sorted by (cohort_id, session_id), each with keys:
    cohort_id, session_id, participant_id, from_text, to_text, from_conf,
    to_conf, from_seg_idx, to_seg_idx, from_end_ms, to_start_ms.
    """
    seen_keys = set()
    examples = []
    has_conf = 'llm_confidence_primary' in df.columns
    has_times = 'end_time_ms' in df.columns and 'start_time_ms' in df.columns

    for (pid, sid), group in df.groupby(['participant_id', 'session_id']):
        cohort = group['cohort_id'].dropna().iloc[0] if group['cohort_id'].notna().any() else None
        cid = int(cohort) if cohort is not None and not pd.isna(cohort) else None
        key = (cid, sid)
        if key in seen_keys:
            continue

        group = group.sort_values('segment_index')
        labels = group['final_label'].astype(int).tolist()
        texts = group['text'].tolist()
        confs = (group['llm_confidence_primary'].tolist() if has_conf else [0.0] * len(texts))
        seg_indices = group['segment_index'].tolist()
        end_times = group['end_time_ms'].fillna(0).astype(int).tolist() if has_times else [0] * len(labels)
        start_times = group['start_time_ms'].fillna(0).astype(int).tolist() if has_times else [0] * len(labels)

        for i in range(len(labels) - 1):
            if labels[i] == from_stage and labels[i + 1] == to_stage:
                seen_keys.add(key)
                examples.append({
                    'cohort_id': cid,
                    'session_id': sid,
                    'participant_id': pid,
                    'from_text': texts[i],
                    'to_text': texts[i + 1],
                    'from_conf': float(confs[i] or 0),
                    'to_conf': float(confs[i + 1] or 0),
                    'from_seg_idx': int(seg_indices[i]),
                    'to_seg_idx': int(seg_indices[i + 1]),
                    'from_end_ms': end_times[i],
                    'to_start_ms': start_times[i + 1],
                })
                break

    examples.sort(key=lambda e: (e['cohort_id'] if e['cohort_id'] is not None else 9999, e['session_id']))
    return examples


def _find_cross_transition_examples_by_cohort_session(
    df: pd.DataFrame, from_stage: int, to_stage: int,
    participant_sequences: dict,
) -> list:
    """Find one FROM/TO quote pair per (cohort_id, from_session_id) for a between-session
    transition matching from_stage → to_stage.

    Also reports how many session numbers were skipped between the two sessions
    (session_gap=0 means truly adjacent, gap>0 means skipped sessions exist but
    the participant had no coded segments in those sessions).

    Returns list of dicts sorted by (cohort_id, from_session_id):
    cohort_id, from_session_id, to_session_id, participant_id,
    from_text, from_conf, from_seg_idx, to_text, to_conf, to_seg_idx,
    session_gap, from_snum, to_snum.
    """
    has_conf = 'llm_confidence_primary' in df.columns
    seen_keys = set()
    examples = []

    snum_lookup = (
        df[['session_id', 'session_number']]
        .drop_duplicates('session_id')
        .set_index('session_id')['session_number']
        .to_dict()
    )

    for pid, seq in sorted(participant_sequences.items()):
        for i in range(len(seq) - 1):
            from_sid, from_dom, _ = seq[i]
            to_sid, to_dom, _ = seq[i + 1]
            if from_dom != from_stage or to_dom != to_stage:
                continue

            from_sub = df[(df['participant_id'] == pid) & (df['session_id'] == from_sid)
                          & (df['final_label'] == from_stage)]
            if from_sub.empty:
                continue

            cohort = from_sub['cohort_id'].dropna().iloc[0] if from_sub['cohort_id'].notna().any() else None
            cid = int(cohort) if cohort is not None and not pd.isna(cohort) else None
            key = (cid, from_sid)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            if has_conf and from_sub['llm_confidence_primary'].notna().any():
                from_row = from_sub.sort_values('llm_confidence_primary', ascending=False).iloc[0]
            else:
                from_row = from_sub.iloc[0]

            to_sub = df[(df['participant_id'] == pid) & (df['session_id'] == to_sid)
                        & (df['final_label'] == to_stage)]
            if to_sub.empty:
                to_sub = df[(df['participant_id'] == pid) & (df['session_id'] == to_sid)]
            if has_conf and not to_sub.empty and to_sub['llm_confidence_primary'].notna().any():
                to_row = to_sub.sort_values('llm_confidence_primary', ascending=False).iloc[0]
            elif not to_sub.empty:
                to_row = to_sub.iloc[0]
            else:
                to_row = None

            from_snum = int(snum_lookup.get(from_sid, 0))
            to_snum = int(snum_lookup.get(to_sid, 0))
            gap = max(0, abs(to_snum - from_snum) - 1)

            ex = {
                'cohort_id': cid,
                'from_session_id': from_sid,
                'to_session_id': to_sid,
                'participant_id': pid,
                'from_text': str(from_row.get('text', '')).strip(),
                'from_conf': float(from_row.get('llm_confidence_primary', 0) or 0),
                'from_seg_idx': int(from_row.get('segment_index', 0)),
                'to_text': str(to_row.get('text', '')).strip() if to_row is not None else '',
                'to_conf': float(to_row.get('llm_confidence_primary', 0) or 0) if to_row is not None else 0.0,
                'to_seg_idx': int(to_row.get('segment_index', 0)) if to_row is not None else 0,
                'session_gap': gap,
                'from_snum': from_snum,
                'to_snum': to_snum,
            }
            examples.append(ex)

    examples.sort(key=lambda e: (e['cohort_id'] if e['cohort_id'] is not None else 9999, e['from_session_id']))
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


# ── Public report functions ────────────────────────────────────────────────────

def generate_transition_explanation(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
    therapist_cue_config=None,
    llm_client=None,
    df_all: pd.DataFrame = None,
) -> str:
    """Generate stage_transitions.txt in the output root.

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
    lines.append('WITHIN-SESSION TRANSITIONS  (see 03_figures/stage_transition_heatmap.png)')
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

    lines.append('All within-session transitions:')
    for cnt, fr, to in pairs:
        direction = 'stay' if fr == to else ('forward' if to > fr else 'backward')
        lines.append(f'  {stage_names[fr]:<20} → {stage_names[to]:<20} {cnt:>4}x  [{direction}]')
    lines.append('')

    # Example quotes for all non-self transitions — one per (cohort, session), max 5 per pair
    non_self = [(cnt, fr, to) for cnt, fr, to in pairs if fr != to]
    if non_self:
        lines.append('Exemplar quotes by cohort and session (within-session transitions):')
        for cnt, fr, to in non_self:
            examples = _find_transition_examples_by_cohort_session(df, fr, to)
            if not examples:
                continue
            direction = 'forward' if to > fr else 'backward'
            lines.append(
                f'\n  ── {stage_names[fr]} → {stage_names[to]}  ({cnt}x, [{direction}]) ──'
            )
            for ex in examples[:5]:
                _show_cue = (
                    therapist_cue_config is not None
                    and getattr(therapist_cue_config, 'enabled', False)
                )
                # Collect cue first to determine if we should skip this example
                _cue_raw = None
                if _show_cue:
                    _cue_raw = _collect_therapist_cue(
                        df_all if df_all is not None else df,
                        ex['session_id'],
                        ex.get('from_end_ms', 0),
                        ex.get('to_start_ms', 0),
                    )
                    # Skip examples with no cue when cue collection is enabled
                    if not _cue_raw:
                        continue
                
                lines.append(
                    f'  [{ex["participant_id"]}  {ex["session_id"]}  '
                    f'seg{ex["from_seg_idx"]:04d}→seg{ex["to_seg_idx"]:04d}]'
                )
                lines.append(
                    f'    FROM: [{stage_names[fr]}={ex["from_conf"]:.2f}] '
                    + _wrap_quote(ex['from_text'].strip(), indent=12).lstrip()
                )
                if _show_cue and _cue_raw:
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

    lines.append('All between-session transitions:')
    for cnt, fr, to in cross_pairs:
        direction = 'stay' if fr == to else ('advance' if to > fr else 'regress')
        lines.append(f'  {stage_names[fr]:<20} → {stage_names[to]:<20} {cnt:>4}x  [{direction}]')
    lines.append('')

    # Exemplar quotes for all between-session transitions — one per (cohort, from_session), max 5 per pair
    non_self_cross = [(cnt, fr, to) for cnt, fr, to in cross_pairs if fr != to]
    if non_self_cross:
        lines.append('Exemplar quotes by cohort and session (between-session transitions):')
        for cnt, fr, to in non_self_cross:
            examples = _find_cross_transition_examples_by_cohort_session(
                df, fr, to, participant_sequences
            )
            if not examples:
                continue
            direction = 'advance' if to > fr else 'regress'
            lines.append(
                f'\n  ── {stage_names[fr]} → {stage_names[to]}  ({cnt}x, [{direction}]) ──'
            )
            for ex in examples[:5]:
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
    path = os.path.join(_paths.human_reports_dir(output_dir), 'stage_transitions.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


def generate_therapist_cues_report(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
    therapist_cue_config,
    llm_client,
    df_all: pd.DataFrame = None,
) -> str:
    """Generate report_cue_responses.txt in 02_human_reports/.

    Iterates all within-session transitions, groups by (from_stage, to_stage),
    and produces averaged FROM / CUE / TO blocks for each transition type.
    Returns path to written file.
    """
    stage_ids = sorted(framework.keys())
    stage_names = {sid: framework[sid].get('short_name', f'Stage {sid}') for sid in stage_ids}
    max_agg = therapist_cue_config.max_length_of_average_cue_responses
    _cue_df = df_all if df_all is not None else df

    # Collect (from_text, cue_text, to_text) per (from_stage, to_stage)
    transitions: dict = defaultdict(list)
    total = total_forward = total_backward = 0
    has_times = 'end_time_ms' in df.columns and 'start_time_ms' in df.columns

    for (pid, sid), group in df.groupby(['participant_id', 'session_id']):
        if 'speaker' in group.columns:
            pgroup = group[group['speaker'] != 'therapist']
        else:
            pgroup = group
        pgroup = pgroup[pgroup['final_label'].notna()].sort_values('segment_index')
        if len(pgroup) < 2:
            continue

        try:
            labels = pgroup['final_label'].astype(int).tolist()
        except (ValueError, TypeError):
            continue
        texts = pgroup['text'].tolist()
        end_times = pgroup['end_time_ms'].fillna(0).astype(int).tolist() if has_times else [0] * len(labels)
        start_times = pgroup['start_time_ms'].fillna(0).astype(int).tolist() if has_times else [0] * len(labels)

        for i in range(len(labels) - 1):
            fr, to = labels[i], labels[i + 1]
            if fr == to:
                continue
            cue = _collect_therapist_cue(_cue_df, sid, end_times[i], start_times[i + 1])
            transitions[(fr, to)].append((
                str(texts[i]).strip(),
                cue,
                str(texts[i + 1]).strip(),
            ))
            total += 1
            if to > fr:
                total_forward += 1
            elif to < fr:
                total_backward += 1

    sorted_pairs = sorted(transitions.items(), key=lambda x: -len(x[1]))

    lines = []
    lines.append('THERAPIST CUE ANALYSIS')
    lines.append('=' * 60)
    lines.append(f'Generated: {date.today().isoformat()}')
    lines.append('')
    lines.append(f'Total within-session transitions (excluding self-transitions): {total}')
    lines.append(f'  Forward:  {total_forward}')
    lines.append(f'  Backward: {total_backward}')
    lines.append('')
    lines.append(
        'For each transition type the "average" blocks show a representative\n'
        'synthesis of all observed examples (LLM-summarized when over the word cap).\n'
    )

    for (fr, to), entries in sorted_pairs:
        try:
            n = len(entries)
            direction = 'forward' if to > fr else ('backward' if to < fr else 'lateral/stay')
            fr_name = stage_names.get(fr, str(fr))
            to_name = stage_names.get(to, str(to))
            n_empty_cues = sum(1 for e in entries if not e[1])

            lines.append(f'── {fr_name} → {to_name}  (n={n}, [{direction}])')
            lines.append('─' * 60)
            lines.append(
                f'  n = {n}  |  empty cues (no therapist speech between segments): {n_empty_cues}'
            )
            lines.append('')

            # average CUE (skip empty-cue entries)
            cue_entries = [e for e in entries if e[1]]
            cue_texts = [e[1] for e in cue_entries]
            if cue_texts:
                # Only summarize transitions that have a therapist cue.
                from_texts = [e[0] for e in cue_entries if e[0]]
                to_texts = [e[2] for e in cue_entries if e[2]]

                agg_from = ' || '.join(from_texts)
                agg_from, _ = _summarize_participant_text(agg_from, llm_client, max_agg)
                lines.append(f'  average FROM [{fr_name}]:')
                lines.append(_wrap_quote(agg_from, indent=4))
                lines.append('')

                agg_cue = ' || '.join(cue_texts)
                agg_cue, _ = _summarize_cue(agg_cue, llm_client, max_agg)
                lines.append('  average CUE:')
                lines.append(_wrap_quote(agg_cue, indent=4))
                lines.append('')

                agg_to = ' || '.join(to_texts)
                agg_to, _ = _summarize_participant_text(agg_to, llm_client, max_agg)
                lines.append(f'  average TO [{to_name}]:')
                lines.append(_wrap_quote(agg_to, indent=4))
                lines.append('')
            else:
                lines.append(
                    '  average CUE: [none — all transitions had no therapist speech between segments]'
                )
                lines.append('')

            lines.append('')
        except Exception as _exc:
            lines.append(f'  [error generating this transition block: {_exc}]')
            lines.append('')

    content = '\n'.join(lines)
    os.makedirs(_paths.human_reports_dir(output_dir), exist_ok=True)
    path = os.path.join(_paths.human_reports_dir(output_dir), 'report_cue_responses.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path
