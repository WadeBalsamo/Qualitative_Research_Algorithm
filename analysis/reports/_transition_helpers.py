"""Transition-specific helper functions for report generation."""

import pandas as pd


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
    to_conf, from_seg_idx, to_seg_idx.
    """
    seen_keys = set()
    examples = []
    has_conf = 'llm_confidence_primary' in df.columns

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
