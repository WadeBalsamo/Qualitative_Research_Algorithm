"""Shared private formatting helpers used across multiple report modules."""

import pandas as pd


def _bar(value: float, width: int = 30) -> str:
    """ASCII progress bar for a proportion (0.0–1.0)."""
    filled = int(round(value * width))
    return '█' * filled + '░' * (width - filled)


def _pct(value: float) -> str:
    return f'{value * 100:.1f}%'


def _wrap_quote(text: str, indent: int = 9, max_width: int = 80) -> str:
    """Wrap a quoted text to max_width, indenting continuation lines.

    Existing line breaks in the incoming text are preserved, and the function
    never truncates any words or reasoning content.
    """
    if not text:
        return ' ' * indent + '""'

    prefix = ' ' * indent + '"'
    continuation_prefix = ' ' * (indent + 1)
    lines = []

    for raw_line in text.replace('\r\n', '\n').split('\n'):
        if not raw_line.strip():
            lines.append(prefix + raw_line.strip() + '"')
            continue

        current = prefix
        for word in raw_line.split():
            if len(current) + len(word) + 1 > max_width:
                lines.append(current)
                current = continuation_prefix + word
            else:
                if current == prefix:
                    current += word
                else:
                    current += ' ' + word
        current += '"'
        lines.append(current)

    return '\n'.join(lines)


def _collect_therapist_cue(
    df: pd.DataFrame, session_id: str, from_end_ms: int, to_start_ms: int,
    annotate_purer: bool = False,
) -> str:
    """Return concatenated therapist text between two participant segments.

    Collects all therapist segments that overlap temporally with the window
    [from_end_ms, to_start_ms]. This captures the interactive response cues
    that a therapist makes between two participant turns.

    Uses temporal overlap (start_time_ms / end_time_ms) rather than segment_index
    because participant and therapist segment indices can live in separate spaces
    after segmentation pipeline updates.

    Parameters
    ----------
    df : pd.DataFrame
        Must include 'speaker', 'start_time_ms', 'end_time_ms', 'text' columns.
        Should be loaded with speaker_filter=None to include therapist rows.
    session_id : str
        Session identifier to filter rows.
    from_end_ms : int
        End time (ms) of the prior participant segment.
    to_start_ms : int
        Start time (ms) of the next participant segment.
    annotate_purer : bool
        If True and 'purer_primary' column exists, prefix each therapist
        segment with its PURER construct label (e.g. "[E]").

    Returns
    -------
    str
        Newline-joined concatenation of therapist text blocks, or empty string
        if no therapist segments overlap the window.
    """
    # Early validation
    if 'speaker' not in df.columns or 'start_time_ms' not in df.columns:
        return ''
    if from_end_ms is None or to_start_ms is None:
        return ''
    if to_start_ms <= from_end_ms:
        return ''

    # Select therapist segments that OVERLAP with the temporal window.
    # Interval overlap condition: start < window_end AND end > window_start
    mask = (
        (df['session_id'] == session_id)
        & (df['speaker'] == 'therapist')
        & (df['start_time_ms'] < to_start_ms)      # segment starts before window ends
        & (df['end_time_ms'] > from_end_ms)         # segment ends after window starts
    )
    rows = df[mask].sort_values('start_time_ms')

    if rows.empty:
        return ''

    has_purer = annotate_purer and 'purer_primary' in df.columns
    texts = []

    for _, r in rows.iterrows():
        t = str(r.get('text', '')).strip()
        if not t:
            continue
        if has_purer:
            purer_val = r.get('purer_primary')
            try:
                purer_id = int(purer_val) if pd.notna(purer_val) else None
            except (ValueError, TypeError):
                purer_id = None
            if purer_id is not None:
                tag = _PURER_SHORT.get(purer_id, str(purer_id))
                t = f'[{tag}] {t}'
        texts.append(t)

    return '\n'.join(texts)


_PURER_SHORT = {0: 'P', 1: 'U', 2: 'R', 3: 'E', 4: 'R2'}
_PURER_NAME  = {0: 'Phenomenology', 1: 'Utilization', 2: 'Reframing',
                3: 'Education', 4: 'Reinforcement'}


def _collect_cue_block_purer_profile(
    df: pd.DataFrame, session_id: str, from_end_ms: int, to_start_ms: int,
    include_secondary: bool = True,
) -> dict:
    """Return PURER label distribution for therapist segments in a cue block.

    Matches the temporal overlap logic of `_collect_therapist_cue()` to ensure
    consistency. Returns a dict mapping purer_construct_id → count of therapist
    segments with that PURER label in the window.

    Primary labels are counted once per constituent therapist segment. When
    include_secondary is True, the secondary label (if present) is also counted
    once per cue block — because all constituent segments share the same secondary
    via uniform propagation, we add exactly one count for the block's secondary
    construct to avoid inflation.

    Parameters
    ----------
    df : pd.DataFrame
        Must include 'speaker', 'start_time_ms', 'end_time_ms', 'purer_primary'.
    session_id : str
        Session identifier.
    from_end_ms : int
        End time (ms) of prior participant segment.
    to_start_ms : int
        Start time (ms) of next participant segment.
    include_secondary : bool
        If True and 'purer_secondary' column exists, also add one count for the
        block's secondary PURER construct (captures co-occurring moves).

    Returns
    -------
    dict
        Maps purer_construct_id → count. Empty dict if no PURER labels or
        no therapist segments overlap the window.
    """
    # Early validation
    if 'purer_primary' not in df.columns:
        return {}
    if 'speaker' not in df.columns or 'start_time_ms' not in df.columns:
        return {}
    if from_end_ms is None or to_start_ms is None or to_start_ms <= from_end_ms:
        return {}

    # Use identical overlap logic as _collect_therapist_cue()
    mask = (
        (df['session_id'] == session_id)
        & (df['speaker'] == 'therapist')
        & (df['start_time_ms'] < to_start_ms)
        & (df['end_time_ms'] > from_end_ms)
    )
    rows = df[mask]

    profile: dict = {}
    for val in rows['purer_primary'].dropna():
        try:
            k = int(val)
            profile[k] = profile.get(k, 0) + 1
        except (ValueError, TypeError):
            pass

    if include_secondary and 'purer_secondary' in df.columns:
        sec_vals = rows['purer_secondary'].dropna()
        if not sec_vals.empty:
            try:
                k = int(sec_vals.iloc[0])
                profile[k] = profile.get(k, 0) + 1
            except (ValueError, TypeError):
                pass

    return profile


def _format_purer_profile(profile: dict) -> str:
    """Return a compact one-line PURER profile string, e.g. '[E×3, P×1, R2×1]'."""
    if not profile:
        return ''
    parts = [
        f"{_PURER_SHORT.get(k, str(k))}×{v}"
        for k, v in sorted(profile.items(), key=lambda x: -x[1])
    ]
    return '[' + ', '.join(parts) + ']'


def _summarize_cue(text: str, llm_client, max_words: int):
    """Summarize therapist cue text while preserving the full original content.

    Returns (text, was_summarized). This helper avoids silent truncation on
    failures or when LLM access is unavailable.
    """
    if not text:
        return text or '', False
    words = text.split()
    if len(words) <= max_words:
        return text, False
    if llm_client is None:
        return text, False
    prompt = (
        f"Summarize the following therapist dialogue in {max_words} words or fewer, "
        "preserving the key therapeutic moves and intent.\n\n"
        f"{text}\n\n"
        f"Summary ({max_words} words max, plain text only):"
    )
    try:
        result, _ = llm_client.request(prompt)
        if result:
            result = result.strip()
            return result, True
    except Exception:
        pass
    return text, False


def _summarize_participant_text(text: str, llm_client, max_words: int):
    """Summarize participant psychotherapy disclosure while preserving the full original content.

    Returns (text, was_summarized). This helper avoids silent truncation on
    failures or when LLM access is unavailable.
    """
    if not text:
        return text or '', False
    words = text.split()
    if len(words) <= max_words:
        return text, False
    if llm_client is None:
        return text, False
    prompt = (
        f"Summarize the following participant disclosures from a psychotherapy session "
        f"in {max_words} words or fewer. Preserve key emotional themes, presenting "
        f"concerns, stage-of-change indicators, and the participant's own language.\n\n"
        f"{text}\n\n"
        f"Summary ({max_words} words max, plain text only):"
    )
    try:
        result, _ = llm_client.request(prompt)
        if result:
            result = result.strip()
            return result, True
    except Exception:
        pass
    return text, False