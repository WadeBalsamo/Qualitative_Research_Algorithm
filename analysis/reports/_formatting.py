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
    df: pd.DataFrame, session_id: str, from_end_ms: int, to_start_ms: int
) -> str:
    """Return concatenated therapist text that falls between two participant segments.

    Uses temporal overlap (start_time_ms / end_time_ms) rather than segment_index
    because participant and therapist segment indices live in separate spaces after
    the LLM merge pass resets participant indices to 0, 1, 2, ...

    Requires df to include therapist rows (load with speaker_filter=None).
    """
    if 'speaker' not in df.columns or 'start_time_ms' not in df.columns:
        return ''
    if from_end_ms is None or to_start_ms is None:
        return ''
    if to_start_ms <= from_end_ms:
        return ''
    mask = (
        (df['session_id'] == session_id)
        & (df['speaker'] == 'therapist')
        & (df['start_time_ms'] >= from_end_ms)
        & (df['end_time_ms'] <= to_start_ms)
    )
    rows = df[mask].sort_values('start_time_ms')
    if rows.empty:
        return ''
    texts = [str(r['text']).strip() for _, r in rows.iterrows() if str(r.get('text', '')).strip()]
    return '\n'.join(texts)


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
