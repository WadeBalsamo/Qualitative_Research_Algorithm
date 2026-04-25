"""Shared private formatting helpers used across multiple report modules."""

import pandas as pd


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
    if not from_end_ms or not to_start_ms:
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
    """Summarize therapist cue text to ≤ max_words words. Returns (text, was_summarized)."""
    words = text.split()
    if len(words) <= max_words:
        return text, False
    if llm_client is None:
        return ' '.join(words[:max_words]), False
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
            result_words = result.split()
            if len(result_words) > max_words:
                result = ' '.join(result_words[:max_words])
            return result, True
    except Exception:
        pass
    return ' '.join(words[:max_words]), False


def _summarize_participant_text(text: str, llm_client, max_words: int):
    """Summarize participant psychotherapy disclosure to ≤ max_words words.

    Returns (text, was_summarized). Prompt is tailored to preserve emotional
    themes, presenting concerns, and stage-of-change indicators rather than
    therapeutic technique language.
    """
    words = text.split()
    if len(words) <= max_words:
        return text, False
    if llm_client is None:
        return ' '.join(words[:max_words]), False
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
            result_words = result.split()
            if len(result_words) > max_words:
                result = ' '.join(result_words[:max_words])
            return result, True
    except Exception:
        pass
    return ' '.join(words[:max_words]), False
