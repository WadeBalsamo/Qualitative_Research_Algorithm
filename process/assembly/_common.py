"""Shared private formatting helpers used across assembly submodules."""

from typing import List


def _ms_to_hms(ms: int) -> str:
    s = ms // 1000
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _fmt_conf(c) -> str:
    return f"{c:.2f}" if isinstance(c, (int, float)) else '?'


def _theme_name_from(stage, id_to_name: dict) -> str:
    if stage is None:
        return '—'
    return id_to_name.get(stage, str(stage))


def _summarize_rationales(justifications: List[str], llm_client) -> str:
    """Ask the primary LLM to produce a ≤50-word summary of multiple rater rationales."""
    combined = ' | '.join(f'R{i+1}: {j}' for i, j in enumerate(justifications) if j)
    prompt = (
        "You are a qualitative research assistant. Given the following rationales "
        "from multiple raters classifying the same therapeutic dialogue segment, "
        "write a concise summary (50 words or fewer) that captures the key reasoning.\n\n"
        f"{combined}\n\n"
        "Summary (50 words max, plain text only, no bullet points):"
    )
    try:
        text, _ = llm_client.request(prompt)
        if text:
            words = text.strip().split()
            return ' '.join(words)
    except Exception:
        pass
    return justifications[0][:300] if justifications else ''
