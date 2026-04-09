"""
analysis/exemplars.py
---------------------
Prototypical exemplar selection logic.

Used across all report modules to surface the most representative,
high-confidence segments as qualitative illustrations.
"""

import pandas as pd


def select_prototypical_exemplars(
    df: pd.DataFrame,
    n: int = 3,
    min_words: int = 15,
    max_words: int = 200,
) -> pd.DataFrame:
    """Filter and rank segments for use as prototypical exemplars.

    Selection criteria (OR logic on confidence, AND on word count):
      - label_confidence_tier == 'high'
        OR (llm_run_consistency == 3 AND llm_confidence_primary > 0.7)
    AND word_count >= min_words AND word_count <= max_words

    Sort order: llm_run_consistency DESC, llm_confidence_primary DESC.

    Parameters
    ----------
    df : DataFrame
        Segment data (filtered to relevant subset before calling).
    n : int
        Maximum number of exemplars to return.
    min_words, max_words : int
        Word count bounds to exclude trivially short or overly long segments.

    Returns
    -------
    DataFrame of up to n rows with exemplar-relevant columns.
    """
    if df.empty:
        return df.iloc[0:0]

    # Word count filter
    mask_words = (df['word_count'] >= min_words) & (df['word_count'] <= max_words)

    # Confidence filter (OR logic)
    mask_high_tier = df['label_confidence_tier'] == 'high'
    mask_consistent = (
        (df['llm_run_consistency'].fillna(0) == 3) &
        (df['llm_confidence_primary'].fillna(0) > 0.7)
    )
    mask_conf = mask_high_tier | mask_consistent

    candidates = df[mask_words & mask_conf].copy()

    # Fallback: if no high-confidence candidates, relax to any labeled segment
    if candidates.empty:
        candidates = df[mask_words].copy()
    if candidates.empty:
        candidates = df.copy()

    # Sort by consistency then confidence
    candidates = candidates.sort_values(
        ['llm_run_consistency', 'llm_confidence_primary'],
        ascending=[False, False],
        na_position='last',
    )

    return candidates.head(n)


def format_exemplar(row: pd.Series) -> dict:
    """Convert a DataFrame row into the standard exemplar dict for JSON reports.

    Returns
    -------
    dict with keys:
        segment_id, participant_id, session_id, session_number,
        text (full), confidence, consistency, justification
    """
    return {
        'segment_id': str(row.get('segment_id', '')),
        'participant_id': str(row.get('participant_id', '')),
        'session_id': str(row.get('session_id', '')),
        'session_number': int(row.get('session_number', 0)),
        'text': str(row.get('text', '')),
        'confidence': round(float(row['llm_confidence_primary']), 4)
            if pd.notna(row.get('llm_confidence_primary')) else None,
        'consistency': int(row['llm_run_consistency'])
            if pd.notna(row.get('llm_run_consistency')) else None,
        'justification': str(row.get('llm_justification', '') or ''),
    }
