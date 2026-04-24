"""
cross_validation.py
-------------------
Cross-validation between theme labels and codebook labels.

Summarizes observed theme-to-code co-occurrence patterns to characterize
empirical associations between VA-MR stages and phenomenology codes.
"""

from typing import Dict, List

import pandas as pd

from constructs.theme_schema import ThemeFramework


def compute_theme_codebook_cooccurrence(
    segments_df: pd.DataFrame,
    framework: ThemeFramework,
    codebook_label_column: str = 'codebook_labels_ensemble',
    theme_label_column: str = 'primary_stage',
) -> Dict:
    """
    Compute co-occurrence statistics between theme labels and codebook labels.

    For each theme, counts how often each codebook code co-occurs with that
    theme across all segments, then compares observed rates to what would be
    expected by chance (base rate).

    Parameters
    ----------
    segments_df : pd.DataFrame
        Must contain ``theme_label_column`` (int) and
        ``codebook_label_column`` (list of str or None).
    framework : ThemeFramework
        Used to map theme IDs to keys.
    codebook_label_column : str
        Column containing list of codebook code IDs per segment.
    theme_label_column : str
        Column containing integer theme ID per segment.

    Returns
    -------
    dict
        Structure: {theme_key: {code_id: {count, rate, base_rate, lift}}}
    """
    # Filter to segments with both labels
    labeled = segments_df[
        (segments_df[theme_label_column].notna())
        & (segments_df[codebook_label_column].notna())
    ].copy()

    if len(labeled) == 0:
        return {}

    # Compute base rates for each code across all labeled segments
    all_codes: Dict[str, int] = {}
    for codes in labeled[codebook_label_column]:
        if isinstance(codes, list):
            for code in codes:
                all_codes[code] = all_codes.get(code, 0) + 1

    total_segments = len(labeled)
    base_rates = {code: count / total_segments for code, count in all_codes.items()}

    # Compute per-theme co-occurrence
    cooccurrence: Dict[str, Dict] = {}

    for theme in framework.themes:
        theme_segments = labeled[labeled[theme_label_column] == theme.theme_id]
        n_theme = len(theme_segments)
        if n_theme == 0:
            cooccurrence[theme.key] = {}
            continue

        theme_codes: Dict[str, int] = {}
        for codes in theme_segments[codebook_label_column]:
            if isinstance(codes, list):
                for code in codes:
                    theme_codes[code] = theme_codes.get(code, 0) + 1

        code_stats = {}
        for code, count in sorted(theme_codes.items(), key=lambda x: -x[1]):
            rate = count / n_theme
            base = base_rates.get(code, 0)
            lift = rate / base if base > 0 else float('inf')
            code_stats[code] = {
                'count': count,
                'rate': round(rate, 4),
                'base_rate': round(base, 4),
                'lift': round(lift, 2),
            }

        cooccurrence[theme.key] = code_stats

    return cooccurrence


def summarize_theme_code_associations(
    cooccurrence: Dict,
    min_lift: float = 1.5,
    min_count: int = 3,
    top_n: int = 10,
) -> Dict:
    """
    Summarize observed theme-to-code associations from co-occurrence data.

    For each theme key, identifies codes with meaningful lift above the
    corpus base rate, filtered to suppress sparse codes.

    Parameters
    ----------
    cooccurrence : dict
        Output of compute_theme_codebook_cooccurrence().
    min_lift : float
        Minimum lift to include a code in top_associations.
    min_count : int
        Minimum raw count to include a code (suppresses sparse codes).
    top_n : int
        Maximum number of associations to return per theme.

    Returns
    -------
    dict
        Per-theme summary with keys:
        - top_associations: list of {code, count, rate, base_rate, lift}
          sorted by lift descending
        - n_segments_in_theme: total labeled segments under that theme
        - n_codes_observed: distinct codes appearing at any frequency
    """
    results = {}

    for theme_key, code_stats in cooccurrence.items():
        n_codes_observed = len(code_stats)

        # Derive n_segments_in_theme from count/rate of any entry
        n_segments_in_theme = 0
        for stats in code_stats.values():
            if stats['rate'] > 0:
                n_segments_in_theme = round(stats['count'] / stats['rate'])
                break

        associations: List[Dict] = []
        for code, stats in code_stats.items():
            if stats['lift'] >= min_lift and stats['count'] >= min_count:
                associations.append({'code': code, **stats})

        associations.sort(key=lambda x: -x['lift'])

        results[theme_key] = {
            'top_associations': associations[:top_n],
            'n_segments_in_theme': n_segments_in_theme,
            'n_codes_observed': n_codes_observed,
        }

    return results
