"""
cross_validation.py
-------------------
Cross-validation between theme labels and codebook labels.

Computes co-occurrence statistics to empirically validate (or refute)
the hypothesized mapping between themes and codebook codes stored in
ThemeFramework.codebook_hypothesis.
"""

from typing import Dict

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
        Used to map theme IDs to keys and retrieve codebook_hypothesis.
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


def validate_codebook_hypothesis(
    cooccurrence: Dict[str, Dict],
    framework: ThemeFramework,
    min_lift: float = 1.5,
) -> Dict:
    """
    Check the hypothesized theme-to-codebook mapping against observed data.

    For each theme, checks whether the hypothesized codes have lift >= min_lift
    in the observed co-occurrence data.

    Parameters
    ----------
    cooccurrence : dict
        Output of compute_theme_codebook_cooccurrence().
    framework : ThemeFramework
        Must have codebook_hypothesis populated.
    min_lift : float
        Minimum lift to consider a hypothesized association confirmed.

    Returns
    -------
    dict
        Per-theme summary with confirmed/unconfirmed/unexpected associations.
    """
    if not framework.codebook_hypothesis:
        return {}

    results = {}

    for theme_key, hypothesized_codes in framework.codebook_hypothesis.items():
        theme_cooc = cooccurrence.get(theme_key, {})

        confirmed = []
        unconfirmed = []
        for code in hypothesized_codes:
            stats = theme_cooc.get(code)
            if stats and stats['lift'] >= min_lift:
                confirmed.append({'code': code, **stats})
            else:
                unconfirmed.append({
                    'code': code,
                    'lift': stats['lift'] if stats else 0,
                    'count': stats['count'] if stats else 0,
                })

        # Find unexpected strong associations (not in hypothesis)
        unexpected = []
        for code, stats in theme_cooc.items():
            if code not in hypothesized_codes and stats['lift'] >= min_lift:
                unexpected.append({'code': code, **stats})

        unexpected.sort(key=lambda x: -x['lift'])

        results[theme_key] = {
            'confirmed': confirmed,
            'unconfirmed': unconfirmed,
            'unexpected': unexpected[:10],
            'confirmation_rate': (
                len(confirmed) / len(hypothesized_codes)
                if hypothesized_codes else 0
            ),
        }

    return results
