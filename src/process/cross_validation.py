"""
cross_validation.py
-------------------
Cross-validation between theme labels and codebook labels.

Summarizes observed theme-to-code co-occurrence patterns to characterize
empirical associations between VA-MR stages and phenomenology codes.
"""

import json
import os
from typing import Dict, List, Tuple

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


def compute_soft_theme_codebook_cooccurrence(
    segments_df: pd.DataFrame,
    framework: dict,
    mixture_column: str = 'mixture',
    codebook_label_column: str = 'codebook_labels_ensemble',
    n_stages: int = 5,
    min_soft_lift: float = 1.5,
    max_hard_lift_for_boundary: float = 1.2,
) -> Dict:
    """Mixture-weighted (expected-count) VAAMR×codebook lift.

    Unlike the hard version, each segment contributes its *mixture weight* to every
    stage instead of a single argmax stage, so codes expressed at stage boundaries
    (cusp segments) get fractional credit. For stage k and code c:

        soft_rate(k, c) = [Σ_seg p_k(seg)·1[c∈seg]] / [Σ_seg p_k(seg)]
        soft_lift(k, c) = soft_rate(k, c) / base_rate(c)

    A code is flagged ``boundary_expressed`` when it is elevated under the soft
    weighting (soft_lift ≥ ``min_soft_lift``) but not under hard argmax assignment
    (hard_lift < ``max_hard_lift_for_boundary``) — i.e. it lives between stages.

    ``framework`` is the analysis dict {int stage_id → {key, short_name, ...}}.
    Returns {stage_key: {code_id: {...stats..., boundary_expressed}}}.
    """
    import numpy as np

    if mixture_column not in segments_df.columns:
        return {}
    labeled = segments_df[segments_df[codebook_label_column].notna()].copy()
    if len(labeled) == 0:
        return {}

    n_total = len(labeled)
    stage_ids = sorted(int(k) for k in framework.keys())

    # Base rate per code + hard counts per (stage, code) via argmax.
    base_counts: Dict[str, int] = {}
    hard_stage_total = {k: 0 for k in stage_ids}
    hard_code_counts = {k: {} for k in stage_ids}
    # Expected (soft) accumulators.
    soft_mass = {k: 0.0 for k in stage_ids}
    soft_code_counts = {k: {} for k in stage_ids}

    for _, row in labeled.iterrows():
        codes = row[codebook_label_column]
        codes = [c for c in codes if c] if isinstance(codes, list) else []
        vec = np.asarray(row[mixture_column], dtype=np.float64)
        if vec.shape[0] != n_stages:
            continue
        s = vec.sum()
        if s > 0:
            vec = vec / s
        hard_k = int(np.argmax(vec))
        for c in codes:
            base_counts[c] = base_counts.get(c, 0) + 1
        if hard_k in hard_stage_total:
            hard_stage_total[hard_k] += 1
            for c in codes:
                hard_code_counts[hard_k][c] = hard_code_counts[hard_k].get(c, 0) + 1
        for k in stage_ids:
            pk = float(vec[k]) if k < vec.shape[0] else 0.0
            soft_mass[k] += pk
            for c in codes:
                soft_code_counts[k][c] = soft_code_counts[k].get(c, 0.0) + pk

    base_rates = {c: cnt / n_total for c, cnt in base_counts.items()}

    result: Dict[str, Dict] = {}
    for k in stage_ids:
        key = framework.get(k, {}).get('key', str(k))
        mass = soft_mass[k]
        code_stats = {}
        for c, exp in sorted(soft_code_counts[k].items(), key=lambda x: -x[1]):
            base = base_rates.get(c, 0.0)
            soft_rate = (exp / mass) if mass > 0 else 0.0
            soft_lift = (soft_rate / base) if base > 0 else float('inf')
            # Hard lift for the same (stage, code) for boundary comparison.
            hk_total = hard_stage_total[k]
            hard_rate = (hard_code_counts[k].get(c, 0) / hk_total) if hk_total > 0 else 0.0
            hard_lift = (hard_rate / base) if base > 0 else float('inf')
            boundary = bool(
                soft_lift >= min_soft_lift
                and hard_lift < max_hard_lift_for_boundary
                and exp >= 1.0
            )
            code_stats[c] = {
                'expected_count': round(exp, 3),
                'soft_rate': round(soft_rate, 4),
                'base_rate': round(base, 4),
                'soft_lift': round(soft_lift, 2) if soft_lift != float('inf') else None,
                'hard_lift': round(hard_lift, 2) if hard_lift != float('inf') else None,
                'boundary_expressed': boundary,
            }
        result[key] = {
            'stage_mass': round(mass, 3),
            'codes': code_stats,
        }
    return result


def export_soft_cross_validation_results(soft_cooccurrence: dict, run_dir: str) -> str:
    """Write soft (mixture-weighted) CV results JSON to 04_validation/cross_validation/.

    Returns the output path.
    """
    from . import output_paths as _paths

    cv_dir = _paths.cross_validation_dir(run_dir)
    os.makedirs(cv_dir, exist_ok=True)
    out_path = os.path.join(cv_dir, 'soft_cross_validation_results.json')

    # Collect a flat list of boundary-expressed codes for quick downstream surfacing.
    boundary = []
    for stage_key, payload in soft_cooccurrence.items():
        for code_id, stats in payload.get('codes', {}).items():
            if stats.get('boundary_expressed'):
                boundary.append({'stage': stage_key, 'code': code_id, **stats})
    boundary.sort(key=lambda x: -(x.get('soft_lift') or 0))

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'method': 'mixture_weighted_expected_count',
                'soft_cooccurrence': soft_cooccurrence,
                'boundary_expressed_codes': boundary,
            },
            f,
            indent=2,
        )
    return out_path


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


def export_cross_validation_results(
    cooccurrence: dict,
    associations_by_theme: dict,
    params: dict,
    run_dir: str,
) -> Tuple[str, str]:
    """Write CV result JSONs to 04_validation/cross_validation/.

    Returns (cv_output_path, assoc_output_path).
    """
    from . import output_paths as _paths
    from .validation_exports import collect_top_associations

    cv_dir = _paths.cross_validation_dir(run_dir)
    os.makedirs(cv_dir, exist_ok=True)

    cv_output = os.path.join(cv_dir, 'cross_validation_results.json')
    with open(cv_output, 'w') as f:
        json.dump(
            {
                'raw_cooccurrence': cooccurrence,
                'associations_by_theme': associations_by_theme,
                'parameters': params,
            },
            f,
            indent=2,
        )

    assoc_output = os.path.join(cv_dir, 'top_theme_code_associations.json')
    top_assoc = collect_top_associations(associations_by_theme)
    if top_assoc:
        with open(assoc_output, 'w') as f:
            json.dump(top_assoc, f, indent=2)

    return cv_output, assoc_output
