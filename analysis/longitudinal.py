"""
analysis/longitudinal.py
------------------------
Cross-participant longitudinal summary and group-level statistics.

Produces longitudinal_summary.json covering:
  - Participant × session heatmap data
  - Group progression trajectory
  - Feasibility assessment (confidence tier distribution)
  - Validity indicators (expected VA-MR progression)
"""

import json
import os
from datetime import date

import pandas as pd

from .loader import sort_session_ids


def _classify_feasibility(pct_high_medium: float) -> str:
    """Rate feasibility based on proportion of high+medium confidence segments.

    >= 0.60 → 'high', >= 0.35 → 'moderate', < 0.35 → 'low'
    """
    if pct_high_medium >= 0.60:
        return 'high'
    if pct_high_medium >= 0.35:
        return 'moderate'
    return 'low'


def _is_non_decreasing(values: list) -> bool:
    """Return True if the sequence is non-decreasing (monotone increasing)."""
    return all(b >= a for a, b in zip(values[:-1], values[1:]))


def compute_group_trajectories(df: pd.DataFrame, framework: dict) -> pd.DataFrame:
    """Compute group-level mean stage proportions per session.

    Averages per-participant proportions (equal weighting), so no participant
    dominates due to having more segments.

    Returns DataFrame with columns:
        session_id, session_number, cohort_id,
        stage_0_mean, ..., stage_{N}_mean, n_participants
    """
    stage_ids = sorted(framework.keys())
    part_rows = []

    for (pid, sid), group in df.groupby(['participant_id', 'session_id']):
        n = len(group)
        snum = int(group['session_number'].iloc[0])
        cid = group['cohort_id'].dropna().iloc[0] if group['cohort_id'].notna().any() else None
        row = {'participant_id': pid, 'session_id': sid, 'session_number': snum, 'cohort_id': cid}
        for st in stage_ids:
            row[f'stage_{st}_pct'] = int((group['final_label'] == st).sum()) / n if n > 0 else 0.0
        part_rows.append(row)

    part_df = pd.DataFrame(part_rows)
    if part_df.empty:
        return part_df

    agg = part_df.groupby(['session_id', 'session_number', 'cohort_id']).agg(
        n_participants=('participant_id', 'count'),
        **{f'stage_{st}_mean': (f'stage_{st}_pct', 'mean') for st in stage_ids},
    ).reset_index()

    session_order = sort_session_ids(agg['session_id'].tolist())
    order_map = {sid: i for i, sid in enumerate(session_order)}
    agg['_order'] = agg['session_id'].map(order_map)
    agg = agg.sort_values('_order').drop(columns='_order').reset_index(drop=True)

    for st in stage_ids:
        agg[f'stage_{st}_mean'] = agg[f'stage_{st}_mean'].round(4)

    return agg


def generate_longitudinal_summary(
    df: pd.DataFrame,
    participant_reports: list,
    framework: dict,
    output_dir: str,
) -> dict:
    """Generate the cross-participant longitudinal summary.

    Parameters
    ----------
    df : DataFrame
        Full participant dataset.
    participant_reports : list
        Already-computed participant report dicts (from participant.py).
    framework : dict
        From loader.load_framework().
    output_dir : str
        Root pipeline output directory.

    Returns
    -------
    Summary dict (also written to reports/analysis/longitudinal_summary.json).
    """
    stage_ids = sorted(framework.keys())
    session_ids_all = sort_session_ids(df['session_id'].unique().tolist())
    participant_ids_all = sorted(df['participant_id'].unique().tolist())

    # Cohort breakdown
    cohorts = {}
    if 'cohort_id' in df.columns:
        for cid in sorted(df['cohort_id'].dropna().unique()):
            cdf = df[df['cohort_id'] == cid]
            cohorts[str(int(cid))] = {
                'session_ids': sort_session_ids(cdf['session_id'].unique().tolist()),
                'n_participants': int(cdf['participant_id'].nunique()),
            }

    # ----------------------------------------------------------------
    # Heatmap data: participant × session_id → dominant stage
    # ----------------------------------------------------------------
    dominant_matrix = []
    dominant_names_matrix = []
    for pid in participant_ids_all:
        dom_row = []
        dom_names_row = []
        pdf = df[df['participant_id'] == pid]
        for sid in session_ids_all:
            sdf = pdf[pdf['session_id'] == sid]
            if sdf.empty:
                dom_row.append(None)
                dom_names_row.append(None)
            else:
                dominant = int(sdf['final_label'].mode().iloc[0])
                dom_row.append(dominant)
                dom_names_row.append(framework.get(dominant, {}).get('short_name', f'Stage {dominant}'))
        dominant_matrix.append(dom_row)
        dominant_names_matrix.append(dom_names_row)

    heatmap_data = {
        'participant_ids': participant_ids_all,
        'session_ids': session_ids_all,
        'dominant_stage_matrix': dominant_matrix,
        'dominant_stage_names_matrix': dominant_names_matrix,
    }

    # ----------------------------------------------------------------
    # Group progression from participant reports
    # ----------------------------------------------------------------
    # Collect progression scores indexed by session_number
    progression_by_snum = {}  # session_number → list of scores
    n_advancing = n_stable = n_regressing = 0

    for report in participant_reports:
        for snum_str, score in report.get('progression_score_by_session', {}).items():
            progression_by_snum.setdefault(snum_str, []).append(score)
        trend = report.get('progression_trend', 0.0)
        if trend > 0.1:
            n_advancing += 1
        elif trend < -0.1:
            n_regressing += 1
        else:
            n_stable += 1

    mean_progression_score_by_session = {
        snum: round(sum(scores) / len(scores), 4)
        for snum, scores in progression_by_snum.items()
    }

    mean_trend_overall = 0.0
    if participant_reports:
        trends = [r.get('progression_trend', 0.0) for r in participant_reports]
        mean_trend_overall = round(sum(trends) / len(trends), 4)

    # Per-session group distribution
    group_traj = compute_group_trajectories(df, framework)
    distribution_by_session = {}
    for _, row in group_traj.iterrows():
        sid = str(row['session_id'])
        distribution_by_session[sid] = {
            f'stage_{st}_mean_pct': float(row.get(f'stage_{st}_mean', 0.0))
            for st in stage_ids
        }
        distribution_by_session[sid]['n_participants'] = int(row['n_participants'])

    group_progression = {
        'mean_progression_score_by_session': mean_progression_score_by_session,
        'mean_progression_trend_overall': mean_trend_overall,
        'n_advancing': n_advancing,
        'n_stable': n_stable,
        'n_regressing': n_regressing,
        'distribution_by_session': distribution_by_session,
    }

    # ----------------------------------------------------------------
    # Feasibility assessment
    # ----------------------------------------------------------------
    n_total = len(df)
    tier_counts = df['label_confidence_tier'].value_counts().to_dict()
    n_high = tier_counts.get('high', 0)
    n_medium = tier_counts.get('medium', 0)
    n_low = tier_counts.get('low', 0)
    pct_high = round(n_high / n_total, 4) if n_total > 0 else 0.0
    pct_medium = round(n_medium / n_total, 4) if n_total > 0 else 0.0
    pct_low = round(n_low / n_total, 4) if n_total > 0 else 0.0
    pct_high_medium = round((n_high + n_medium) / n_total, 4) if n_total > 0 else 0.0
    feasibility_rating = _classify_feasibility(pct_high_medium)

    feasibility_narrative = (
        f"Of {n_total} classified participant segments, "
        f"{round(pct_high * 100, 1)}% were high-confidence and "
        f"{round(pct_medium * 100, 1)}% were medium-confidence "
        f"({round(pct_high_medium * 100, 1)}% combined). "
        f"Feasibility rating: {feasibility_rating}."
    )

    feasibility_assessment = {
        'total_segments': n_total,
        'pct_high_confidence': pct_high,
        'pct_medium_confidence': pct_medium,
        'pct_low_confidence': pct_low,
        'high_plus_medium_pct': pct_high_medium,
        'feasibility_rating': feasibility_rating,
        'feasibility_narrative': feasibility_narrative,
    }

    # ----------------------------------------------------------------
    # Validity indicators
    # ----------------------------------------------------------------
    expected_progression = mean_trend_overall > 0

    # Check if mean progression scores are non-decreasing across sessions
    sorted_snums = sorted(mean_progression_score_by_session.keys(), key=lambda x: int(x))
    ordered_scores = [mean_progression_score_by_session[s] for s in sorted_snums]
    stage_ordering_consistent = _is_non_decreasing(ordered_scores) if len(ordered_scores) > 1 else True

    # Within-session variance (mean SD of progression score per session)
    within_session_vars = []
    for snum, scores in progression_by_snum.items():
        if len(scores) > 1:
            mean_s = sum(scores) / len(scores)
            variance = sum((s - mean_s) ** 2 for s in scores) / len(scores)
            within_session_vars.append(variance ** 0.5)
    within_session_variance = round(
        sum(within_session_vars) / len(within_session_vars), 4
    ) if within_session_vars else None

    # Codebook hypothesis support
    has_codebook = (
        'codebook_labels_ensemble' in df.columns and
        df['codebook_labels_ensemble'].apply(lambda x: bool(x)).any()
    )
    codebook_support = 'not_tested' if not has_codebook else 'mixed'

    validity_narrative_parts = []
    if expected_progression:
        validity_narrative_parts.append(
            "Expected VA-MR progression observed: mean group trend is positive "
            f"(slope={mean_trend_overall:+.3f}/session)."
        )
    else:
        validity_narrative_parts.append(
            "Expected VA-MR progression NOT observed: mean group trend is flat or negative "
            f"(slope={mean_trend_overall:+.3f}/session)."
        )
    if stage_ordering_consistent:
        validity_narrative_parts.append("Stage ordering is consistent across sessions.")
    else:
        validity_narrative_parts.append(
            "Stage ordering shows non-monotone pattern — investigate session-level variation."
        )

    validity_indicators = {
        'expected_progression_observed': expected_progression,
        'stage_ordering_consistent': stage_ordering_consistent,
        'within_session_variance': within_session_variance,
        'codebook_hypothesis_support': codebook_support,
        'validity_narrative': ' '.join(validity_narrative_parts),
    }

    # ----------------------------------------------------------------
    # Narrative summary
    # ----------------------------------------------------------------
    narrative = (
        f"Longitudinal analysis across {len(participant_ids_all)} participant(s) and "
        f"{len(session_ids_all)} session(s). "
        f"{n_advancing} participant(s) showed advancing trajectories, "
        f"{n_stable} stable, {n_regressing} regressing. "
        f"Feasibility: {feasibility_rating} ({round(pct_high_medium * 100, 1)}% high+medium confidence). "
        f"{'VA-MR progression supported.' if expected_progression else 'VA-MR progression not confirmed — further analysis needed.'}"
    )

    summary = {
        'generated': date.today().isoformat(),
        'n_participants': len(participant_ids_all),
        'n_sessions_total': len(session_ids_all),
        'session_ids_ordered': session_ids_all,
        'cohorts': cohorts,
        'heatmap_data': heatmap_data,
        'group_progression': group_progression,
        'feasibility_assessment': feasibility_assessment,
        'validity_indicators': validity_indicators,
        'narrative_summary': narrative,
    }

    out_dir = os.path.join(output_dir, 'reports', 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'longitudinal_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
