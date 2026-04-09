"""
analysis/participant.py
-----------------------
Per-participant longitudinal analysis reports.

Each report covers one participant's trajectory across all sessions:
stage proportions per session, progression score, trend slope, and
prototypical exemplars per stage.
"""

import json
import os

import pandas as pd

from .exemplars import select_prototypical_exemplars, format_exemplar
from .loader import sort_session_ids


def _compute_progression_score(stage_proportions: dict) -> float:
    """Weighted average stage ID by proportion.

    score = sum(stage_id * proportion)
    Range [0, 3] for a 4-stage framework. Higher = more advanced.
    """
    return sum(int(k) * v for k, v in stage_proportions.items())


def _compute_progression_trend(scores_by_session: dict, session_order: list) -> float:
    """Linear regression slope across ordered sessions.

    Positive slope → advancing; negative → regressing.
    Returns 0.0 if fewer than 2 data points.
    """
    x, y = [], []
    for sid in session_order:
        snum_str = str(sid)
        if snum_str in scores_by_session:
            # Use position index rather than session_number to handle gaps cleanly
            x.append(len(x))
            y.append(scores_by_session[snum_str])

    if len(x) < 2:
        return 0.0

    try:
        from scipy.stats import linregress
        slope, *_ = linregress(x, y)
        return round(float(slope), 4)
    except ImportError:
        pass

    # Manual least-squares fallback
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den = sum((xi - mean_x) ** 2 for xi in x)
    return round(num / den, 4) if den != 0 else 0.0


def _build_narrative_summary(
    participant_id: str,
    n_sessions: int,
    session_ids: list,
    progression_trend: float,
    dominant_stages_by_session: dict,
    framework: dict,
) -> str:
    """Compose a plain-English narrative summary of the participant's trajectory."""
    if not session_ids:
        return f"{participant_id}: no sessions found."

    session_range = f"{session_ids[0]}–{session_ids[-1]}" if len(session_ids) > 1 else session_ids[0]
    dominant_names = []
    for sid in session_ids:
        stage = dominant_stages_by_session.get(str(sid))
        if stage is not None:
            name = framework.get(int(stage), {}).get('short_name', f'Stage {stage}')
            dominant_names.append(name)

    if dominant_names:
        from collections import Counter
        most_common_stage, count = Counter(dominant_names).most_common(1)[0]
        stage_summary = f"dominant stage was {most_common_stage} in {count}/{n_sessions} sessions"
    else:
        stage_summary = "stage data unavailable"

    if progression_trend > 0.1:
        trend_desc = f"positive progression trend (slope={progression_trend:+.2f}/session), suggesting advancement toward later VA-MR stages"
    elif progression_trend < -0.1:
        trend_desc = f"negative progression trend (slope={progression_trend:+.2f}/session), suggesting regression or fluctuation"
    else:
        trend_desc = f"stable progression across sessions (slope={progression_trend:+.2f}/session)"

    return (
        f"{participant_id} participated in {n_sessions} session(s) ({session_range}). "
        f"Their {stage_summary}, with a {trend_desc}."
    )


def generate_participant_report(
    df: pd.DataFrame,
    participant_id: str,
    framework: dict,
    output_dir: str,
) -> dict:
    """Generate a longitudinal report for a single participant.

    Parameters
    ----------
    df : DataFrame
        Full dataset (all participants). Filtered internally.
    participant_id : str
    framework : dict
        From loader.load_framework() — maps int stage_id → {name, short_name, ...}
    output_dir : str
        Root pipeline output directory.

    Returns
    -------
    Report dict (also written to reports/analysis/participants/participant_{id}.json).
    """
    pdf = df[df['participant_id'] == participant_id].copy()
    if pdf.empty:
        return {}

    stage_ids = sorted(framework.keys())
    n_stages = len(stage_ids)

    # Determine session order
    session_ids = sort_session_ids(pdf['session_id'].unique().tolist())
    cohort_id = pdf['cohort_id'].dropna().iloc[0] if pdf['cohort_id'].notna().any() else None

    sessions_detail = {}
    longitudinal_trajectory = {}
    progression_score_by_session = {}
    dominant_stages_by_session = {}

    for sid in session_ids:
        sdf = pdf[pdf['session_id'] == sid]
        n_seg = len(sdf)
        snum = int(sdf['session_number'].iloc[0]) if n_seg > 0 else 0

        # Stage proportions
        stage_counts = sdf['final_label'].value_counts().to_dict()
        stage_props = {}
        for st in stage_ids:
            cnt = stage_counts.get(st, 0)
            stage_props[str(st)] = round(cnt / n_seg, 4) if n_seg > 0 else 0.0

        dominant_stage = int(sdf['final_label'].mode().iloc[0]) if n_seg > 0 else None
        dominant_name = framework.get(dominant_stage, {}).get('short_name', '') if dominant_stage is not None else ''

        mean_conf = round(float(sdf['llm_confidence_primary'].mean()), 4) \
            if sdf['llm_confidence_primary'].notna().any() else None

        prog_score = round(_compute_progression_score(stage_props), 4)

        # Exemplar per stage for this session
        exemplars_by_stage = {}
        for st in stage_ids:
            st_df = sdf[sdf['final_label'] == st]
            ex_rows = select_prototypical_exemplars(st_df, n=1)
            if not ex_rows.empty:
                exemplars_by_stage[str(st)] = format_exemplar(ex_rows.iloc[0])
            else:
                exemplars_by_stage[str(st)] = None

        sessions_detail[sid] = {
            'session_id': sid,
            'session_number': snum,
            'n_segments': n_seg,
            'stage_proportions': stage_props,
            'dominant_stage': dominant_stage,
            'dominant_stage_name': dominant_name,
            'mean_confidence': mean_conf,
            'progression_score': prog_score,
            'exemplars_by_stage': exemplars_by_stage,
        }

        longitudinal_trajectory[str(snum)] = stage_props
        progression_score_by_session[str(snum)] = prog_score
        if dominant_stage is not None:
            dominant_stages_by_session[sid] = dominant_stage

    # Overall trend
    progression_trend = _compute_progression_trend(
        progression_score_by_session,
        [str(sessions_detail[s]['session_number']) for s in session_ids],
    )

    if progression_trend > 0.1:
        trend_interp = "advancing — progression scores increase across sessions"
    elif progression_trend < -0.1:
        trend_interp = "regressing — progression scores decrease across sessions"
    else:
        trend_interp = "stable — no consistent directional trend across sessions"

    # Overall exemplars per stage (best across all sessions)
    stage_exemplars_overall = {}
    for st in stage_ids:
        st_df = pdf[pdf['final_label'] == st]
        ex_rows = select_prototypical_exemplars(st_df, n=1)
        if not ex_rows.empty:
            stage_exemplars_overall[str(st)] = format_exemplar(ex_rows.iloc[0])
        else:
            stage_exemplars_overall[str(st)] = None

    # Human coding agreement (if validation data present)
    human_agreement = None
    if 'in_human_coded_subset' in pdf.columns:
        coded = pdf[pdf['in_human_coded_subset'] == True]
        if len(coded) > 0 and 'human_label' in coded.columns:
            coded = coded[coded['human_label'].notna()]
            if len(coded) > 0:
                agreed = (coded['human_label'].astype(int) == coded['final_label']).sum()
                human_agreement = {
                    'n_coded': len(coded),
                    'n_agreed': int(agreed),
                    'agreement_rate': round(float(agreed) / len(coded), 4),
                }

    narrative = _build_narrative_summary(
        participant_id,
        len(session_ids),
        session_ids,
        progression_trend,
        dominant_stages_by_session,
        framework,
    )

    report = {
        'participant_id': participant_id,
        'cohort_id': int(cohort_id) if cohort_id is not None and not pd.isna(cohort_id) else None,
        'n_sessions': len(session_ids),
        'session_ids': session_ids,
        'sessions': sessions_detail,
        'longitudinal_trajectory': longitudinal_trajectory,
        'progression_score_by_session': progression_score_by_session,
        'progression_score_overall': round(
            sum(progression_score_by_session.values()) / len(progression_score_by_session), 4
        ) if progression_score_by_session else None,
        'progression_trend': progression_trend,
        'progression_trend_interpretation': trend_interp,
        'stage_exemplars_overall': stage_exemplars_overall,
        'human_coding_agreement': human_agreement,
        'narrative_summary': narrative,
    }

    # Write report
    out_dir = os.path.join(output_dir, 'reports', 'analysis', 'participants')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'participant_{participant_id}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def generate_all_participant_reports(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
) -> list:
    """Generate reports for all unique participants in df.

    Returns list of report dicts.
    """
    participant_ids = sorted(df['participant_id'].unique().tolist())
    reports = []
    for pid in participant_ids:
        try:
            report = generate_participant_report(df, pid, framework, output_dir)
            if report:
                reports.append(report)
        except Exception as e:
            print(f"  Warning: participant report failed for {pid}: {e}")
    return reports
