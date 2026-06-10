"""Comprehensive longitudinal analysis report generator."""

import json
import os
from collections import Counter, defaultdict
from datetime import date

import pandas as pd
import numpy as np

from ..loader import sort_session_ids
from ..stage_progression import compute_cross_session_transitions
from process import output_paths as _paths
from ._formatting import _bar, _pct, _wrap_quote, _PURER_NAME, _PURER_SHORT


def _parse_codebook_labels(cell) -> list:
    """Parse codebook labels from cell (list, JSON string, or comma-separated string)."""
    if isinstance(cell, list):
        return cell
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, str):
        cell = cell.strip()
        if not cell:
            return []
        try:
            result = json.loads(cell)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        return [c.strip() for c in cell.split(',') if c.strip()]
    return []


def _check_codebook_present(df: pd.DataFrame) -> bool:
    """Check if codebook labels are present and non-empty."""
    if 'codebook_labels_ensemble' not in df.columns:
        return False
    return any(df['codebook_labels_ensemble'].apply(lambda x: bool(_parse_codebook_labels(x))))


def _check_purer_present(df: pd.DataFrame) -> bool:
    """Check if PURER labels are present for therapist turns."""
    if 'purer_primary' not in df.columns or 'speaker' not in df.columns:
        return False
    therapist_df = df[df['speaker'] == 'therapist']
    return therapist_df['purer_primary'].notna().any()


def _build_snum_lookup(df: pd.DataFrame) -> dict:
    """Build session_id → session_number mapping."""
    return (
        df[['session_id', 'session_number']]
        .drop_duplicates('session_id')
        .set_index('session_id')['session_number']
        .to_dict()
    )


def _compute_regression_patterns(participant_sequences: dict, snum_lookup: dict, stage_names: dict) -> dict:
    """Compute regression statistics across participant trajectories."""
    regression_at_snum = defaultdict(int)
    regression_types = Counter()
    participants_with_regressions = []

    for pid, seq in participant_sequences.items():
        pid_regressions = []
        for i in range(len(seq) - 1):
            from_stage_id, to_stage_id = seq[i][1], seq[i + 1][1]
            if to_stage_id < from_stage_id:
                snum_to = snum_lookup.get(seq[i + 1][0])
                if snum_to is not None:
                    regression_at_snum[snum_to] += 1
                    regression_types[(seq[i][2], seq[i + 1][2])] += 1
                    pid_regressions.append((seq[i][2], seq[i + 1][2], snum_to))
        if pid_regressions:
            participants_with_regressions.append((pid, pid_regressions))

    return {
        'regression_at_snum': dict(regression_at_snum),
        'regression_types': dict(regression_types),
        'participants_with_regressions': participants_with_regressions,
        'total_regressions': sum(regression_at_snum.values()),
    }


def _compute_purer_per_session(df: pd.DataFrame) -> dict:
    """Compute dominant PURER move per session."""
    if 'purer_primary' not in df.columns or 'speaker' not in df.columns:
        return {}

    therapist_df = df[(df['speaker'] == 'therapist') & df['purer_primary'].notna()]
    if therapist_df.empty:
        return {}

    result = {}
    for snum in therapist_df['session_number'].unique():
        sdf = therapist_df[therapist_df['session_number'] == snum]
        counts = sdf['purer_primary'].value_counts()
        if not counts.empty:
            result[int(snum)] = int(counts.idxmax())

    return result


def _try_load_purer_csvs(output_dir: str) -> tuple:
    """Try to load PURER CSV files. Returns (lift_df or None, profiles_df or None)."""
    lift_df = None
    profiles_df = None

    analysis_dir = _paths.analysis_data_dir(output_dir)

    for fname, var_name in [('purer_vammr_lift.csv', 'lift_df'), ('purer_transition_profiles.csv', 'profiles_df')]:
        path = os.path.join(analysis_dir, fname)
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if not df.empty:
                    if var_name == 'lift_df':
                        lift_df = df
                    else:
                        profiles_df = df
        except Exception:
            pass

    return lift_df, profiles_df


def _compute_codebook_data(df: pd.DataFrame, stage_ids: list) -> dict:
    """Compute codebook prevalence by stage and by session."""
    if 'codebook_labels_ensemble' not in df.columns:
        return {'by_stage': {}, 'by_session': {}}

    by_stage = {}
    for stage_id in stage_ids:
        stage_df = df[df['final_label'] == stage_id]
        if stage_df.empty:
            continue
        codes = Counter()
        for cell in stage_df['codebook_labels_ensemble']:
            for code in _parse_codebook_labels(cell):
                codes[code] += 1
        n = len(stage_df)
        if codes and n > 0:
            by_stage[stage_id] = {
                code: round(count / n, 3)
                for code, count in codes.most_common(8)
            }

    by_session = {}
    for snum in df['session_number'].unique():
        sdf = df[df['session_number'] == snum]
        codes = Counter()
        for cell in sdf['codebook_labels_ensemble']:
            for code in _parse_codebook_labels(cell):
                codes[code] += 1
        n = len(sdf)
        if codes and n > 0:
            by_session[int(snum)] = {
                code: round(count / n, 3)
                for code, count in codes.most_common(10)
            }

    return {'by_stage': by_stage, 'by_session': by_session}


def _find_illustrative_advances(
    df: pd.DataFrame,
    participant_sequences: dict,
    participant_reports: list,
    snum_lookup: dict,
) -> dict:
    """Find one illustrative advance per session number."""
    reports_by_pid = {r['participant_id']: r for r in participant_reports if r}
    advances_by_snum = {}

    for pid, seq in participant_sequences.items():
        for i in range(len(seq) - 1):
            from_stage_id, to_stage_id = seq[i][1], seq[i + 1][1]
            if to_stage_id > from_stage_id:
                snum_to = snum_lookup.get(seq[i + 1][0])
                if snum_to is not None and snum_to not in advances_by_snum:
                    sdf = df[
                        (df['participant_id'] == pid)
                        & (df['session_id'] == seq[i + 1][0])
                        & (df['final_label'] == to_stage_id)
                    ]
                    if not sdf.empty:
                        if 'llm_confidence_primary' in sdf.columns:
                            best = sdf.sort_values('llm_confidence_primary', ascending=False).iloc[0]
                        else:
                            best = sdf.iloc[0]
                        quote = str(best.get('text', '')).strip()
                        if quote:
                            advances_by_snum[snum_to] = {
                                'pid': pid,
                                'from_stage_name': seq[i][2],
                                'to_stage_name': seq[i + 1][2],
                                'quote': quote,
                            }

    return advances_by_snum


def generate_longitudinal_text_report(
    df: pd.DataFrame,
    participant_reports: list,
    framework: dict,
    output_dir: str,
) -> str:
    """Generate comprehensive 01_outcomes/longitudinal.txt report.

    Covers: VAAMR trajectory with emphasis on trend, stage distributions,
    regression patterns, PURER influence (if available), codebook relationships
    (if available), and illustrative journey quotes.
    """
    # Setup
    stage_ids = sorted(framework.keys())
    stage_names = {sid: framework[sid].get('short_name', f'Stage {sid}') for sid in stage_ids}
    n_participants = df['participant_id'].nunique()
    session_ids_all = sort_session_ids(df['session_id'].unique().tolist())
    n_sessions = len(session_ids_all)
    session_numbers = sorted(df['session_number'].unique().tolist())
    snum_lookup = _build_snum_lookup(df)

    # Per-stage counts per session
    session_stage_counts = {}
    for snum in session_numbers:
        sdf = df[df['session_number'] == snum]
        n = len(sdf)
        session_stage_counts[snum] = {st: int((sdf['final_label'] == st).sum()) for st in stage_ids}
        session_stage_counts[snum]['_total'] = n

    # Participant trend stats
    n_advancing = sum(1 for r in participant_reports if r.get('progression_trend', 0) > 0.1)
    n_stable = sum(1 for r in participant_reports if abs(r.get('progression_trend', 0)) <= 0.1)
    n_regressing = sum(1 for r in participant_reports if r.get('progression_trend', 0) < -0.1)
    mean_trend = (
        sum(r.get('progression_trend', 0) for r in participant_reports) / len(participant_reports)
        if participant_reports
        else 0.0
    )

    # Mean progression score per session (group average)
    mean_prog_by_snum = {}
    for snum in session_numbers:
        scores = [
            r['progression_score_by_session'].get(str(snum))
            for r in participant_reports
            if str(snum) in r.get('progression_score_by_session', {})
        ]
        scores = [s for s in scores if s is not None]
        if scores:
            mean_prog_by_snum[snum] = round(sum(scores) / len(scores), 3)

    # Cross-session transitions and regressions
    cross_matrix, participant_sequences = compute_cross_session_transitions(df, framework)
    regression_data = _compute_regression_patterns(participant_sequences, snum_lookup, stage_names)

    # Guard flags
    has_purer = _check_purer_present(df)
    has_codebook = _check_codebook_present(df)

    # Load optional data
    purer_per_snum = _compute_purer_per_session(df) if has_purer else {}
    lift_df, profiles_df = _try_load_purer_csvs(output_dir) if has_purer else (None, None)
    codebook_data = _compute_codebook_data(df, stage_ids) if has_codebook else {'by_stage': {}, 'by_session': {}}

    # Illustrative advances
    advances_by_snum = _find_illustrative_advances(df, participant_sequences, participant_reports, snum_lookup)
    reports_by_pid = {r['participant_id']: r for r in participant_reports if r}

    from .stat_format import m_ref, provenance_header

    lines = []

    # ─────────────────────────────────────────────────────────────────
    # SECTION 1: Header
    # ─────────────────────────────────────────────────────────────────
    lines.append('QRA LONGITUDINAL ANALYSIS REPORT')
    lines.append('=' * 70)
    # Compact provenance block
    for hline in provenance_header(
        ['vaamr_labels', 'occupancy_trend', 'estage'],
        extra=(
            "MeanScore = E[stage] (mixture-weighted mean VAAMR stage, interval-scale SENSITIVITY). "
            "Dominant-stage trend = slope of the per-participant dominant-stage sequence "
            "(ordinal, different estimator from E[stage] OLS slope). "
            "For primary ordinal-safe outcomes see 06_reports/02_outcomes/progression_summary.txt."
        ),
    ):
        lines.append(hline)
    lines.append('')
    lines.append(f'Generated:      {date.today().isoformat()}')
    lines.append(f'Participants:   {n_participants}  |  Sessions: {n_sessions}')
    lines.append(f'Sessions range: {min(session_numbers)}–{max(session_numbers)}')
    lines.append('')

    # ─────────────────────────────────────────────────────────────────
    # SECTION 2: VAAMR Group Trajectory (enhanced with emphasis)
    # ─────────────────────────────────────────────────────────────────
    lines.append('VAAMR GROUP TRAJECTORY')
    lines.append('─' * 70)
    lines.append('MeanScore = E[stage]: mixture-weighted mean VAAMR stage per session (0.0–4.0)  '
                 + m_ref('estage'))
    lines.append('  0.0 = Vigilance dominant   |   2.0 = Attention dominant   |   4.0 = Reappraisal dominant')
    lines.append('')
    lines.append('Dominant-stage trend: slope of dominant stage across sessions (ordinal, per-participant)  '
                 + m_ref('occupancy_trend'))
    lines.append('  Positive trend → dominant stage advancing toward higher mindfulness stages over time')
    lines.append('  NOTE: this is NOT the E[stage] OLS slope; they share direction but differ numerically.')
    lines.append('')

    trend_dir = 'ADVANCING' if mean_trend > 0.02 else ('REGRESSING' if mean_trend < -0.02 else 'STABLE')
    bar_value = (mean_trend + 2.0) / 4.0
    bar_value = max(0.0, min(1.0, bar_value))
    trend_bar = _bar(bar_value, width=40)
    lines.append('━' * 70)
    lines.append(f'  MEAN GROUP DOMINANT-STAGE TREND:  {mean_trend:+.4f}/session  ▶  {trend_dir}')
    lines.append(f'  {trend_bar}  (scale: -2.0 ←——→ +2.0)')
    lines.append('━' * 70)
    lines.append('')
    lines.append(f'  Advancing: {n_advancing}  |  Stable: {n_stable}  |  Regressing: {n_regressing} participants')
    lines.append('')

    # ─────────────────────────────────────────────────────────────────
    # SECTION 3: Stage Proportions and Counts Table
    # ─────────────────────────────────────────────────────────────────
    lines.append('STAGE PROPORTIONS AND COUNTS BY SESSION (combined cohorts)')
    lines.append('─' * 70)
    lines.append('')

    # Build dynamic column widths
    col_width = 13
    header_parts = ['Session']
    for st in stage_ids:
        sn = stage_names[st]
        header_parts.append(f'{sn[:col_width-1]:<{col_width}}')
    header_parts.append('MeanScore')
    header = '  '.join(header_parts)
    lines.append(header)

    subheader_parts = ['   ']
    for st in stage_ids:
        subheader_parts.append(f'{"N":<5} {"%":<7}')
    subheader_parts.append('')
    lines.append('─'.join(['─' * 9] + ['─' * 12] * len(stage_ids) + ['─' * 9]))

    for snum in session_numbers:
        counts = session_stage_counts[snum]
        total = counts['_total']
        row_parts = [f'{snum:>7}']
        for st in stage_ids:
            n_st = counts[st]
            pct = 100.0 * n_st / total if total > 0 else 0.0
            row_parts.append(f'{n_st:>4} {pct:>5.1f}%')
        mean_score = mean_prog_by_snum.get(snum, None)
        if mean_score is not None:
            row_parts.append(f'{mean_score:>8.2f}')
        else:
            row_parts.append('   —   ')
        lines.append('  '.join(row_parts))

    lines.append('')
    lines.append('[See figure: 05_figures/group_longitudinal_trajectory.png]')
    lines.append('')

    # ─────────────────────────────────────────────────────────────────
    # SECTION 4: Regression Pattern Analysis
    # ─────────────────────────────────────────────────────────────────
    lines.append('REGRESSION PATTERN ANALYSIS')
    lines.append('─' * 70)

    if regression_data['total_regressions'] == 0:
        lines.append('No between-session regressions observed across the group.')
    else:
        lines.append(f'Total between-session dominant-stage regressions: {regression_data["total_regressions"]}')
        lines.append('')

        if regression_data['regression_types']:
            lines.append('Most common regression types:')
            for (from_name, to_name), count in sorted(
                regression_data['regression_types'].items(), key=lambda x: -x[1]
            )[:8]:
                lines.append(f'  {from_name:<20} → {to_name:<20} {count:>3}x')
            lines.append('')

        if regression_data['regression_at_snum']:
            lines.append('Regressions by session (at which session did regression occur):')
            for snum in sorted(regression_data['regression_at_snum'].keys()):
                count = regression_data['regression_at_snum'][snum]
                lines.append(f'  Session {snum:>2}: {count} participant(s) regressed')
            lines.append('')

        if regression_data['participants_with_regressions']:
            lines.append('Participants with regressions:')
            for pid, pid_regressions in regression_data['participants_with_regressions']:
                regressions_str = ', '.join(
                    f'Session {snum} ({from_name}→{to_name})'
                    for from_name, to_name, snum in pid_regressions
                )
                lines.append(f'  {pid}: {regressions_str}')

    lines.append('')

    # ─────────────────────────────────────────────────────────────────
    # SECTION 5: Per-Participant Trajectories (summary table)
    # ─────────────────────────────────────────────────────────────────
    lines.append('PER-PARTICIPANT TRAJECTORY SUMMARY')
    lines.append('─' * 70)
    lines.append('[See figures: 05_figures/participant_*_trajectory.png and 05_per_participant/]')
    lines.append('')
    lines.append('  Participant      Cohort  Sessions  Dominant-stage trend  Direction')
    lines.append('  ─────────────    ──────  ────────  ──────────────────── ─────────')

    for pid, seq in sorted(participant_sequences.items()):
        if not seq:
            continue
        report = reports_by_pid.get(pid, {})
        cohort = report.get('cohort_id', '?')
        n_sess = len(seq)
        trend = report.get('progression_trend', 0.0)
        trend_dir_p = ('advancing' if trend > 0.1
                       else ('regressing' if trend < -0.1 else 'stable'))
        trend_s = f'{trend:+.3f}/session'
        lines.append(f'  {pid:<16} {str(cohort):<6}  {n_sess:>7}   {trend_s:<20} {trend_dir_p}')

    lines.append('')

    # ─────────────────────────────────────────────────────────────────
    # SECTION 6: PURER × VAAMR Longitudinal Influence
    # ─────────────────────────────────────────────────────────────────
    if has_purer:
        lines.append('PURER × VAAMR LONGITUDINAL INFLUENCE')
        lines.append('─' * 70)
        lines.append('How therapist intervention type relates to participant VAAMR progress.')
        lines.append('PURER constructs: P=Phenomenology  U=Utilization  R=Reframing  E=Education  R2=Reinforcement')
        lines.append('')

        lines.append('Session  Dominant PURER          Mean VAAMR Score   Change')
        lines.append('───────  ──────────────────────  ────────────────   ──────')

        for snum in session_numbers:
            purer_id = purer_per_snum.get(snum)
            purer_name = _PURER_NAME.get(purer_id, '?') if purer_id is not None else '—'
            mean_score = mean_prog_by_snum.get(snum)
            score_str = f'{mean_score:.2f}' if mean_score is not None else '—'
            prev_score = mean_prog_by_snum.get(snum - 1)
            if prev_score is not None and mean_score is not None:
                change = mean_score - prev_score
                change_str = f'{change:+.2f}' if abs(change) > 0.01 else '±0.00'
            else:
                change_str = '—'
            lines.append(f'{snum:>7}  {purer_name:<22}  {score_str:>14}   {change_str:>6}')

        lines.append('')

        # Session advancement analysis
        advancing_snums = [
            snum
            for snum in session_numbers
            if snum > session_numbers[0]
            and mean_prog_by_snum.get(snum, 0) > mean_prog_by_snum.get(snum - 1, 0)
        ]
        if advancing_snums:
            advancing_purer = [_PURER_NAME.get(purer_per_snum.get(snum), '?') for snum in advancing_snums]
            lines.append(f'Sessions where group advanced: dominant PURER moves were {advancing_purer}')
        lines.append('')

        # Lift table (if available)
        if lift_df is not None:
            lines.append('PURER → VAAMR Lift (from purer_vammr_lift.csv):')
            lines.append('(Lift = P(VAAMR stage | dominant PURER move) / P(VAAMR stage))')
            lines.append('(Lift > 1.0 = elevated above base rate; * = lift ≥ 1.5)')
            lines.append('')

            lift_cols = [c for c in lift_df.columns if c.startswith('lift_to_')]
            if lift_cols:
                header = 'Construct                 ' + '  '.join(f'→ Stage {i:<6}' for i in range(len(stage_ids)))
                lines.append(header)
                lines.append('─' * len(header))

                for _, row in lift_df.iterrows():
                    construct_name = row.get('purer_construct', row.get('purer_short', '?'))
                    parts = [f'{construct_name:<25}']
                    has_any_high_lift = False
                    for col in lift_cols:
                        lift_val = row.get(col)
                        if lift_val is not None:
                            try:
                                lift_val = float(lift_val)
                                if lift_val >= 1.5:
                                    has_any_high_lift = True
                                    parts.append(f'{lift_val:>7.2f}*')
                                else:
                                    parts.append(f'{lift_val:>8.2f}')
                            except (ValueError, TypeError):
                                parts.append('   —   ')
                        else:
                            parts.append('   —   ')
                    if has_any_high_lift:
                        lines.append('  '.join(parts))

            lines.append('')
        else:
            lines.append('[PURER lift data not found — run pipeline with purer_analysis enabled]')
            lines.append('')

    # ─────────────────────────────────────────────────────────────────
    # SECTION 7: VAAMR × Phenomenology Relationships
    # ─────────────────────────────────────────────────────────────────
    if has_codebook and codebook_data['by_stage']:
        lines.append('VAAMR × PHENOMENOLOGY RELATIONSHIPS')
        lines.append('─' * 70)
        lines.append('Top phenomenology codes by VAAMR stage (prevalence = count / n_segments in stage):')
        lines.append('')

        for stage_id in stage_ids:
            stage_name = stage_names[stage_id]
            codes_dict = codebook_data['by_stage'].get(stage_id, {})
            if codes_dict:
                lines.append(f'  {stage_name.upper()} (Stage {stage_id}):')
                for code, prevalence in sorted(codes_dict.items(), key=lambda x: -x[1])[:5]:
                    bar = _bar(prevalence, width=30)
                    lines.append(f'    {code:<30} {prevalence:.2f}  {bar}')
                lines.append('')

        # Session trend for top 3 codes
        all_codes = Counter()
        for codes_dict in codebook_data['by_session'].values():
            all_codes.update(codes_dict)
        top_codes = [code for code, _ in all_codes.most_common(3)]

        if top_codes:
            lines.append('Top phenomenology code trends across sessions:')
            lines.append('  Session  ' + '  '.join(f'{code[:15]:<16}' for code in top_codes))
            lines.append('  ───────  ' + '  '.join('─' * 16 for _ in top_codes))
            for snum in session_numbers:
                parts = [f'{snum:>7}  ']
                for code in top_codes:
                    prev = codebook_data['by_session'].get(snum, {}).get(code, 0.0)
                    parts.append(f'{prev:.3f}         ')
                lines.append('  '.join(parts))
            lines.append('')

    # ─────────────────────────────────────────────────────────────────
    # SECTION 8: Illustrative Journey Quotes
    # ─────────────────────────────────────────────────────────────────
    lines.append('ILLUSTRATIVE JOURNEY QUOTES')
    lines.append('─' * 70)
    lines.append('One representative participant advance per session, selected by LLM confidence.')
    lines.append('')

    if not advances_by_snum:
        lines.append('[No between-session advances found to illustrate.]')
    else:
        for snum_to in sorted(advances_by_snum.keys()):
            adv = advances_by_snum[snum_to]
            snum_from = snum_to - 1
            lines.append(f'Session {snum_from}→{snum_to}  [{adv["pid"]}: {adv["from_stage_name"]} → {adv["to_stage_name"]}]')
            lines.append(_wrap_quote(adv['quote'], indent=2))
            lines.append('')

    # ─────────────────────────────────────────────────────────────────
    # Write to file
    # ─────────────────────────────────────────────────────────────────
    content = '\n'.join(lines)
    os.makedirs(_paths.reports_outcomes_dir(output_dir), exist_ok=True)
    path = os.path.join(_paths.reports_outcomes_dir(output_dir), 'longitudinal.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path
