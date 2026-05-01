"""
analysis/purer_analysis.py
--------------------------
PURER × VAMMR cue-block influence analysis.

Operates on the full master_segments dataset (all speakers) to:

1. Build cue blocks — groups of consecutive therapist segments that fall
   between two consecutive participant segments, paired with the FROM and TO
   VAMMR stages of those surrounding participant segments.

2. Compute PURER profiles for each cue block — the distribution of PURER
   constructs across all therapist segments in the block.

3. Compute conditional influence — for each (from_stage, to_stage) transition
   type, what is the PURER move distribution across cue blocks of that type?
   Also computes the standard marginal lift P(to_stage | dominant_purer) / P(to_stage).

4. Track empty cue blocks — transitions where no therapist speech occurred
   between the FROM and TO participant segments. These are spontaneous
   (unmediated) transitions and are reported separately from mediated transitions.

5. Generate a plain-text report (report_purer_analysis.txt) and three CSVs:
   - purer_transition_profiles.csv
   - purer_vammr_lift.csv
   - purer_empty_cue_rates.csv

The primary unit of analysis is the cue block, not the individual
therapist-to-participant adjacency pair. This matches the existing cue-response
report structure and avoids the adjacency-definition ambiguity identified in
the PURER specs.

Called from analysis/runner.py when therapist segments with purer_primary
labels are present in the dataset.
"""

import os
import textwrap
from collections import defaultdict
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from process import output_paths as _paths


# ── Constants ──────────────────────────────────────────────────────────────

_PURER_ID_TO_NAME = {
    0: 'Phenomenology',
    1: 'Utilization',
    2: 'Reframing',
    3: 'Education',
    4: 'Reinforcement',
}
_PURER_ID_TO_SHORT = {
    0: 'P',
    1: 'U',
    2: 'R',
    3: 'E',
    4: 'R2',
}
_ALL_PURER_IDS = list(_PURER_ID_TO_NAME)


# ── Data type ──────────────────────────────────────────────────────────────

class CueBlock:
    """
    All therapist segments between two consecutive participant segments.

    from_stage / to_stage are the VAMMR final_label values of the bounding
    participant segments. purer_profile maps construct_id → count within the
    block. dominant_purer is the most frequent construct (None if empty block
    or all labels absent). is_empty is True when no therapist segments exist
    between the two participant segments.
    """
    __slots__ = (
        'session_id', 'cohort_id',
        'from_seg_id', 'to_seg_id',
        'from_stage', 'to_stage',
        'transition_type',          # 'forward' | 'backward' | 'lateral'
        'purer_profile',            # {construct_id: count}
        'dominant_purer',           # int or None
        'is_empty',                 # True when no therapist segs present
        'n_therapist_segs',         # count of therapist turns in block
    )

    def __init__(
        self,
        session_id: str,
        cohort_id: Optional[int],
        from_seg_id: str,
        to_seg_id: str,
        from_stage: int,
        to_stage: int,
        purer_labels: List[Optional[int]],
    ):
        self.session_id = session_id
        self.cohort_id = cohort_id
        self.from_seg_id = from_seg_id
        self.to_seg_id = to_seg_id
        self.from_stage = from_stage
        self.to_stage = to_stage

        if from_stage < to_stage:
            self.transition_type = 'forward'
        elif from_stage > to_stage:
            self.transition_type = 'backward'
        else:
            self.transition_type = 'lateral'

        self.n_therapist_segs = len(purer_labels)
        self.is_empty = (self.n_therapist_segs == 0)

        profile: Dict[int, int] = defaultdict(int)
        for label in purer_labels:
            if label is not None:
                profile[label] += 1
        self.purer_profile = dict(profile)

        if self.purer_profile:
            self.dominant_purer = max(self.purer_profile, key=self.purer_profile.get)
        else:
            self.dominant_purer = None


# ── Core computation ───────────────────────────────────────────────────────

def compute_cue_block_purer_profiles(df_all: pd.DataFrame) -> List[CueBlock]:
    """
    Build CueBlock objects from a full master_segments DataFrame.

    Uses timestamps (start_time_ms / end_time_ms) to identify therapist segments
    that fall between two consecutive participant segments. This is consistent with
    how _collect_therapist_cue works in the formatting module and is robust to
    datasets where participant and therapist segment_index values are in separate
    namespaces (each resetting from 0 per speaker type).

    Parameters
    ----------
    df_all : pd.DataFrame
        Full dataset including therapist segments. Must have columns:
        session_id, speaker, final_label, start_time_ms, end_time_ms, segment_id.
        purer_primary is optional — absent or all-null means cue blocks will be
        built but all purer_profile values will be empty (empty-cue rate analysis
        still works without PURER labels).

    Returns
    -------
    List[CueBlock]
        One CueBlock per consecutive participant-to-participant pair in each
        session that both have non-null final_label values.
    """
    required = {'session_id', 'speaker', 'final_label', 'start_time_ms', 'end_time_ms'}
    missing = required - set(df_all.columns)
    if missing:
        return []

    has_purer = 'purer_primary' in df_all.columns
    cue_blocks: List[CueBlock] = []

    for session_id, session_df in df_all.groupby('session_id'):
        cohort_id = None
        if 'cohort_id' in session_df.columns:
            cids = session_df['cohort_id'].dropna()
            if len(cids):
                try:
                    cohort_id = int(cids.iloc[0])
                except (ValueError, TypeError):
                    cohort_id = None

        # Participant rows with valid final_label, sorted by start time
        participant_mask = (
            (session_df['speaker'] == 'participant')
            & session_df['final_label'].notna()
        )
        participant_rows = (
            session_df[participant_mask]
            .sort_values('start_time_ms')
            .reset_index(drop=True)
        )

        if len(participant_rows) < 2:
            continue

        # Therapist rows in this session for fast lookup
        therapist_rows = session_df[session_df['speaker'] == 'therapist']

        for i in range(len(participant_rows) - 1):
            from_row = participant_rows.iloc[i]
            to_row = participant_rows.iloc[i + 1]

            from_end_ms = int(from_row.get('end_time_ms', 0))
            to_start_ms = int(to_row.get('start_time_ms', 0))
            from_stage = int(from_row['final_label'])
            to_stage = int(to_row['final_label'])

            # Collect therapist segments whose timestamps fall within the window
            if to_start_ms > from_end_ms:
                between_mask = (
                    (therapist_rows['start_time_ms'] < to_start_ms)     # segment starts before window ends
                    & (therapist_rows['end_time_ms'] > from_end_ms)      # segment ends after window starts
                )
                between_rows = therapist_rows[between_mask]
            else:
                between_rows = therapist_rows.iloc[:0]  # empty slice

            if has_purer:
                purer_labels = [
                    (int(v) if pd.notna(v) else None)
                    for v in between_rows['purer_primary'].tolist()
                ]
            else:
                purer_labels = [None] * len(between_rows)

            block = CueBlock(
                session_id=str(session_id),
                cohort_id=cohort_id,
                from_seg_id=str(from_row.get('segment_id', '')),
                to_seg_id=str(to_row.get('segment_id', '')),
                from_stage=from_stage,
                to_stage=to_stage,
                purer_labels=purer_labels,
            )
            cue_blocks.append(block)

    return cue_blocks


def compute_purer_transition_influence(
    cue_blocks: List[CueBlock],
    framework=None,
) -> Dict[str, Any]:
    """
    Compute PURER influence tables from cue blocks.

    Returns a dict with:
        transition_profiles : DataFrame — one row per (from_stage, to_stage,
            purer_construct), showing count and fraction of mediated cue blocks
            where that construct is dominant.
        lift_matrix : DataFrame — 5×5 pivot: PURER construct (rows) × VAMMR
            to_stage (cols), cell = lift = P(to_stage | dominant_purer) / P(to_stage).
        empty_cue_rates : DataFrame — one row per (from_stage, to_stage),
            showing total count, empty count, and empty fraction.
        conditional_table : DataFrame — one row per
            (from_stage, to_stage, dominant_purer), cell = fraction.
        raw_cue_blocks : list[CueBlock]
    """
    if not cue_blocks:
        empty_df = pd.DataFrame()
        return {
            'transition_profiles': empty_df,
            'lift_matrix': empty_df,
            'empty_cue_rates': empty_df,
            'conditional_table': empty_df,
            'raw_cue_blocks': cue_blocks,
        }

    # ── Build empty-cue rate table ─────────────────────────────────────────
    empty_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    total_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for cb in cue_blocks:
        key = (cb.from_stage, cb.to_stage)
        total_counts[key] += 1
        if cb.is_empty:
            empty_counts[key] += 1

    empty_rows = []
    for key in sorted(total_counts):
        fs, ts = key
        total = total_counts[key]
        empty = empty_counts.get(key, 0)
        empty_rows.append({
            'from_stage': fs,
            'to_stage': ts,
            'total_cue_blocks': total,
            'empty_cue_blocks': empty,
            'empty_fraction': round(empty / total, 3) if total else 0.0,
        })
    empty_cue_rates = pd.DataFrame(empty_rows)

    # ── Build conditional table and transition profiles ────────────────────
    # Only mediated (non-empty) blocks where dominant_purer is set
    mediated = [cb for cb in cue_blocks if not cb.is_empty and cb.dominant_purer is not None]

    # conditional_counts[(from_stage, to_stage, dominant_purer)] = count
    conditional_counts: Dict[Tuple[int, int, int], int] = defaultdict(int)
    mediated_totals: Dict[Tuple[int, int], int] = defaultdict(int)

    for cb in mediated:
        key = (cb.from_stage, cb.to_stage)
        mediated_totals[key] += 1
        conditional_counts[(cb.from_stage, cb.to_stage, cb.dominant_purer)] += 1

    cond_rows = []
    for (fs, ts, dp), cnt in sorted(conditional_counts.items()):
        med_total = mediated_totals.get((fs, ts), 0)
        cond_rows.append({
            'from_stage': fs,
            'to_stage': ts,
            'dominant_purer': dp,
            'purer_name': _PURER_ID_TO_NAME.get(dp, str(dp)),
            'purer_short': _PURER_ID_TO_SHORT.get(dp, str(dp)),
            'count': cnt,
            'fraction_of_mediated': round(cnt / med_total, 3) if med_total else 0.0,
            'mediated_total': med_total,
        })
    conditional_table = pd.DataFrame(cond_rows)
    transition_profiles = conditional_table.copy()

    # ── Marginal lift matrix ───────────────────────────────────────────────
    # P(to_stage) base rates across all mediated blocks
    to_stage_counts: Dict[int, int] = defaultdict(int)
    for cb in mediated:
        to_stage_counts[cb.to_stage] += 1
    n_mediated_total = len(mediated)

    # P(to_stage | dominant_purer) — for each purer construct
    purer_to_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    purer_totals: Dict[int, int] = defaultdict(int)
    for cb in mediated:
        purer_to_counts[(cb.dominant_purer, cb.to_stage)] += 1
        purer_totals[cb.dominant_purer] += 1

    # Gather all to_stages seen
    all_to_stages = sorted({cb.to_stage for cb in mediated})
    all_purer_ids = sorted({cb.dominant_purer for cb in mediated if cb.dominant_purer is not None})

    lift_rows = []
    for dp in all_purer_ids:
        row = {
            'purer_construct': _PURER_ID_TO_NAME.get(dp, str(dp)),
            'purer_short': _PURER_ID_TO_SHORT.get(dp, str(dp)),
            'n_blocks': purer_totals[dp],
        }
        for ts in all_to_stages:
            p_to_given_purer = (
                purer_to_counts[(dp, ts)] / purer_totals[dp]
                if purer_totals[dp] else 0.0
            )
            p_to_marginal = (
                to_stage_counts[ts] / n_mediated_total
                if n_mediated_total else 0.0
            )
            lift = (p_to_given_purer / p_to_marginal) if p_to_marginal > 0 else 0.0
            row[f'lift_to_{ts}'] = round(lift, 2)
            row[f'count_to_{ts}'] = purer_to_counts[(dp, ts)]
        lift_rows.append(row)
    lift_matrix = pd.DataFrame(lift_rows)

    return {
        'transition_profiles': transition_profiles,
        'lift_matrix': lift_matrix,
        'empty_cue_rates': empty_cue_rates,
        'conditional_table': conditional_table,
        'raw_cue_blocks': cue_blocks,
    }


# ── Export ─────────────────────────────────────────────────────────────────

def export_purer_csvs(influence: Dict[str, Any], output_dir: str) -> List[str]:
    """Write the three PURER analysis CSVs to 03_analysis_data/."""
    analysis_dir = _paths.analysis_data_dir(output_dir)
    os.makedirs(analysis_dir, exist_ok=True)

    written = []
    for key, filename in [
        ('transition_profiles', 'purer_transition_profiles.csv'),
        ('lift_matrix', 'purer_vammr_lift.csv'),
        ('empty_cue_rates', 'purer_empty_cue_rates.csv'),
    ]:
        df = influence.get(key)
        if df is not None and len(df):
            path = os.path.join(analysis_dir, filename)
            df.to_csv(path, index=False)
            written.append(path)
    return written


# ── Plain-text report ──────────────────────────────────────────────────────

def generate_purer_report(
    influence: Dict[str, Any],
    output_dir: str,
    framework=None,
    df_all: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """
    Write report_purer_analysis.txt to 06_reports/.

    Sections:
    0. Overall PURER construct distribution (corpus-wide)
    1. Summary statistics (cue blocks)
    2. Empty cue rates per transition type
    3. PURER move distribution per transition type (mediated blocks only)
    4. Marginal lift matrix
    5. Theoretical predictions vs observed lift
    6. Per-session PURER profiles
    """
    cue_blocks: List[CueBlock] = influence.get('raw_cue_blocks', [])
    if not cue_blocks:
        return None

    reports_dir = _paths.human_reports_dir(output_dir)
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, 'report_purer_analysis.txt')

    # Build VAMMR stage name lookup from framework if available.
    # framework may be a ThemeFramework object (has .themes) or a dict
    # (from analysis.loader.load_framework, keyed by int stage_id).
    stage_names: Dict[int, str] = {}
    if framework is not None:
        if hasattr(framework, 'themes'):
            for t in framework.themes:
                stage_names[t.theme_id] = t.short_name
        elif isinstance(framework, dict):
            for sid, info in framework.items():
                stage_names[int(sid)] = info.get('short_name', str(sid))

    def stage_label(sid: int) -> str:
        return stage_names.get(sid, str(sid))

    W = 72

    def _hr(char='─'):
        return char * W

    def _wrap(text: str, indent: int = 0) -> str:
        prefix = ' ' * indent
        return '\n'.join(
            textwrap.fill(line, width=W, initial_indent=prefix, subsequent_indent=prefix)
            for line in text.splitlines()
        )

    total_blocks = len(cue_blocks)
    empty_blocks = sum(1 for cb in cue_blocks if cb.is_empty)
    mediated_blocks = total_blocks - empty_blocks
    labeled_blocks = sum(
        1 for cb in cue_blocks if not cb.is_empty and cb.dominant_purer is not None
    )
    purer_available = labeled_blocks > 0

    transition_profiles: pd.DataFrame = influence.get('transition_profiles', pd.DataFrame())
    lift_matrix: pd.DataFrame = influence.get('lift_matrix', pd.DataFrame())
    empty_cue_rates: pd.DataFrame = influence.get('empty_cue_rates', pd.DataFrame())

    # ── Section 0: Overall PURER construct distribution ────────────────────
    # Computed from raw therapist segments (not cue blocks) for a complete corpus view.
    overall_dist_lines: List[str] = []
    _purer_display_order = [3, 0, 4, 2, 1]  # E, P, R2, R, U — typical frequency

    has_corpus_purer = (
        df_all is not None
        and 'purer_primary' in df_all.columns
        and 'speaker' in df_all.columns
        and df_all[(df_all['speaker'] == 'therapist') & df_all['purer_primary'].notna()].shape[0] > 0
    )

    if has_corpus_purer:
        t_all = df_all[
            (df_all['speaker'] == 'therapist') & df_all['purer_primary'].notna()
        ]
        total_labeled = len(t_all)
        purer_counts_all = t_all['purer_primary'].value_counts()
        overall_dist_lines.append(_hr('─'))
        overall_dist_lines.append('SECTION 0: OVERALL PURER CONSTRUCT DISTRIBUTION (corpus-wide)')
        overall_dist_lines.append(_hr('─'))
        overall_dist_lines.append(f'  {total_labeled} therapist turns classified across all sessions.')
        overall_dist_lines.append('')
        for pid in _purer_display_order:
            cnt = int(purer_counts_all.get(pid, 0))
            if cnt == 0:
                continue
            short = _PURER_ID_TO_SHORT.get(pid, str(pid))
            name  = _PURER_ID_TO_NAME.get(pid, str(pid))
            prop  = cnt / total_labeled
            bar_filled = int(prop * 25)
            bar_str = '█' * bar_filled + '░' * (25 - bar_filled)
            overall_dist_lines.append(
                f'  {short:<3} {name:<20} {cnt:>5} turns  ({100*prop:5.1f}%)  {bar_str}'
            )
        overall_dist_lines.append('')

    lines = []
    lines.append('═' * W)
    lines.append('PURER × VAMMR CUE-BLOCK INFLUENCE ANALYSIS')
    lines.append('═' * W)

    if overall_dist_lines:
        lines.extend(overall_dist_lines)
    lines.append(f'Generated : {date.today().isoformat()}')
    lines.append(f'Total cue blocks : {total_blocks}')
    lines.append(f'  Mediated (therapist speech present) : {mediated_blocks}')
    lines.append(f'  Empty (no therapist speech)         : {empty_blocks}  '
                 f'({100 * empty_blocks // total_blocks if total_blocks else 0}%)')
    if purer_available:
        lines.append(f'  PURER-labeled mediated blocks       : {labeled_blocks}')
    else:
        lines.append(
            'NOTE: No PURER labels found in dataset. Re-run pipeline with '
            'run_purer_labeler=True to enable PURER classification.'
        )
    lines.append('')

    # ── Section 1: Empty cue rates ─────────────────────────────────────────
    lines.append(_hr('─'))
    lines.append('SECTION 1: EMPTY CUE RATES BY TRANSITION TYPE')
    lines.append(_hr('─'))
    lines.append(
        'Empty cues are transitions where no therapist speech occurred between '
        'the FROM and TO participant segments. These are spontaneous (unmediated) '
        'transitions — stage movement driven by the participant, peer interaction, '
        'or carry-over from prior therapist input. They are excluded from PURER '
        'influence analysis.'
    )
    lines.append('')
    lines.append(
        f'  {"Transition":<34}  {"Total":>5}  {"Empty":>5}  {"Empty%":>6}'
    )
    lines.append(f'  {"-" * 34}  {"-" * 5}  {"-" * 5}  {"-" * 6}')

    if not empty_cue_rates.empty:
        for _, row in empty_cue_rates.iterrows():
            fs, ts = int(row['from_stage']), int(row['to_stage'])
            direction = '→' if fs < ts else ('←' if fs > ts else '↔')
            trans_label = f'{stage_label(fs)} {direction} {stage_label(ts)}'
            total = int(row['total_cue_blocks'])
            empty = int(row['empty_cue_blocks'])
            pct = f"{100 * row['empty_fraction']:.0f}%"
            lines.append(f'  {trans_label:<34}  {total:>5}  {empty:>5}  {pct:>6}')
    else:
        lines.append('  (no transition data)')
    lines.append('')

    if not purer_available:
        lines.append(
            'PURER labels not yet available. Sections 2–4 will populate once '
            'the pipeline is re-run with PURER classification enabled.'
        )
        lines.append('')
        lines.append('═' * W)
        with open(out_path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines) + '\n')
        return out_path

    # ── Section 2: PURER distribution per transition type ─────────────────
    lines.append(_hr('─'))
    lines.append('SECTION 2: PURER MOVE DISTRIBUTION PER TRANSITION TYPE')
    lines.append(_hr('─'))
    lines.append(
        'For each transition type, shows the fraction of mediated cue blocks '
        'where each PURER construct was the dominant move. Only mediated blocks '
        '(with therapist speech) are included. Empty blocks are excluded and '
        'reported separately in Section 1.'
    )
    lines.append('')

    if not transition_profiles.empty:
        # Group by (from_stage, to_stage) and show PURER breakdown
        trans_groups = transition_profiles.groupby(['from_stage', 'to_stage'])
        for (fs, ts), grp in sorted(trans_groups):
            direction = '→' if fs < ts else ('←' if fs > ts else '↔')
            trans_label = f'{stage_label(fs)} {direction} {stage_label(ts)}'
            med_total = int(grp['mediated_total'].iloc[0]) if 'mediated_total' in grp.columns else '?'
            lines.append(f'  {trans_label}  (n={med_total} mediated cue blocks)')
            for _, row in grp.sort_values('fraction_of_mediated', ascending=False).iterrows():
                pname = row.get('purer_name', '')
                pshort = row.get('purer_short', '')
                frac = float(row.get('fraction_of_mediated', 0))
                cnt = int(row.get('count', 0))
                bar = '█' * int(frac * 20)
                lines.append(f'    {pshort:<2} {pname:<16}  {bar:<20}  {frac:.0%}  (n={cnt})')
            lines.append('')
    else:
        lines.append('  (no transition profile data)')
        lines.append('')

    # ── Section 3: Marginal lift matrix ───────────────────────────────────
    lines.append(_hr('─'))
    lines.append('SECTION 3: MARGINAL LIFT MATRIX  (PURER → VAMMR to_stage)')
    lines.append(_hr('─'))
    lines.append(
        'Lift = P(to_stage | dominant_purer) / P(to_stage). '
        'Lift > 1.0 means the VAMMR to_stage appears more often after that '
        'therapist move than its base rate. Only mediated blocks included. '
        'Does not condition on from_stage — see Section 2 for conditional analysis.'
    )
    lines.append('')

    if not lift_matrix.empty:
        to_stage_cols = [c for c in lift_matrix.columns if c.startswith('lift_to_')]
        stage_ids = [int(c.replace('lift_to_', '')) for c in to_stage_cols]
        # Header
        header = f'  {"PURER construct":<22}  {"N":>4}'
        for sid in stage_ids:
            header += f'  {stage_label(sid):>8}'
        lines.append(header)
        lines.append(f'  {"-" * 22}  {"-" * 4}' + '  ' + '  '.join(['-' * 8] * len(stage_ids)))
        for _, row in lift_matrix.iterrows():
            pname = str(row.get('purer_construct', ''))
            n = int(row.get('n_blocks', 0))
            line = f'  {pname:<22}  {n:>4}'
            for sid in stage_ids:
                lift_val = row.get(f'lift_to_{sid}', 0.0)
                marker = '*' if float(lift_val) >= 1.5 else ' '
                line += f'  {lift_val:>7.2f}{marker}'
            lines.append(line)
        lines.append('  (* = lift ≥ 1.5 — notably elevated)')
    else:
        lines.append('  (no lift data)')
    lines.append('')

    # ── Section 4: Interpretation notes ───────────────────────────────────
    lines.append(_hr('─'))
    lines.append('SECTION 4: THEORETICAL PREDICTIONS VS OBSERVED LIFT')
    lines.append(_hr('─'))
    predictions = [
        ('Phenomenology', 'Metacognition',  'P opens reflexive space that facilitates observing perspective'),
        ('Reframing',     'Reappraisal',    'R provides the alternative interpretive lens that Reappraisal requires'),
        ('Education',     'Vigilance',      'Explaining does not resolve hypervigilance (expected positive lift)'),
        ('Reinforcement', 'Metacognition',  'Consolidation following R2 promotes decentering'),
        ('Reinforcement', 'Reappraisal',    'Consolidation following R2 promotes transformative insight'),
        ('Utilization',   'Metacognition',  'Capacity-invoking promotes observational stance'),
    ]
    lines.append('  PURER move      → VAMMR stage    Theory')
    lines.append(f'  {"-" * 68}')
    for purer_name, vammr_name, rationale in predictions:
        # Try to find the observed lift value
        observed = '?'
        if not lift_matrix.empty:
            purer_row = lift_matrix[lift_matrix['purer_construct'] == purer_name]
            if not purer_row.empty and framework is not None:
                # Resolve vammr_name to stage_id using whichever framework format is available
                matching_stages = []
                if hasattr(framework, 'themes'):
                    matching_stages = [
                        t.theme_id for t in framework.themes
                        if t.short_name == vammr_name or t.name == vammr_name
                    ]
                elif isinstance(framework, dict):
                    matching_stages = [
                        sid for sid, info in framework.items()
                        if info.get('short_name') == vammr_name or info.get('name') == vammr_name
                    ]
                if matching_stages:
                    col = f'lift_to_{matching_stages[0]}'
                    if col in purer_row.columns:
                        val = float(purer_row[col].iloc[0])
                        observed = f'{val:.2f}'
        lines.append(f'  {purer_name:<16} → {vammr_name:<16}  lift={observed:>6}  {rationale}')
    lines.append('')

    lines.append(_hr('═'))
    lines.append(
        'NOTE: Lift is computed across all sessions and participants. The '
        'conditional analysis in Section 2 (from_stage-conditioned) is the '
        'primary clinical output. The marginal lift matrix in Section 3 is a '
        'secondary check — it ignores from_stage and may be confounded by '
        'participant baseline differences across transitions.'
    )
    lines.append('')

    # ── Section 5: Per-session PURER profiles ─────────────────────────────
    if has_corpus_purer and df_all is not None:
        lines.append(_hr('─'))
        lines.append('SECTION 5: PER-SESSION PURER PROFILES')
        lines.append(_hr('─'))
        lines.append(
            'PURER move distribution per session (therapist turns only). '
            'Columns show percentage of classified turns per construct.'
        )
        lines.append('')

        t_all_s = df_all[
            (df_all['speaker'] == 'therapist') & df_all['purer_primary'].notna()
        ].copy()
        if 'session_id' in t_all_s.columns:
            # Sort sessions chronologically
            from analysis.loader import sort_session_ids
            all_sids = sort_session_ids(t_all_s['session_id'].unique().tolist())

            # Header
            construct_cols = [_PURER_ID_TO_SHORT.get(p, str(p)) for p in _purer_display_order]
            header = f'  {"Session":<12}' + ''.join(f'  {c:<5}' for c in construct_cols) + '   N   Dominant'
            lines.append(header)
            lines.append(f'  {"-" * 12}' + ''.join(['  -----'] * len(construct_cols)) + '  ---  --------')

            for sid in all_sids:
                s_segs = t_all_s[t_all_s['session_id'] == sid]
                if s_segs.empty:
                    continue
                n_s = len(s_segs)
                s_counts = s_segs['purer_primary'].value_counts()
                dom_id = int(s_counts.idxmax())
                dom_short = _PURER_ID_TO_SHORT.get(dom_id, str(dom_id))
                row = f'  {str(sid):<12}'
                for pid in _purer_display_order:
                    cnt = int(s_counts.get(pid, 0))
                    pct = int(100 * cnt / n_s) if n_s else 0
                    row += f'  {pct:>3}% '
                row += f'  {n_s:>3}  {dom_short}'
                lines.append(row)
            lines.append('')

    lines.append('═' * W)

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')

    return out_path


# ── Top-level entry point ──────────────────────────────────────────────────

def run_purer_analysis(
    df_all: pd.DataFrame,
    output_dir: str,
    framework=None,
) -> Dict[str, Any]:
    """
    Run the full PURER × VAMMR analysis and write all outputs.

    Parameters
    ----------
    df_all : pd.DataFrame
        Full master_segments dataset (all speakers, all columns).
    output_dir : str
        Pipeline output directory.
    framework : ThemeFramework, optional
        VAMMR framework for stage name lookup in reports.

    Returns
    -------
    dict with keys: cue_blocks, influence, files_written
    """
    cue_blocks = compute_cue_block_purer_profiles(df_all)
    n_empty    = sum(1 for cb in cue_blocks if cb.is_empty)
    n_mediated = sum(1 for cb in cue_blocks if not cb.is_empty)
    n_labeled  = sum(1 for cb in cue_blocks if not cb.is_empty and cb.dominant_purer is not None)

    influence = compute_purer_transition_influence(cue_blocks, framework=framework)
    # Store n_mediated in influence so figures can annotate with it
    influence['n_mediated'] = n_mediated

    files_written = []
    csv_paths = export_purer_csvs(influence, output_dir)
    files_written.extend(csv_paths)

    report_path = generate_purer_report(influence, output_dir, framework=framework, df_all=df_all)
    if report_path:
        files_written.append(report_path)

    return {
        'cue_blocks': cue_blocks,
        'influence': influence,
        'files_written': files_written,
        'n_cue_blocks': len(cue_blocks),
        'n_empty': n_empty,
        'n_mediated': n_mediated,
        'purer_labeled': n_labeled,
    }
