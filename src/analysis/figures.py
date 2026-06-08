"""
analysis/figures.py
-------------------
Matplotlib figure generation for the analysis module.

All figures are saved as PNGs to {output_dir}/03_figures/.
Uses the Agg backend for headless compatibility.
"""

import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

import re

from .loader import sort_session_ids
from .stage_progression import compute_state_transition_matrix, compute_cross_session_transitions
from process import output_paths as _paths

# VAAMR theme colors from constructs/vaamr.py
_VAAMR_COLORS = {
    0: '#DC267F',  # Vigilance — magenta/pink
    1: '#FE6100',  # Avoidance — orange
    2: "#3AB75D",  # Attention Regulation — green
    3: '#648FFF',  # Metacognition — blue
    4: '#785EF0',  # Reappraisal — purple
}

# Fallback palette for non-VAAMR frameworks
_BASE_COLORS = [
    '#4C72B0',  # blue
    '#DD8452',  # orange
    '#55A868',  # green
    '#C44E52',  # red
    '#8172B3',  # purple
    '#937860',  # brown
    '#DA8BC3',  # pink
    '#8C8C8C',  # gray
]

_MISSING_COLOR = '#E0E0E0'  # light gray for missing cells


def _stage_colors(framework: dict) -> dict:
    """Return {stage_id: hex_color} for consistent coloring.
    
    Uses VAAMR-specific colors for stages 0-4, falls back to generic palette otherwise.
    """
    stage_ids = sorted(framework.keys())
    colors = {}
    for i, sid in enumerate(stage_ids):
        # Use VAAMR color if available, else use fallback palette
        colors[sid] = _VAAMR_COLORS.get(sid, _BASE_COLORS[i % len(_BASE_COLORS)])
    return colors


def _ensure_figures_dir(output_dir: str) -> str:
    out = _paths.figures_dir(output_dir)
    os.makedirs(out, exist_ok=True)
    return out


# -----------------------------------------------------------------------
# Figure 1: Dominant-stage heatmap (participant x session_number)
# -----------------------------------------------------------------------

def plot_dominant_stage_heatmap(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
) -> str:
    """Heatmap of dominant stage per participant across session numbers.

    Combines cohorts by session_number. Missing cells are gray.
    Returns path to saved PNG.
    """
    stage_ids = sorted(framework.keys())
    colors = _stage_colors(framework)
    participant_ids = sorted(df['participant_id'].unique().tolist())
    session_numbers = sorted(df['session_number'].unique().tolist())

    n_participants = len(participant_ids)
    n_sessions = len(session_numbers)

    # Build matrix: rows=participants, cols=session_numbers, values=stage_id or NaN
    matrix = np.full((n_participants, n_sessions), np.nan)
    for i, pid in enumerate(participant_ids):
        pdf = df[df['participant_id'] == pid]
        for j, snum in enumerate(session_numbers):
            sdf = pdf[pdf['session_number'] == snum]
            if len(sdf) > 0:
                matrix[i, j] = int(sdf['final_label'].mode().iloc[0])

    # Build colormap: map each stage_id to a color, plus one for missing
    cmap_colors = [_MISSING_COLOR]  # index 0 = missing
    for sid in stage_ids:
        cmap_colors.append(colors[sid])
    cmap = mcolors.ListedColormap(cmap_colors)

    # Remap matrix: missing=-0.5 (will map to index 0), stage N maps to N+0.5
    display = np.full_like(matrix, -0.5)
    for idx, sid in enumerate(stage_ids):
        display[matrix == sid] = idx + 0.5
    bounds = [-1] + list(range(len(stage_ids) + 1))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig_h = max(6, n_participants * 0.45)
    fig_w = max(8, n_sessions * 0.9 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(display, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')

    ax.set_xticks(range(n_sessions))
    ax.set_xticklabels([str(s) for s in session_numbers], fontsize=9)
    ax.set_yticks(range(n_participants))
    ax.set_yticklabels(participant_ids, fontsize=8)
    ax.set_xlabel('Session Number')
    ax.set_ylabel('Participant')
    ax.set_title('Dominant Stage per Participant Across Sessions')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[sid],
              label=framework[sid].get('short_name', f'Stage {sid}'))
        for sid in stage_ids
    ]
    legend_elements.append(Patch(facecolor=_MISSING_COLOR, label='No data'))
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1),
              fontsize=8, frameon=True)

    fig.tight_layout()
    out_dir = _ensure_figures_dir(output_dir)
    path = os.path.join(out_dir, 'dominant_stage_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# -----------------------------------------------------------------------
# Figure 2: Group longitudinal trajectory (combined cohorts)
# -----------------------------------------------------------------------

def plot_group_longitudinal_trajectory(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
) -> str:
    """Line graph of mean stage proportions across session numbers.

    Combines all cohorts by session_number. Equal-weighted over participants.
    Returns path to saved PNG.
    """
    if df.empty:
        return ''

    stage_ids = sorted(framework.keys())
    colors = _stage_colors(framework)

    # Per-participant proportions per session_number
    part_rows = []
    for (pid, snum), group in df.groupby(['participant_id', 'session_number']):
        n = len(group)
        row = {'participant_id': pid, 'session_number': int(snum)}
        for st in stage_ids:
            row[f'stage_{st}_pct'] = int((group['final_label'] == st).sum()) / n if n > 0 else 0.0
        part_rows.append(row)

    part_df = pd.DataFrame(part_rows)
    if part_df.empty:
        return ''

    # Aggregate: mean over participants per session_number
    session_numbers = sorted(part_df['session_number'].unique())
    agg = part_df.groupby('session_number').agg(
        n_participants=('participant_id', 'count'),
        **{f'stage_{st}_mean': (f'stage_{st}_pct', 'mean') for st in stage_ids},
    ).reindex(session_numbers)

    fig, ax = plt.subplots(figsize=(max(8, len(session_numbers) * 0.8 + 2), 6))

    for st in stage_ids:
        label = framework[st].get('short_name', f'Stage {st}')
        vals = agg[f'stage_{st}_mean'].values
        ax.plot(session_numbers, vals, marker='o', color=colors[st], label=label, linewidth=2)

    ax.set_xlabel('Session Number')
    ax.set_ylabel('Mean Stage Proportion')
    ax.set_title('Group Longitudinal Stage Trajectory (All Cohorts Combined)')
    ax.set_xticks(session_numbers)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_dir = _ensure_figures_dir(output_dir)
    path = os.path.join(out_dir, 'group_longitudinal_trajectory.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# -----------------------------------------------------------------------
# Figure 3: Per-participant longitudinal trajectories
# -----------------------------------------------------------------------

def plot_participant_longitudinal_trajectory(
    df: pd.DataFrame,
    participant_id: str,
    framework: dict,
    output_dir: str,
) -> str:
    """Line graph of one participant's stage proportions across their sessions.

    X-axis uses actual session_ids the participant attended.
    Returns path to saved PNG.
    """
    stage_ids = sorted(framework.keys())
    colors = _stage_colors(framework)

    pdf = df[df['participant_id'] == participant_id]
    if pdf.empty:
        return ''

    session_ids = sort_session_ids(pdf['session_id'].unique().tolist())

    props_by_session = {}
    for sid in session_ids:
        sdf = pdf[pdf['session_id'] == sid]
        n = len(sdf)
        if n == 0:
            continue
        props_by_session[sid] = {
            st: int((sdf['final_label'] == st).sum()) / n for st in stage_ids
        }

    if not props_by_session:
        return ''

    x_labels = list(props_by_session.keys())
    x_pos = list(range(len(x_labels)))

    fig, ax = plt.subplots(figsize=(max(7, len(x_labels) * 0.9 + 2), 5))

    for st in stage_ids:
        label = framework[st].get('short_name', f'Stage {st}')
        vals = [props_by_session[sid][st] for sid in x_labels]
        ax.plot(x_pos, vals, marker='o', color=colors[st], label=label, linewidth=2)

    ax.set_xlabel('Session')
    ax.set_ylabel('Stage Proportion')
    ax.set_title(f'Longitudinal Stage Trajectory: {participant_id}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_dir = _ensure_figures_dir(output_dir)
    path = os.path.join(out_dir, f'participant_{participant_id}_trajectory.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_all_participant_trajectories(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
) -> list:
    """Generate a trajectory figure for every participant. Returns list of paths."""
    paths = []
    for pid in sorted(df['participant_id'].unique().tolist()):
        path = plot_participant_longitudinal_trajectory(df, pid, framework, output_dir)
        if path:
            paths.append(path)
    return paths


# -----------------------------------------------------------------------
# Figure 4: State transition heatmap
# -----------------------------------------------------------------------

def plot_transition_heatmap(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
) -> str:
    """Heatmap of from-stage -> to-stage transition counts.

    Returns path to saved PNG.
    """
    trans_df = compute_state_transition_matrix(df, framework)
    if trans_df.empty:
        return ''

    labels = trans_df.index.tolist()
    matrix = trans_df.values

    fig, ax = plt.subplots(figsize=(max(6, len(labels) + 2), max(5, len(labels) + 1)))

    im = ax.imshow(matrix, cmap='YlOrRd', interpolation='nearest')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('To Stage')
    ax.set_ylabel('From Stage')
    ax.set_title('State Transition Frequencies')

    # Annotate cells with counts
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            text_color = 'white' if val > matrix.max() * 0.6 else 'black'
            ax.text(j, i, str(int(val)), ha='center', va='center',
                    fontsize=10, color=text_color, fontweight='bold')

    fig.colorbar(im, ax=ax, shrink=0.8, label='Count')
    fig.tight_layout()

    path = os.path.join(_ensure_figures_dir(output_dir), 'stage_transition_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# -----------------------------------------------------------------------
# Figure 5: Between-session (cross-session) transition heatmap
# -----------------------------------------------------------------------

def plot_cross_session_transition_heatmap(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
) -> str:
    """Heatmap of between-session dominant-theme transition counts.

    Each cell (row=from_stage, col=to_stage) counts how many times a
    participant's dominant theme changed from one session to the next.
    Saved to reports/analysis/figures/cross_session_transition_heatmap.png.
    """
    cross_matrix, _ = compute_cross_session_transitions(df, framework)
    if cross_matrix.empty:
        return ''

    labels = cross_matrix.index.tolist()
    matrix = cross_matrix.values

    fig, ax = plt.subplots(figsize=(max(6, len(labels) + 2), max(5, len(labels) + 1)))

    im = ax.imshow(matrix, cmap='Blues', interpolation='nearest')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('To Stage (next session)')
    ax.set_ylabel('From Stage (current session)')
    ax.set_title('Between-Session Theme Transitions\n(Dominant Stage: Session N → Session N+1)')

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            text_color = 'white' if val > matrix.max() * 0.6 else 'black'
            ax.text(j, i, str(int(val)), ha='center', va='center',
                    fontsize=10, color=text_color, fontweight='bold')

    fig.colorbar(im, ax=ax, shrink=0.8, label='Transitions')
    fig.tight_layout()

    out_dir = _ensure_figures_dir(output_dir)
    path = os.path.join(out_dir, 'cross_session_transition_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# -----------------------------------------------------------------------
# Figure 6: Per-stage prevalence by session (bar chart)
# -----------------------------------------------------------------------

def plot_stage_prevalence_by_session(
    df: pd.DataFrame,
    stage_id: int,
    framework: dict,
    output_dir: str,
) -> str:
    """Bar chart of per-session prevalence for a single VA-MR stage.

    Includes a horizontal line for overall mean prevalence.
    Saved to reports/analysis/figures/stage_{name}_prevalence.png.
    """
    stage_info = framework.get(stage_id, {})
    stage_name = stage_info.get('short_name', f'Stage {stage_id}')
    color = _stage_colors(framework)[stage_id]

    session_numbers = sorted(df['session_number'].unique().tolist())
    prevalences = []
    for snum in session_numbers:
        sdf = df[df['session_number'] == snum]
        n = len(sdf)
        cnt = int((sdf['final_label'] == stage_id).sum())
        prevalences.append(cnt / n if n > 0 else 0.0)

    overall_mean = sum(prevalences) / len(prevalences) if prevalences else 0.0

    fig, ax = plt.subplots(figsize=(max(7, len(session_numbers) * 0.8 + 2), 5))

    bars = ax.bar(range(len(session_numbers)), prevalences, color=color, alpha=0.8, edgecolor='white')
    ax.axhline(overall_mean, color='#333333', linestyle='--', linewidth=1.5,
               label=f'Mean {overall_mean:.1%}')

    ax.set_xticks(range(len(session_numbers)))
    ax.set_xticklabels([str(s) for s in session_numbers])
    ax.set_xlabel('Session Number')
    ax.set_ylabel('Proportion of Segments')
    ax.set_title(f'{stage_name}: Prevalence Across Sessions')
    ax.set_ylim(0, min(1.05, max(prevalences) * 1.3 + 0.1) if prevalences else 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    slug = re.sub(r'[^a-z0-9]+', '_', stage_name.lower()).strip('_')
    out_dir = _ensure_figures_dir(output_dir)
    path = os.path.join(out_dir, f'stage_{slug}_prevalence.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_all_stage_prevalence_charts(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
) -> list:
    """Generate prevalence-by-session bar charts for all stages."""
    paths = []
    for stage_id in sorted(framework.keys()):
        try:
            path = plot_stage_prevalence_by_session(df, stage_id, framework, output_dir)
            if path:
                paths.append(path)
        except Exception as e:
            print(f"  Warning: stage prevalence chart failed for stage {stage_id}: {e}")
    return paths


# -----------------------------------------------------------------------
# Figure 7: Per-session stage timeline strip chart
# -----------------------------------------------------------------------

def plot_session_stage_timeline(
    df: pd.DataFrame,
    session_id: str,
    framework: dict,
    output_dir: str,
) -> str:
    """Color-coded strip chart showing stage sequence across a session.

    One horizontal row per participant; each cell = one segment, colored by stage.
    X-axis = segment position in temporal order within the session.
    Returns path to saved PNG.
    """
    sdf = df[df['session_id'] == session_id].copy()
    if sdf.empty:
        return ''

    stage_ids = sorted(framework.keys())
    colors = _stage_colors(framework)
    participant_ids = sorted(sdf['participant_id'].unique().tolist())

    sort_col = 'segment_index' if 'segment_index' in sdf.columns else 'start_time_ms'
    sdf = sdf.sort_values(sort_col)

    # Assign a within-session position to each participant's segments, preserving temporal order
    # Collect per-participant sequence of stage labels sorted by time
    participant_sequences = {}
    participant_secondary_sequences = {}
    has_secondary = 'secondary_stage' in sdf.columns and sdf['secondary_stage'].notna().any()
    for pid in participant_ids:
        psdf = sdf[sdf['participant_id'] == pid].sort_values(sort_col)
        participant_sequences[pid] = [int(v) for v in psdf['final_label'].tolist() if pd.notna(v)]
        if has_secondary:
            participant_secondary_sequences[pid] = [
                int(v) if pd.notna(v) else None
                for v in psdf['secondary_stage'].tolist()
            ]

    max_len = max((len(seq) for seq in participant_sequences.values()), default=0)
    if max_len == 0:
        return ''

    n_participants = len(participant_ids)
    cell_w = max(0.3, min(0.6, 20.0 / max_len))
    fig_w = max(8, max_len * cell_w + 3)
    fig_h = max(2.5, n_participants * 0.8 + 1.5)

    fig, axes = plt.subplots(
        n_participants, 1,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={'hspace': 0.08},
    )

    for row_idx, pid in enumerate(participant_ids):
        ax = axes[row_idx][0]
        seq = participant_sequences[pid]
        n_segs = len(seq)
        for seg_idx, stage in enumerate(seq):
            color = colors.get(stage, '#AAAAAA')
            ax.barh(0, 1, left=seg_idx, height=0.8, color=color, edgecolor='white', linewidth=0.5)

            # Secondary stage overlay (hatching if cell wide enough and secondary exists)
            if has_secondary and cell_w > 0.25 and pid in participant_secondary_sequences:
                sec_stage = participant_secondary_sequences[pid][seg_idx] if seg_idx < len(participant_secondary_sequences[pid]) else None
                if sec_stage is not None and sec_stage != stage:
                    sec_color = colors.get(sec_stage, '#AAAAAA')
                    ax.barh(0, 1, left=seg_idx, height=0.8,
                            color='none', edgecolor=sec_color,
                            linewidth=1.5, hatch='///', alpha=0.6)

        ax.set_xlim(0, max_len)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([0])
        ax.set_yticklabels([pid], fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
        if row_idx < n_participants - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Segment position (temporal order)', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(f'Session {session_id} — Stage Sequence by Participant', fontsize=11, y=1.01)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[sid], label=framework[sid].get('short_name', f'Stage {sid}'))
        for sid in stage_ids
    ]
    if has_secondary and any(
        v is not None for seqs in participant_secondary_sequences.values() for v in seqs
    ):
        legend_elements.append(
            Patch(facecolor='white', edgecolor='gray', hatch='///',
                  label='Hatching = secondary stage (color = stage)')
        )
    axes[0][0].legend(
        handles=legend_elements,
        loc='upper left', bbox_to_anchor=(1.01, 1),
        fontsize=8, frameon=True,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*tight_layout.*')
        fig.tight_layout()
    out_dir = _ensure_figures_dir(output_dir)
    path = os.path.join(out_dir, f'session_{session_id}_stage_timeline.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def generate_all_session_stage_timelines(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
) -> list:
    """Generate stage timeline figures for all sessions. Returns list of paths."""
    from .loader import sort_session_ids
    paths = []
    session_ids = sort_session_ids(df['session_id'].unique().tolist())
    for sid in session_ids:
        try:
            path = plot_session_stage_timeline(df, sid, framework, output_dir)
            if path:
                paths.append(path)
        except Exception as e:
            print(f"  Warning: stage timeline figure failed for session {sid}: {e}")
    return paths


# -----------------------------------------------------------------------
# Figure 8: Full program longitudinal progression strip chart
# -----------------------------------------------------------------------

def plot_program_longitudinal_progression(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
) -> str:
    """Color-coded strip chart showing stage sequence across entire program.

    One horizontal row per participant; each cell = one segment, colored by stage.
    X-axis = segment position in global temporal order across all sessions.
    Shows the complete longitudinal progression for each participant.
    Returns path to saved PNG.
    """
    if df.empty:
        return ''

    stage_ids = sorted(framework.keys())
    colors = _stage_colors(framework)
    participant_ids = sorted(df['participant_id'].unique().tolist())

    # Sort all data temporally (globally across all sessions)
    sort_col = 'segment_index' if 'segment_index' in df.columns else 'start_time_ms'
    sdf = df.sort_values(sort_col).copy()

    # Collect per-participant sequence of stage labels in temporal order across entire program
    participant_sequences = {}
    participant_secondary_sequences = {}
    has_secondary = 'secondary_stage' in sdf.columns and sdf['secondary_stage'].notna().any()
    for pid in participant_ids:
        psdf = sdf[sdf['participant_id'] == pid].sort_values(sort_col)
        participant_sequences[pid] = [int(v) for v in psdf['final_label'].tolist() if pd.notna(v)]
        if has_secondary:
            participant_secondary_sequences[pid] = [
                int(v) if pd.notna(v) else None
                for v in psdf['secondary_stage'].tolist()
            ]

    max_len = max((len(seq) for seq in participant_sequences.values()), default=0)
    if max_len == 0:
        return ''

    n_participants = len(participant_ids)
    cell_w = max(0.2, min(0.5, 20.0 / max_len))
    fig_w = max(10, max_len * cell_w + 3)
    fig_h = max(3, n_participants * 0.8 + 2)

    fig, axes = plt.subplots(
        n_participants, 1,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={'hspace': 0.1},
    )

    for row_idx, pid in enumerate(participant_ids):
        ax = axes[row_idx][0]
        seq = participant_sequences[pid]
        n_segs = len(seq)
        for seg_idx, stage in enumerate(seq):
            color = colors.get(stage, '#AAAAAA')
            ax.barh(0, 1, left=seg_idx, height=0.8, color=color, edgecolor='white', linewidth=0.5)

            # Secondary stage overlay (hatching if cell wide enough and secondary exists)
            if has_secondary and cell_w > 0.25 and pid in participant_secondary_sequences:
                sec_stage = participant_secondary_sequences[pid][seg_idx] if seg_idx < len(participant_secondary_sequences[pid]) else None
                if sec_stage is not None and sec_stage != stage:
                    sec_color = colors.get(sec_stage, '#AAAAAA')
                    ax.barh(0, 1, left=seg_idx, height=0.8,
                            color='none', edgecolor=sec_color,
                            linewidth=1.5, hatch='///', alpha=0.6)

        ax.set_xlim(0, max_len)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([0])
        ax.set_yticklabels([pid], fontsize=8)
        ax.tick_params(axis='x', labelsize=6)
        if row_idx < n_participants - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Segment position (global temporal order)', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(
        f'Full Program Longitudinal Progression — Stage Sequence by Participant\n'
        f'({n_participants} participants, {max_len} total segments)',
        fontsize=12, y=0.995, fontweight='bold'
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[sid], label=framework[sid].get('short_name', f'Stage {sid}'))
        for sid in stage_ids
    ]
    if has_secondary and any(
        v is not None for seqs in participant_secondary_sequences.values() for v in seqs
    ):
        legend_elements.append(
            Patch(facecolor='white', edgecolor='gray', hatch='///',
                  label='Hatching = secondary stage (color = stage)')
        )
    axes[0][0].legend(
        handles=legend_elements,
        loc='upper left', bbox_to_anchor=(1.01, 1),
        fontsize=8, frameon=True,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*tight_layout.*')
        fig.tight_layout()
    out_dir = _ensure_figures_dir(output_dir)
    path = os.path.join(out_dir, 'program_longitudinal_progression.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# -----------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Superposition figures (mixture timelines, continuous trajectories, cusps)
# -----------------------------------------------------------------------

def plot_session_mixture_timeline(df, session_id, framework, output_dir):
    """Stacked-area of per-segment VAAMR mixture across one session."""
    if 'mixture' not in df.columns:
        return None
    sdf = df[df['session_id'] == session_id].sort_values('segment_index')
    if sdf.empty:
        return None
    stage_ids = sorted(framework.keys())
    n_stages = len(stage_ids)
    mat = np.array([np.asarray(m, dtype=float) for m in sdf['mixture']
                    if m is not None and len(m) == n_stages])
    if mat.size == 0:
        return None
    colors = _stage_colors(framework)
    x = np.arange(mat.shape[0])
    fig, ax = plt.subplots(figsize=(max(6, mat.shape[0] * 0.25), 4))
    ax.stackplot(x, *[mat[:, k] for k in range(n_stages)],
                 colors=[colors[stage_ids[k]] for k in range(n_stages)],
                 labels=[framework[stage_ids[k]].get('short_name', str(stage_ids[k])) for k in range(n_stages)])
    ax.set_xlim(0, max(1, mat.shape[0] - 1))
    ax.set_ylim(0, 1)
    ax.set_xlabel('Segment (in order)')
    ax.set_ylabel('Stage mixture')
    ax.set_title(f'Session {session_id} — VAAMR stage superposition')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=7, frameon=True)
    fig.tight_layout()
    out = _ensure_figures_dir(output_dir)
    path = os.path.join(out, f'session_{session_id}_mixture_timeline.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_participant_progression_trajectory(df, participant_id, framework, output_dir):
    """Continuous progression coordinate over sessions with an entropy band."""
    if 'progression_coord' not in df.columns:
        return None
    pdf = df[df['participant_id'] == participant_id]
    if pdf.empty:
        return None
    sids = sort_session_ids(pdf['session_id'].unique().tolist())
    means, ents, labels = [], [], []
    for sid in sids:
        vals = pdf[pdf['session_id'] == sid]['progression_coord'].dropna()
        if len(vals) == 0:
            continue
        means.append(float(vals.mean()))
        e = pdf[pdf['session_id'] == sid]['mixture_entropy'].dropna()
        ents.append(float(e.mean()) if len(e) else 0.0)
        labels.append(sid)
    if not means:
        return None
    x = np.arange(len(means))
    means = np.array(means)
    ents = np.array(ents)
    fig, ax = plt.subplots(figsize=(max(5, len(means) * 0.7), 4))
    ax.fill_between(x, means - ents, means + ents, color='#648FFF', alpha=0.18,
                    label='±mean entropy (liminality)')
    ax.plot(x, means, '-o', color='#648FFF', lw=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(-0.2, len(framework) - 0.8)
    ax.set_ylabel('Progression coordinate (E[stage])')
    ax.set_title(f'{participant_id} — continuous VAAMR progression')
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = _ensure_figures_dir(output_dir)
    path = os.path.join(out, f'participant_{participant_id}_progression_trajectory.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_superposition_entropy_by_session(df, framework, output_dir):
    """Group mean liminality (mixture entropy) across sessions, longitudinal order."""
    if 'mixture_entropy' not in df.columns:
        return None
    sids = sort_session_ids(df['session_id'].unique().tolist())
    means, labels = [], []
    for sid in sids:
        e = df[df['session_id'] == sid]['mixture_entropy'].dropna()
        if len(e):
            means.append(float(e.mean()))
            labels.append(sid)
    if not means:
        return None
    x = np.arange(len(means))
    fig, ax = plt.subplots(figsize=(max(5, len(means) * 0.5), 4))
    ax.plot(x, means, '-o', color='#FE6100', lw=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Mean mixture entropy (liminality)')
    ax.set_title('Liminality across the program')
    fig.tight_layout()
    out = _ensure_figures_dir(output_dir)
    path = os.path.join(out, 'superposition_entropy_by_session.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_stage_cooccurrence_matrix(df, framework, output_dir):
    """Heatmap of which VAAMR stage pairs co-express (the cusp matrix)."""
    if 'mixture' not in df.columns:
        return None
    from .superposition import stage_cooccurrence_matrix
    stage_ids = sorted(framework.keys())
    n_stages = len(stage_ids)
    mat = np.array(stage_cooccurrence_matrix(df, n_stages))
    if mat.size == 0:
        return None
    names = [framework[s].get('short_name', str(s)) for s in stage_ids]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(mat, cmap='magma')
    ax.set_xticks(range(n_stages)); ax.set_yticks(range(n_stages))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(n_stages):
        for j in range(n_stages):
            ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center',
                    color='white' if mat[i, j] < mat.max() * 0.6 else 'black', fontsize=7)
    ax.set_title('Stage co-occurrence (cusp matrix)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = _ensure_figures_dir(output_dir)
    path = os.path.join(out, 'stage_cooccurrence_matrix.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def generate_superposition_figures(df, framework, output_dir):
    """Generate all superposition figures. Returns list of PNG paths."""
    paths = []
    for name, gen in (
        ('entropy by session', lambda: plot_superposition_entropy_by_session(df, framework, output_dir)),
        ('stage cooccurrence', lambda: plot_stage_cooccurrence_matrix(df, framework, output_dir)),
    ):
        try:
            r = gen()
            if r:
                paths.append(r)
        except Exception as e:
            print(f"  Warning: {name} figure failed: {e}")
    # Per-session mixture timelines.
    for sid in sort_session_ids(df['session_id'].unique().tolist()):
        try:
            r = plot_session_mixture_timeline(df, sid, framework, output_dir)
            if r:
                paths.append(r)
        except Exception as e:
            print(f"  Warning: mixture timeline for {sid} failed: {e}")
    # Per-participant continuous trajectories.
    for pid in sorted(df['participant_id'].unique().tolist()):
        try:
            r = plot_participant_progression_trajectory(df, pid, framework, output_dir)
            if r:
                paths.append(r)
        except Exception as e:
            print(f"  Warning: progression trajectory for {pid} failed: {e}")
    return paths


def plot_delta_progression_heatmap(output_dir, framework):
    """Behaviour × FROM-stage heatmap of mean Δprogression (reads mechanism CSV)."""
    csv = os.path.join(_paths.mechanism_dir(output_dir), 'mechanism_delta_progression.csv')
    if not os.path.isfile(csv):
        return None
    mdf = pd.read_csv(csv)
    # Prefer motif grouping if present, else PURER.
    grouping = 'motif' if (mdf['grouping'] == 'motif').any() else 'purer'
    g = mdf[mdf['grouping'] == grouping]
    if g.empty:
        return None
    behaviors = sorted(g['behavior'].astype(str).unique().tolist())
    from_stages = sorted(g['from_stage'].unique().tolist())
    mat = np.full((len(behaviors), len(from_stages)), np.nan)
    sig = np.zeros((len(behaviors), len(from_stages)), dtype=bool)
    has_fdr = 'fdr_significant' in g.columns
    for _, r in g.iterrows():
        i = behaviors.index(str(r['behavior']))
        j = from_stages.index(r['from_stage'])
        mat[i, j] = r['mean_delta_prog']
        if has_fdr and bool(r['fdr_significant']):
            sig[i, j] = True
    fig, ax = plt.subplots(figsize=(max(4, len(from_stages) * 1.2), max(3, len(behaviors) * 0.4)))
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(from_stages)))
    ax.set_xticklabels([framework.get(s, {}).get('short_name', str(s)) for s in from_stages],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(behaviors)))
    ax.set_yticklabels([b[:22] for b in behaviors], fontsize=7)
    # Star FDR-significant cells (q<.05) so only defensible effects pop visually.
    for i in range(len(behaviors)):
        for j in range(len(from_stages)):
            if sig[i, j]:
                ax.text(j, i, '*', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
    ttl = f'Mean Δprogression by {grouping} × FROM-stage'
    if has_fdr:
        ttl += '  (* = FDR q<.05)'
    ax.set_title(ttl)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = _ensure_figures_dir(output_dir)
    path = os.path.join(out, 'cue_motif_delta_progression.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_gnn_vs_llm_convergence(output_dir):
    """Scatter of GNN lift vs LLM lift from gnn_vs_llm_lift.csv (construct validity)."""
    csv = os.path.join(_paths.gnn_data_dir(output_dir), 'gnn_vs_llm_lift.csv')
    if not os.path.isfile(csv):
        return None
    cdf = pd.read_csv(csv)
    cols = set(cdf.columns)
    xcol = 'lift_llm' if 'lift_llm' in cols else None
    ycol = 'lift_gnn' if 'lift_gnn' in cols else None
    if not xcol or not ycol:
        return None
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(cdf[xcol], cdf[ycol], alpha=0.5, color='#785EF0', s=18)
    lim = max(float(cdf[xcol].max()), float(cdf[ycol].max()), 2.0)
    ax.plot([0, lim], [0, lim], '--', color='gray', lw=1)
    ax.axhline(1.5, color='#FE6100', lw=0.8, ls=':'); ax.axvline(1.5, color='#FE6100', lw=0.8, ls=':')
    ax.set_xlabel('LLM lift'); ax.set_ylabel('GNN lift')
    ax.set_title('GNN ↔ LLM lift convergence')
    fig.tight_layout()
    out = _ensure_figures_dir(output_dir)
    path = os.path.join(out, 'gnn_vs_llm_lift_convergence.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def generate_mechanism_figures(output_dir, framework):
    """Figures that depend on mechanism CSVs. Returns list of PNG paths."""
    paths = []
    for name, gen in (
        ('delta progression heatmap', lambda: plot_delta_progression_heatmap(output_dir, framework)),
        ('gnn vs llm convergence', lambda: plot_gnn_vs_llm_convergence(output_dir)),
    ):
        try:
            r = gen()
            if r:
                paths.append(r)
        except Exception as e:
            print(f"  Warning: {name} figure failed: {e}")
    return paths


def generate_all_figures(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
) -> list:
    """Generate all analysis figures. Returns list of PNG paths."""
    paths = []

    generators = [
        ('dominant stage heatmap', lambda: plot_dominant_stage_heatmap(df, framework, output_dir)),
        ('group longitudinal trajectory', lambda: plot_group_longitudinal_trajectory(df, framework, output_dir)),
        ('per-participant trajectories', lambda: plot_all_participant_trajectories(df, framework, output_dir)),
        ('state transition heatmap', lambda: plot_transition_heatmap(df, framework, output_dir)),
        ('cross-session transition heatmap', lambda: plot_cross_session_transition_heatmap(df, framework, output_dir)),
        ('stage prevalence charts', lambda: plot_all_stage_prevalence_charts(df, framework, output_dir)),
        ('program longitudinal progression', lambda: plot_program_longitudinal_progression(df, framework, output_dir)),
    ]

    for name, gen in generators:
        try:
            result = gen()
            if isinstance(result, list):
                paths.extend(result)
            elif result:
                paths.append(result)
        except Exception as e:
            print(f"  Warning: {name} figure failed: {e}")

    return paths
