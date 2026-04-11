"""
analysis/figures.py
-------------------
Matplotlib figure generation for the analysis module.

All figures are saved as PNGs to {output_dir}/reports/analysis/figures/.
Uses the Agg backend for headless compatibility.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from .loader import sort_session_ids
from .stage_progression import compute_state_transition_matrix

# Qualitative palette for up to 8 stages; extra stages get cycled.
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
    """Return {stage_id: hex_color} for consistent coloring."""
    stage_ids = sorted(framework.keys())
    return {sid: _BASE_COLORS[i % len(_BASE_COLORS)] for i, sid in enumerate(stage_ids)}


def _ensure_figures_dir(output_dir: str) -> str:
    out = os.path.join(output_dir, 'reports', 'analysis', 'figures')
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

    out_dir = _ensure_figures_dir(output_dir)
    path = os.path.join(out_dir, 'state_transition_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# -----------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------

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
