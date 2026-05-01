"""
analysis/purer_figures.py
-------------------------
Matplotlib figures for PURER × VAMMR analysis.

Parallel to analysis/figures.py. Uses the same Agg backend and PNG/150-dpi pattern.

Exports:
    generate_purer_figures(influence, framework, output_dir) -> list[str]
        influence: dict returned by purer_analysis.run_purer_analysis()
        framework: dict {stage_id: {short_name, name, ...}} from loader.load_framework()
        Returns list of PNG paths written (empty when no PURER labels present).
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from process import output_paths as _paths

# PURER construct ordering and labels
_PURER_IDS   = [0, 1, 2, 3, 4]
_PURER_SHORT = {0: 'P', 1: 'U', 2: 'R', 3: 'E', 4: 'R2'}
_PURER_NAME  = {
    0: 'Phenomenology',
    1: 'Utilization',
    2: 'Reframing',
    3: 'Education',
    4: 'Reinforcement',
}
# Per-construct colors (visually distinct, accessible)
_PURER_COLORS = {
    0: '#5B8DB8',   # steel blue   — Phenomenology
    1: '#8DA87E',   # sage green   — Utilization
    2: '#C97B3A',   # burnt orange — Reframing
    3: '#9B6BA8',   # plum         — Education
    4: '#D4A853',   # gold         — Reinforcement
}


def _ensure_figures_dir(output_dir: str) -> str:
    out = _paths.figures_dir(output_dir)
    os.makedirs(out, exist_ok=True)
    return out


# ── Figure A: PURER × VAMMR Lift Heatmap ──────────────────────────────────

def plot_purer_lift_heatmap(
    influence: dict,
    framework: dict,
    output_dir: str,
) -> str:
    """
    5×5 annotated heatmap: rows = PURER constructs, cols = VAMMR stages.
    Cell value = lift = P(stage | dominant PURER) / P(stage base rate).
    Diverging colormap centred at 1.0.

    Returns path to saved PNG, or '' if insufficient data.
    """
    lift_df: pd.DataFrame = influence.get('lift_matrix', pd.DataFrame())
    if lift_df is None or lift_df.empty:
        return ''

    # Identify which VAMMR stage columns are present
    to_stage_cols = sorted(
        [c for c in lift_df.columns if c.startswith('lift_to_')],
        key=lambda c: int(c.replace('lift_to_', '')),
    )
    if not to_stage_cols:
        return ''
    stage_ids = [int(c.replace('lift_to_', '')) for c in to_stage_cols]
    stage_labels = [
        framework.get(sid, {}).get('short_name', str(sid))
        if isinstance(framework, dict) else str(sid)
        for sid in stage_ids
    ]

    # Build matrix: rows ordered by PURER ID, cols ordered by stage_id
    purer_labels_rows = []
    matrix_rows = []
    n_blocks_rows = []

    for pid in _PURER_IDS:
        row_label = f"{_PURER_NAME.get(pid, str(pid))} ({_PURER_SHORT.get(pid, str(pid))})"
        # Find matching row in lift_df by purer_short
        short = _PURER_SHORT.get(pid, str(pid))
        name  = _PURER_NAME.get(pid, str(pid))
        mask = (
            (lift_df.get('purer_short', pd.Series(dtype=str)) == short) |
            (lift_df.get('purer_construct', pd.Series(dtype=str)) == name)
        )
        matching = lift_df[mask]
        if matching.empty:
            vals = [0.0] * len(to_stage_cols)
            n_b = 0
        else:
            row = matching.iloc[0]
            vals = [float(row.get(c, 0.0)) for c in to_stage_cols]
            n_b = int(row.get('n_blocks', 0))
        purer_labels_rows.append(row_label)
        matrix_rows.append(vals)
        n_blocks_rows.append(n_b)

    matrix = np.array(matrix_rows, dtype=float)
    n_rows, n_cols = matrix.shape

    # Diverging colormap centred at 1.0
    # Clamp display range to [0, 3] so extreme values don't wash out the map.
    vmin, vmax = 0.0, 3.0
    vcenter = 1.0
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = plt.cm.RdYlGn

    fig, ax = plt.subplots(figsize=(max(7, n_cols * 1.5 + 3), max(5, n_rows * 0.9 + 2)))

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(stage_labels, fontsize=10, fontweight='bold')
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(purer_labels_rows, fontsize=9)
    ax.set_xlabel('VAMMR Stage (participant, next turn)', fontsize=10)
    ax.set_ylabel('PURER Construct (therapist, dominant in cue block)', fontsize=10)
    ax.set_title('PURER × VAMMR Lift Matrix\n'
                 'P(stage | dominant PURER move) / P(stage base rate)',
                 fontsize=11, fontweight='bold')

    # Annotate each cell
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            txt = f'{val:.2f}'
            # Use dark text on light cells, light text on dark cells
            bg = cmap(norm(val))
            brightness = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            text_color = 'black' if brightness > 0.5 else 'white'
            weight = 'bold' if val >= 1.5 else 'normal'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=11, color=text_color, fontweight=weight)

    # N-blocks annotation on y-axis (after PURER label)
    for i, n_b in enumerate(n_blocks_rows):
        ax.annotate(
            f'n={n_b}',
            xy=(n_cols - 0.5, i),
            xytext=(n_cols + 0.1, i),
            xycoords='data',
            textcoords='data',
            fontsize=8,
            color='#555555',
            va='center',
        )

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.12)
    cbar.set_label('Lift  (1.0 = base rate)', fontsize=9)
    cbar.ax.axhline(y=1.0, color='black', linewidth=1.5, linestyle='--')

    # Neutral line at 1.0
    n_mediated = influence.get('n_mediated', '?')
    if not isinstance(n_mediated, str):
        n_mediated = int(n_mediated) if n_mediated else '?'
    fig.text(
        0.5, 0.01,
        f'Lift = P(stage | dominant PURER) / P(stage base rate).  '
        f'Only mediated cue blocks (N={n_mediated}).  '
        f'Bold cell ≥ 1.5.  Dashed colorbar line = 1.0.',
        ha='center', fontsize=7.5, color='#555555',
    )

    fig.tight_layout(rect=[0, 0.03, 1, 1])

    path = os.path.join(_ensure_figures_dir(output_dir), 'purer_lift_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Figure B: PURER Transition Profiles (horizontal bar chart) ────────────

def plot_purer_transition_profiles(
    influence: dict,
    framework: dict,
    output_dir: str,
) -> str:
    """
    Horizontal grouped bar chart: for each transition type, shows the
    fraction of mediated cue blocks where each PURER construct is dominant.

    Rows = transition types (top 8 by n, non-lateral only).
    Bar groups = PURER constructs.

    Returns path to saved PNG, or '' if insufficient data.
    """
    profiles: pd.DataFrame = influence.get('transition_profiles', pd.DataFrame())
    if profiles is None or profiles.empty:
        return ''

    required = {'from_stage', 'to_stage', 'dominant_purer', 'fraction_of_mediated', 'mediated_total'}
    if not required.issubset(profiles.columns):
        return ''

    # Select non-lateral transitions, sorted by total mediated blocks desc
    non_lateral = profiles[profiles['from_stage'] != profiles['to_stage']].copy()
    if non_lateral.empty:
        return ''

    # Get top 8 transition types by mediated_total
    top_trans = (
        non_lateral.groupby(['from_stage', 'to_stage'])['mediated_total']
        .first()
        .sort_values(ascending=False)
        .head(8)
        .reset_index()
    )

    def stage_short(sid: int) -> str:
        if isinstance(framework, dict):
            return framework.get(sid, {}).get('short_name', str(sid))
        return str(sid)

    def arrow(fr: int, to: int) -> str:
        sym = '→' if to > fr else '←'
        return f'{stage_short(fr)} {sym} {stage_short(to)}'

    # Build y-axis labels and matrix
    trans_labels = []
    matrix = np.zeros((len(top_trans), len(_PURER_IDS)), dtype=float)
    n_totals = []

    for row_i, row in top_trans.iterrows():
        fs, ts = int(row['from_stage']), int(row['to_stage'])
        n_med = int(row['mediated_total'])
        trans_labels.append(f'{arrow(fs, ts)}  (n={n_med})')
        n_totals.append(n_med)
        sub = non_lateral[(non_lateral['from_stage'] == fs) & (non_lateral['to_stage'] == ts)]
        for col_j, pid in enumerate(_PURER_IDS):
            frac_row = sub[sub['dominant_purer'] == pid]
            if not frac_row.empty:
                matrix[row_i, col_j] = float(frac_row['fraction_of_mediated'].iloc[0])

    n_rows = len(trans_labels)
    bar_height = 0.12
    group_height = len(_PURER_IDS) * bar_height + 0.08

    fig_height = max(4.0, n_rows * group_height + 1.0)
    fig, ax = plt.subplots(figsize=(9, fig_height))

    y_positions = np.arange(n_rows) * group_height

    for col_j, pid in enumerate(_PURER_IDS):
        offsets = y_positions + col_j * bar_height
        ax.barh(
            offsets,
            matrix[:, col_j],
            height=bar_height * 0.85,
            color=_PURER_COLORS.get(pid, '#999999'),
            label=f'{_PURER_SHORT[pid]} — {_PURER_NAME[pid]}',
            alpha=0.88,
        )

    # Y-tick at group centre
    group_centres = y_positions + (len(_PURER_IDS) / 2 - 0.5) * bar_height
    ax.set_yticks(group_centres)
    ax.set_yticklabels(trans_labels, fontsize=9)
    ax.invert_yaxis()

    ax.set_xlabel('Fraction of mediated cue blocks (dominant PURER move)', fontsize=9)
    ax.set_title('PURER Move Distribution by Transition Type\n'
                 '(mediated cue blocks only; dominant construct per block)',
                 fontsize=10, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.axvline(x=0, color='black', linewidth=0.5)

    ax.legend(
        loc='lower right',
        fontsize=8,
        title='PURER construct',
        title_fontsize=8,
        framealpha=0.85,
    )
    fig.tight_layout()

    path = os.path.join(_ensure_figures_dir(output_dir), 'purer_transition_profiles.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Orchestrator ──────────────────────────────────────────────────────────

def generate_purer_figures(
    influence: dict,
    framework: dict,
    output_dir: str,
) -> list:
    """
    Generate all PURER figures. Returns list of PNG paths written.

    Skips gracefully when influence tables are empty (no PURER labels yet).
    """
    paths = []
    generators = [
        ('PURER lift heatmap',          lambda: plot_purer_lift_heatmap(influence, framework, output_dir)),
        ('PURER transition profiles',   lambda: plot_purer_transition_profiles(influence, framework, output_dir)),
    ]
    for name, gen in generators:
        try:
            result = gen()
            if result:
                paths.append(result)
        except Exception as e:
            print(f'  Warning: {name} figure failed: {e}')
    return paths
