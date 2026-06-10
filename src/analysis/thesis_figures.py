"""
analysis/thesis_figures.py
--------------------------
The three flagship "thesis figures" for the QRA pipeline, written to the
06_reports root next to 00_RESULTS.txt. These are presentation-grade,
self-contained figures intended to be shown at a research meeting.

  Fig 1 — THE RE-HABITUATION ARC (validated participant-side data only)
  Fig 2 — DYADIC MECHANISM MAP (directional: PURER labels unvalidated)
  Fig 3 — FOUR-PANEL DASHBOARD (outcomes / mechanism / reliability at a glance)

Entry point: ``generate_thesis_figures(df, df_all, framework, output_dir)``.
Each figure is independently wrapped: a single failure (or missing input)
skips that figure with a printed warning, never killing the others.

Everything is computed from on-disk artifacts under ``output_dir`` (the
efficacy / mechanism CSVs, the IRR results JSON) and from the labeled
participant DataFrame — no numbers are hard-coded.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch, Circle
import numpy as np
import pandas as pd

from process import output_paths as _paths
from .figures import _VAAMR_COLORS
from .reports.stat_format import fmt_signed, fmt_p, fmt_est_ci


# ── shared presentation style ──────────────────────────────────────────────

_BG = 'white'
_GRID = '#D8D8D8'
_INK = '#1A1A1A'
_MUTED = '#555555'

# Maladaptive (warm) → adaptive (cool) bottom-to-top stacking order for the
# 100% occupancy "stream". Vigilance + Avoidance at the bottom (the obstacle),
# the three adaptive stages above.
_STREAM_ORDER = [0, 1, 2, 3, 4]
_MALADAPTIVE = (0, 1)
_BARRIER_BELOW = (0, 1)  # bands below the barrier line


def _short(framework: dict, sid: int, fallback: str = None) -> str:
    info = framework.get(sid, {}) if framework else {}
    return info.get('short_name') or info.get('name') or (fallback or f'Stage {sid}')


# Spelled-out display names (the framework short_name truncates stage 2).
_FULL_NAME = {
    0: 'Vigilance',
    1: 'Avoidance',
    2: 'Attention Regulation',
    3: 'Metacognition',
    4: 'Reappraisal',
}


def _stage_name(framework: dict, sid: int) -> str:
    return _FULL_NAME.get(sid, _short(framework, sid))


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _apply_axis_style(ax):
    ax.set_facecolor(_BG)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#888888')
    ax.tick_params(colors=_MUTED, labelsize=9)


# ── occupancy stream (shared by Fig 1 + Fig 3A) ────────────────────────────

def _occupancy_by_session(df: pd.DataFrame, stage_ids) -> tuple:
    """100%-stacked occupancy per session number.

    Returns (sessions, {stage_id: array_of_fractions}) where fractions are the
    participant-equal-weighted mean share of each stage at that session number.
    """
    sessions = sorted(int(s) for s in df['session_number'].dropna().unique() if int(s) > 0)
    # Per-participant per-session stage shares, then mean over participants so a
    # talkative participant doesn't dominate (mirrors figures.py group trajectory).
    part_rows = []
    for (pid, snum), g in df.groupby(['participant_id', 'session_number']):
        snum = int(snum)
        if snum <= 0 or len(g) == 0:
            continue
        row = {'participant_id': pid, 'session_number': snum}
        for st in stage_ids:
            row[st] = float((g['final_label'] == st).mean())
        part_rows.append(row)
    pdf = pd.DataFrame(part_rows)
    occ = {st: [] for st in stage_ids}
    keep = []
    for snum in sessions:
        sub = pdf[pdf['session_number'] == snum]
        if sub.empty:
            continue
        keep.append(snum)
        tot = sum(float(sub[st].mean()) for st in stage_ids) or 1.0
        for st in stage_ids:
            occ[st].append(float(sub[st].mean()) / tot)
    return keep, {st: np.array(occ[st]) for st in stage_ids}


def _tint(color, frac=0.45):
    """Blend a color toward white by `frac` — opaque pastels keep the stack
    crisp (no translucent-overlap muddiness) while dark overlays stay legible."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


def _draw_occupancy_stream(ax, sessions, occ, framework, alpha=1.0, legend=True):
    """Draw the 100% stacked occupancy stream; return the barrier-top y per session."""
    x = np.array(sessions, dtype=float)
    order = _STREAM_ORDER
    ys = np.vstack([occ[st] for st in order])
    ax.stackplot(
        x, *ys,
        colors=[_tint(_VAAMR_COLORS[st]) for st in order],
        labels=[_stage_name(framework, st) for st in order],
        alpha=alpha, edgecolor='white', linewidth=0.8,
    )
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 1)
    ax.set_xticks(sessions)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    # Barrier top = cumulative occupancy of the maladaptive bands (Vigilance+Avoidance).
    barrier_top = np.zeros_like(x)
    for st in _BARRIER_BELOW:
        barrier_top = barrier_top + occ[st]
    return barrier_top


# ── Figure 1 — THE RE-HABITUATION ARC ──────────────────────────────────────

def plot_rehabituation_arc(df, framework, output_dir) -> str:
    stage_ids = sorted(framework.keys())
    eff = _paths.efficacy_dir(output_dir)

    group_csv = os.path.join(eff, 'group_progression_trajectory.csv')
    pso_csv = os.path.join(eff, 'participant_session_outcomes.csv')
    barrier_csv = os.path.join(eff, 'barrier_crossing.csv')
    summary_json = os.path.join(eff, 'efficacy_summary.json')
    if not os.path.isfile(group_csv):
        print("  Warning: Fig1 skipped — group_progression_trajectory.csv missing")
        return ''

    group = pd.read_csv(group_csv).sort_values('session_number')
    pso = pd.read_csv(pso_csv) if os.path.isfile(pso_csv) else pd.DataFrame()
    barrier = pd.read_csv(barrier_csv) if os.path.isfile(barrier_csv) else pd.DataFrame()
    summary = json.load(open(summary_json)) if os.path.isfile(summary_json) else {}

    sessions, occ = _occupancy_by_session(df, stage_ids)
    if not sessions:
        print("  Warning: Fig1 skipped — no occupancy data")
        return ''

    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.patch.set_facecolor(_BG)
    _apply_axis_style(ax)

    barrier_top = _draw_occupancy_stream(ax, sessions, occ, framework)
    ax.set_xlabel('Session number', fontsize=12, color=_INK)
    ax.set_ylabel('Stage occupancy (share of coded participant language)',
                  fontsize=11, color=_INK)

    # The barrier line — bold dashed in a distinct maroon so it reads clearly
    # apart from the black group-E[stage] overlay line.
    _BARRIER_C = '#8B0000'
    xs = np.array(sessions, dtype=float)
    ax.plot(xs, barrier_top, color=_BARRIER_C, lw=3.0, ls=(0, (6, 3)), zorder=6)
    # Label the barrier near a readable point (mid-left, slightly above the line).
    li = max(0, len(xs) // 4)
    ax.annotate(
        'Avoidance → Attention-Regulation barrier',
        xy=(xs[li], barrier_top[li]),
        xytext=(xs[li], min(0.90, barrier_top[li] + 0.18)),
        fontsize=10.5, fontweight='bold', color=_BARRIER_C,
        ha='left', va='bottom',
        arrowprops=dict(arrowstyle='-', color=_BARRIER_C, lw=1.0),
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=_BARRIER_C, alpha=0.88),
        zorder=7,
    )
    # Crossing annotation computed from barrier_crossing.csv.
    cross_txt = ''
    if not barrier.empty and 'crossed_to_attention_regulation' in barrier.columns:
        crossed = int(barrier['crossed_to_attention_regulation'].sum())
        total = int(len(barrier))
        fp = (barrier.loc[barrier['crossed_to_attention_regulation'],
                          'first_passage_session_index'].dropna() + 1)
        med = int(fp.median()) if len(fp) else None
        cross_txt = f'{crossed}/{total} participants crossed'
        if med is not None:
            cross_txt += f' (median: {_ordinal(med)} attended session)'
    if cross_txt:
        ax.annotate(
            cross_txt,
            xy=(xs[-1], barrier_top[-1]),
            xytext=(xs[-1], max(0.05, barrier_top[-1] - 0.10)),
            fontsize=10, fontweight='bold', color='white',
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='#8B0000', ec='none', alpha=0.88),
            zorder=7,
        )

    # Secondary axis: group E[stage] 0–4 with stage-name ticks.
    ax2 = ax.twinx()
    ax2.set_ylim(0, 4)
    ax2.set_yticks(stage_ids)
    ax2.set_yticklabels([_stage_name(framework, s) for s in stage_ids], fontsize=9)
    ax2.set_ylabel('Group mean E[stage]  (progression coordinate)',
                   fontsize=11, color=_INK)
    ax2.tick_params(colors=_MUTED)
    for spine in ('top',):
        ax2.spines[spine].set_visible(False)

    # Faint per-participant spaghetti (E[stage] = progression_coord).
    if not pso.empty and 'progression_coord' in pso.columns:
        for pid, g in pso.groupby('participant_id'):
            g = g.sort_values('session_number')
            if len(g) < 2:
                continue
            ax2.plot(g['session_number'], g['progression_coord'],
                     color='#333333', lw=0.8, alpha=0.22, zorder=3)

    # Strong group E[stage] line + per-session 95% CI error bars. A filled CI
    # band would smear a gray wash across half the colored stack (the CI is
    # wide at this n), so point-wise error bars keep both the stack and the
    # uncertainty legible.
    gx = group['session_number'].to_numpy(dtype=float)
    gm = group['mean'].to_numpy(dtype=float)
    glo = group['ci_lo'].to_numpy(dtype=float)
    ghi = group['ci_hi'].to_numpy(dtype=float)
    ax2.plot(gx, gm, '-', color='white', lw=5.5, zorder=5, solid_capstyle='round')
    ax2.errorbar(gx, gm, yerr=np.vstack([gm - glo, ghi - gm]),
                 fmt='-o', color='#0B0B0B', lw=3.0, ms=6, zorder=6,
                 ecolor='#0B0B0B', elinewidth=1.4, capsize=4, capthick=1.4,
                 label='Group mean E[stage] ± 95% participant-bootstrap CI')

    # Annotation box (top-left): the headline statistics, pulled from summary.
    mk = summary.get('mk_adaptive_occupancy', {}) or {}
    tr = summary.get('trend_interval_sensitivity', {}) or {}
    lines = []
    if mk.get('tau') is not None:
        lines.append(
            'Adaptive-stage occupancy: Mann–Kendall '
            f"τ={fmt_signed(mk.get('tau'), 2)}, {fmt_p(mk.get('p_value'))} "
            '(primary, ordinal-safe)')
    if tr.get('slope') is not None:
        lines.append(
            'Group E[stage] slope '
            f"{fmt_est_ci(tr.get('slope'), tr.get('ci_lo'), tr.get('ci_hi'), nd=3)}"
            '/session (sensitivity)')
    if lines:
        ax.text(0.012, 0.975, '\n'.join(lines),
                transform=ax.transAxes, fontsize=10, va='top', ha='left',
                color=_INK,
                bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='#999999',
                          alpha=0.92), zorder=8)

    # Legend (stages + group line + barrier) — placed below to avoid collisions.
    from matplotlib.lines import Line2D
    stage_handles = [Patch(facecolor=_VAAMR_COLORS[s], alpha=0.62,
                           label=_stage_name(framework, s)) for s in _STREAM_ORDER]
    line_handles, line_labels = ax2.get_legend_handles_labels()
    barrier_handle = Line2D([0], [0], color='#8B0000', lw=3.0, ls=(0, (6, 3)),
                            label='Avoidance barrier (top of maladaptive bands)')
    handles = stage_handles + line_handles + [barrier_handle]
    labels = ([_stage_name(framework, s) for s in _STREAM_ORDER] + line_labels +
              ['Avoidance barrier (top of maladaptive bands)'])
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.10),
              ncol=4, fontsize=9, frameon=False)

    fig.suptitle('The Re-habituation Arc — therapeutic progression in coded participant language',
                 fontsize=15, fontweight='bold', color=_INK, y=0.985)
    ax.set_title(
        'MoveMORE Cohorts 1–3 (N=20) · VAAMR operationalization of the published '
        'VA-MR model (Wexler, Balsamo et al. 2026)',
        fontsize=10.5, color=_MUTED, pad=10)

    fig.text(0.5, 0.005,
             'Single-arm descriptive trend in coded language; not an efficacy claim.  '
             'Methods: 08_methods.txt [M4][M5][M7]',
             ha='center', va='bottom', fontsize=8.5, color=_MUTED, style='italic')

    fig.subplots_adjust(left=0.07, right=0.88, top=0.90, bottom=0.16)
    path = _paths.thesis_figure_path(output_dir, 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=220, facecolor=_BG)
    plt.close(fig)
    return path


# ── transition counting (within-session) for Fig 2 top ─────────────────────

def _within_session_transitions(df: pd.DataFrame, stage_ids) -> np.ndarray:
    """Counts[from, to] of consecutive within-session participant stage transitions."""
    n = len(stage_ids)
    idx = {s: i for i, s in enumerate(stage_ids)}
    counts = np.zeros((n, n), dtype=float)
    sort_col = 'segment_index' if 'segment_index' in df.columns else 'start_time_ms'
    for (_pid, _sid), g in df.groupby(['participant_id', 'session_id']):
        seq = [int(v) for v in g.sort_values(sort_col)['final_label'].tolist()
               if pd.notna(v) and int(v) in idx]
        for a, b in zip(seq[:-1], seq[1:]):
            counts[idx[a], idx[b]] += 1
    return counts


# ── Figure 2 — DYADIC MECHANISM MAP ────────────────────────────────────────

def _curved_arrow(ax, x0, x1, y, width, color, above=True):
    """Draw a curved arrow from x0→x1 bowing above/below the node row."""
    rad = 0.45 if above else -0.45
    arr = FancyArrowPatch(
        (x0, y), (x1, y),
        connectionstyle=f'arc3,rad={rad}',
        arrowstyle='-|>', mutation_scale=12 + width * 1.5,
        lw=width, color=color, alpha=0.7, zorder=4,
    )
    ax.add_patch(arr)


def plot_dyadic_mechanism(df, framework, output_dir) -> str:
    stage_ids = sorted(framework.keys())
    mech_csv = os.path.join(_paths.mechanism_dir(output_dir), 'mechanism_delta_progression.csv')
    if not os.path.isfile(mech_csv):
        print("  Warning: Fig2 skipped — mechanism_delta_progression.csv missing")
        return ''
    mdf = pd.read_csv(mech_csv)
    mdf = mdf[mdf['grouping'] == 'purer'].copy()
    if mdf.empty:
        print("  Warning: Fig2 skipped — no PURER mechanism rows")
        return ''

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [1.0, 1.25]})
    fig.patch.set_facecolor(_BG)

    # ---- TOP: arc nodes + transition arrows -------------------------------
    _apply_axis_style(ax_top)
    ax_top.spines['left'].set_visible(False)
    ax_top.spines['bottom'].set_visible(False)
    counts = _within_session_transitions(df, stage_ids)
    node_x = {s: i for i, s in enumerate(stage_ids)}
    node_y = 0.0
    maxc = counts.max() if counts.size and counts.max() > 0 else 1.0

    # Barrier band between Avoidance (1) and Attention Regulation (2).
    bx = (node_x[1] + node_x[2]) / 2.0
    ax_top.axvspan(bx - 0.07, bx + 0.07, ymin=0.02, ymax=0.98,
                   color='#111111', alpha=0.10, zorder=1)
    ax_top.text(bx, 1.18, 'barrier', ha='center', va='bottom',
                fontsize=9, style='italic', color='#111111')

    # Arrows (forward above, backward below), width ∝ transition count.
    for i, s_from in enumerate(stage_ids):
        for j, s_to in enumerate(stage_ids):
            c = counts[i, j]
            if c <= 0 or s_from == s_to:
                continue
            w = 0.6 + 5.0 * (c / maxc)
            forward = s_to > s_from
            color = '#2E7D32' if forward else '#C62828'
            _curved_arrow(ax_top, node_x[s_from], node_x[s_to], node_y, w, color,
                          above=forward)

    # Nodes on top of arrows.
    for s in stage_ids:
        ax_top.add_patch(Circle((node_x[s], node_y), 0.16, color=_VAAMR_COLORS[s],
                                zorder=6, ec='white', lw=1.5))
        ax_top.text(node_x[s], node_y - 0.42, _stage_name(framework, s),
                    ha='center', va='top', fontsize=9.5, fontweight='bold',
                    color=_INK, zorder=7)

    # Annotate 2-3 strongest forward + backward mechanism cells (by |Δ|, n≥4).
    sig = mdf[mdf['n'] >= 4].copy()
    sig['absd'] = sig['mean_delta_prog'].abs()
    fwd = sig[sig['mean_delta_prog'] > 0].sort_values('absd', ascending=False).head(2)
    bwd = sig[sig['mean_delta_prog'] < 0].sort_values('absd', ascending=False).head(2)
    note_y_f = 0.62
    for _, r in fwd.iterrows():
        move = str(r['behavior']).split('(')[0]
        ax_top.text(0.5, note_y_f,
                    f"▲ {move} at {_stage_name(framework, int(r['from_stage']))}: "
                    f"Δ={fmt_signed(r['mean_delta_prog'], 2)}",
                    transform=ax_top.transData, ha='center', fontsize=9,
                    color='#2E7D32', fontweight='bold')
        note_y_f += 0.30
    note_y_b = -0.95
    for _, r in bwd.iterrows():
        move = str(r['behavior']).split('(')[0]
        ax_top.text(0.5, note_y_b,
                    f"▼ {move} at {_stage_name(framework, int(r['from_stage']))}: "
                    f"Δ={fmt_signed(r['mean_delta_prog'], 2)}",
                    transform=ax_top.transData, ha='center', fontsize=9,
                    color='#C62828', fontweight='bold')
        note_y_b -= 0.30

    ax_top.set_xlim(-0.6, len(stage_ids) - 0.4)
    ax_top.set_ylim(-1.9, 1.7)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_title('Within-session VAAMR transitions (arrow width ∝ count)  +  '
                     'strongest therapist-move Δprogression cells',
                     fontsize=11, fontweight='bold', color=_INK, pad=6)
    ax_top.legend(handles=[
        Patch(facecolor='#2E7D32', alpha=0.7, label='forward (advance)'),
        Patch(facecolor='#C62828', alpha=0.7, label='backward (regress)'),
    ], loc='upper right', fontsize=8.5, frameon=False)

    # ---- BOTTOM: forest of Δprogression per (FROM-stage × move), n≥4 ------
    _apply_axis_style(ax_bot)
    forest = mdf[(mdf['n'] >= 4) & mdf['ci_lo'].notna() & mdf['ci_hi'].notna()].copy()
    forest = forest.sort_values('mean_delta_prog').reset_index(drop=True)
    if forest.empty:
        ax_bot.text(0.5, 0.5, 'No mechanism cells with n≥4 and a CI',
                    ha='center', va='center', transform=ax_bot.transAxes,
                    color=_MUTED, fontsize=11)
    else:
        ys = np.arange(len(forest))
        for y, (_, r) in zip(ys, forest.iterrows()):
            color = _VAAMR_COLORS.get(int(r['from_stage']), '#666666')
            ax_bot.plot([r['ci_lo'], r['ci_hi']], [y, y], color=color, lw=2.2, alpha=0.85)
            ax_bot.plot(r['mean_delta_prog'], y, 'o', color=color, ms=7,
                        mec='white', mew=1.0, zorder=5)
        labels = []
        for _, r in forest.iterrows():
            move = str(r['behavior']).split('(')[0]
            mark = ' ★' if bool(r.get('fdr_significant', False)) else ''
            labels.append(f"{move} @ {_stage_name(framework, int(r['from_stage']))} "
                          f"(n={int(r['n'])}){mark}")
        ax_bot.axvline(0, color='#333333', lw=1.2, ls='--', zorder=2)
        ax_bot.set_yticks(ys)
        ax_bot.set_yticklabels(labels, fontsize=8.5)
        ax_bot.set_ylim(-0.7, len(forest) - 0.3)
        ax_bot.set_xlabel('Δprogression  (E[stage]ₒᵤₜ − E[stage]ᵢₙ)  '
                          '· mean with 95% participant-bootstrap CI', fontsize=10.5)
        ax_bot.set_title('Therapist-move Δprogression by FROM-stage  '
                         '(cells with n≥4; ★ = FDR q<.05; color = FROM-stage)',
                         fontsize=11.5, fontweight='bold', color=_INK, pad=8)
        # FROM-stage color legend.
        from_stages = sorted(forest['from_stage'].unique())
        ax_bot.legend(handles=[
            Patch(facecolor=_VAAMR_COLORS.get(int(s), '#666'),
                  label=_stage_name(framework, int(s))) for s in from_stages],
            loc='lower right', fontsize=8.5, frameon=False, title='FROM-stage',
            title_fontsize=8.5)

    fig.suptitle('Dyadic Mechanism Map — how therapist moves move the participant (directional)',
                 fontsize=14.5, fontweight='bold', color=_INK, y=0.985)
    fig.text(0.5, 0.008,
             'PURER therapist-move labels are computationally generated and NOT yet '
             'human-validated — all panels are directional, hypothesis-generating '
             'evidence [M3][M8].',
             ha='center', va='bottom', fontsize=9, color='#B00020', style='italic',
             fontweight='bold')

    fig.subplots_adjust(left=0.27, right=0.97, top=0.91, bottom=0.075, hspace=0.32)
    path = _paths.thesis_figure_path(output_dir, 2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=220, facecolor=_BG)
    plt.close(fig)
    return path


# ── Figure 3 — FOUR-PANEL DASHBOARD ────────────────────────────────────────

def _panel_arc_compact(ax, df, framework, output_dir):
    stage_ids = sorted(framework.keys())
    sessions, occ = _occupancy_by_session(df, stage_ids)
    if not sessions:
        ax.text(0.5, 0.5, 'no occupancy data', ha='center', va='center',
                transform=ax.transAxes, color=_MUTED)
        return
    _apply_axis_style(ax)
    barrier_top = _draw_occupancy_stream(ax, sessions, occ, framework, legend=False)
    xs = np.array(sessions, dtype=float)
    ax.plot(xs, barrier_top, color='#111111', lw=2.0, ls=(0, (5, 3)), zorder=6)
    ax.set_xlabel('Session number', fontsize=10)
    ax.set_ylabel('Stage occupancy', fontsize=10)

    group_csv = os.path.join(_paths.efficacy_dir(output_dir),
                             'group_progression_trajectory.csv')
    if os.path.isfile(group_csv):
        group = pd.read_csv(group_csv).sort_values('session_number')
        ax2 = ax.twinx()
        ax2.set_ylim(0, 4)
        ax2.set_yticks(stage_ids)
        ax2.set_yticklabels([str(s) for s in stage_ids], fontsize=8)
        ax2.set_ylabel('E[stage]', fontsize=9)
        ax2.plot(group['session_number'], group['mean'], '-o', color='#0B0B0B',
                 lw=2.4, ms=4, zorder=5)
        ax2.spines['top'].set_visible(False)
        ax2.tick_params(colors=_MUTED, labelsize=8)
    ax.set_title('(A)  Re-habituation arc — occupancy stream + group E[stage]',
                 fontsize=11, fontweight='bold', color=_INK, loc='left')


def _panel_barrier_passage(ax, output_dir):
    barrier_csv = os.path.join(_paths.efficacy_dir(output_dir), 'barrier_crossing.csv')
    if not os.path.isfile(barrier_csv):
        ax.text(0.5, 0.5, 'barrier_crossing.csv missing', ha='center', va='center',
                transform=ax.transAxes, color=_MUTED)
        return
    b = pd.read_csv(barrier_csv)
    total = int(len(b))
    crossed_mask = b['crossed_to_attention_regulation'].astype(bool)
    crossed = int(crossed_mask.sum())
    fp = (b.loc[crossed_mask, 'first_passage_session_index'].dropna() + 1).astype(int)
    _apply_axis_style(ax)
    max_sess = int(fp.max()) if len(fp) else 1
    xs = list(range(1, max(max_sess, 1) + 1))
    cum = [int((fp <= s).sum()) / total * 100 for s in xs]
    # Step curve from session 0 (0%).
    ax.step([0] + xs, [0] + cum, where='post', color='#2E7D32', lw=2.8)
    ax.fill_between([0] + xs, 0, [0] + cum, step='post', color='#2E7D32', alpha=0.12)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(max_sess, 1) + 0.6)
    ax.set_xticks(xs)
    ax.set_xlabel("Participant's k-th attended session (first passage)", fontsize=10)
    ax.set_ylabel('% participants having crossed', fontsize=10)
    med = int(fp.median()) if len(fp) else None
    ax.axhline(crossed / total * 100, color='#888888', ls=':', lw=1.2)
    txt = f'{crossed}/{total} crossed'
    if med is not None:
        txt += f'  ·  median: {_ordinal(med)} attended session'
    ax.text(0.97, 0.06, txt, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, fontweight='bold', color='#2E7D32',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#2E7D32', alpha=0.9))
    ax.set_title('(B)  Avoidance barrier first-passage (cumulative)',
                 fontsize=11, fontweight='bold', color=_INK, loc='left')


def _panel_delta_forest(ax, framework, output_dir):
    mech_csv = os.path.join(_paths.mechanism_dir(output_dir),
                            'mechanism_delta_progression.csv')
    if not os.path.isfile(mech_csv):
        ax.text(0.5, 0.5, 'mechanism CSV missing', ha='center', va='center',
                transform=ax.transAxes, color=_MUTED)
        return
    mdf = pd.read_csv(mech_csv)
    mdf = mdf[(mdf['grouping'] == 'purer') & (mdf['n'] >= 4) &
              mdf['ci_lo'].notna() & mdf['ci_hi'].notna()].copy()
    if mdf.empty:
        ax.text(0.5, 0.5, 'no n≥4 cells', ha='center', va='center',
                transform=ax.transAxes, color=_MUTED)
        return
    mdf['absd'] = mdf['mean_delta_prog'].abs()
    top = mdf.sort_values('absd', ascending=False).head(10).sort_values('mean_delta_prog')
    top = top.reset_index(drop=True)
    _apply_axis_style(ax)
    ys = np.arange(len(top))
    labels = []
    for y, (_, r) in zip(ys, top.iterrows()):
        color = _VAAMR_COLORS.get(int(r['from_stage']), '#666')
        ax.plot([r['ci_lo'], r['ci_hi']], [y, y], color=color, lw=2.0, alpha=0.85)
        ax.plot(r['mean_delta_prog'], y, 'o', color=color, ms=6, mec='white', mew=0.8)
        move = str(r['behavior']).split('(')[0]
        mark = ' ★' if bool(r.get('fdr_significant', False)) else ''
        labels.append(f"{move[:10]} @ {_stage_name(framework, int(r['from_stage']))[:11]}{mark}")
    ax.axvline(0, color='#333333', lw=1.0, ls='--')
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(-0.7, len(top) - 0.3)
    ax.set_xlabel('Δprogression (mean, 95% CI)', fontsize=10)
    ax.set_title('(C)  Top therapist-move Δprogression cells (directional)',
                 fontsize=11, fontweight='bold', color=_INK, loc='left')


def _collect_reliability_stats(output_dir) -> list:
    """Return list of {label, value, lo, hi, group} reliability points from IRR JSON."""
    irr_dir = _paths.irr_dir(output_dir) if hasattr(_paths, 'irr_dir') else \
        os.path.join(output_dir, '04_validation', 'irr')
    path = None
    for cand in ('results.json', 'irr_results.json'):
        p = os.path.join(irr_dir, cand)
        if os.path.isfile(p):
            path = p
            break
    if path is None:
        return []
    d = json.load(open(path))
    pts = []
    # Human↔Human: per-testset Krippendorff α + pairwise Cohen κ.
    hh = d.get('human_human', {}) or {}
    for ts_key in sorted(hh.keys()):
        ts = hh[ts_key]
        prim = ts.get('primary') or {}
        a = prim.get('krippendorff_alpha')
        if a is not None:
            pts.append({'label': f'H↔H α (set {ts_key})', 'value': float(a),
                        'lo': None, 'hi': None, 'group': 'Human↔Human'})
        for pw in (prim.get('pairwise') or []):
            k = pw.get('cohen_kappa')
            if k is not None:
                pts.append({'label': f"H↔H κ {pw.get('rater_a')}/{pw.get('rater_b')} (set {ts_key})",
                            'value': float(k), 'lo': None, 'hi': None,
                            'group': 'Human↔Human'})
    # Human↔LLM consensus + per-model.
    hl = d.get('human_vs_llm', {}) or {}
    if hl.get('cohen_kappa') is not None:
        pts.append({'label': 'H↔LLM consensus κ', 'value': float(hl['cohen_kappa']),
                    'lo': None, 'hi': None, 'group': 'Human↔LLM'})
    for model, mv in (hl.get('per_llm_rater') or {}).items():
        if mv.get('cohen_kappa') is not None:
            short = model.split('/')[-1]
            pts.append({'label': f'H↔LLM κ ({short})', 'value': float(mv['cohen_kappa']),
                        'lo': None, 'hi': None, 'group': 'Human↔LLM'})
    return pts


def _panel_reliability_forest(ax, output_dir):
    pts = _collect_reliability_stats(output_dir)
    _apply_axis_style(ax)
    if not pts:
        ax.text(0.5, 0.5, 'IRR results.json missing', ha='center', va='center',
                transform=ax.transAxes, color=_MUTED)
        ax.set_title('(D)  Inter-rater reliability', fontsize=11,
                     fontweight='bold', color=_INK, loc='left')
        return
    # Landis–Koch bands as shaded vertical regions.
    bands = [(-1.0, 0.0, 'poor', '#F2F2F2'),
             (0.0, 0.20, 'slight', '#EAEAEA'),
             (0.20, 0.40, 'fair', '#DEDEDE'),
             (0.40, 0.60, 'moderate', '#CFCFCF'),
             (0.60, 0.80, 'substantial', '#C0C0C0'),
             (0.80, 1.0, 'almost perfect', '#B2B2B2')]
    for lo, hi, name, col in bands:
        ax.axvspan(lo, hi, color=col, alpha=0.55, zorder=0)
    # Order points: Human↔Human first (ceiling), then Human↔LLM.
    order = {'Human↔Human': 0, 'Human↔LLM': 1}
    pts_sorted = sorted(pts, key=lambda p: (order.get(p['group'], 9), -p['value']))
    ys = np.arange(len(pts_sorted))[::-1]
    group_color = {'Human↔Human': '#0B0B0B', 'Human↔LLM': '#785EF0'}
    for y, p in zip(ys, pts_sorted):
        c = group_color.get(p['group'], '#666')
        if p['lo'] is not None and p['hi'] is not None:
            ax.plot([p['lo'], p['hi']], [y, y], color=c, lw=2.0, alpha=0.8)
        ax.plot(p['value'], y, 'o', color=c, ms=6, mec='white', mew=0.8, zorder=5)
    n = len(pts_sorted)
    top_y = n - 0.4   # headroom row reserved for band labels
    ax.set_yticks(ys)
    ax.set_yticklabels([p['label'] for p in pts_sorted], fontsize=7.2)
    ax.set_ylim(-1.2, n + 0.4)
    ax.set_xlim(-0.05, 1.0)
    ax.set_xlabel('Agreement coefficient (κ / α)', fontsize=10)
    # Band labels along the top, inside the reserved headroom row.
    for lo, hi, name, col in bands:
        if hi <= -0.001:
            continue
        ax.text((max(lo, -0.05) + hi) / 2, n + 0.05, name,
                ha='center', va='bottom', fontsize=6.6, color=_MUTED, rotation=0,
                clip_on=True)
    # Human↔human ceiling annotation.
    hh_vals = [p['value'] for p in pts if p['group'] == 'Human↔Human']
    if hh_vals:
        ceil = float(np.median(hh_vals))
        ax.axvline(ceil, color='#0B0B0B', ls=':', lw=1.2, alpha=0.6)
        ax.text(0.475, -1.05, 'human↔human band = ceiling for any machine rater',
                ha='center', va='bottom', fontsize=7.4, color='#0B0B0B',
                style='italic')
    ax.legend(handles=[
        Patch(facecolor='#0B0B0B', label='Human↔Human'),
        Patch(facecolor='#785EF0', label='Human↔LLM'),
    ], loc='upper left', fontsize=8, frameon=True, framealpha=0.9)
    ax.set_title('(D)  Inter-rater reliability vs Landis–Koch bands',
                 fontsize=11, fontweight='bold', color=_INK, loc='left')


def plot_dashboard(df, framework, output_dir) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(_BG)
    try:
        _panel_arc_compact(axes[0][0], df, framework, output_dir)
    except Exception as e:
        print(f"  Warning: dashboard panel A failed: {e}")
    try:
        _panel_barrier_passage(axes[0][1], output_dir)
    except Exception as e:
        print(f"  Warning: dashboard panel B failed: {e}")
    try:
        _panel_delta_forest(axes[1][0], framework, output_dir)
    except Exception as e:
        print(f"  Warning: dashboard panel C failed: {e}")
    try:
        _panel_reliability_forest(axes[1][1], output_dir)
    except Exception as e:
        print(f"  Warning: dashboard panel D failed: {e}")

    fig.suptitle('MoveMORE computational phenomenology — outcomes, mechanism, '
                 'reliability at a glance',
                 fontsize=15, fontweight='bold', color=_INK, y=0.995)
    fig.text(0.5, 0.005,
             'Single-arm descriptive (B, A) + DIRECTIONAL PURER mechanism (C) + '
             'reliability ceiling (D).  Methods: 08_methods.txt.',
             ha='center', va='bottom', fontsize=8.5, color=_MUTED, style='italic')

    fig.subplots_adjust(left=0.18, right=0.95, top=0.93, bottom=0.07,
                        wspace=0.55, hspace=0.30)
    path = _paths.thesis_figure_path(output_dir, 3)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=_BG)
    plt.close(fig)
    return path


# ── public entry point ─────────────────────────────────────────────────────

def generate_thesis_figures(df, df_all, framework, output_dir) -> list:
    """Generate the three flagship thesis figures. Returns list of written paths.

    Each figure is independently wrapped; a failure (or missing input) skips
    only that figure (printed warning) without affecting the others.
    """
    paths = []
    generators = [
        ('Fig1 re-habituation arc', lambda: plot_rehabituation_arc(df, framework, output_dir)),
        ('Fig2 dyadic mechanism map', lambda: plot_dyadic_mechanism(df, framework, output_dir)),
        ('Fig3 dashboard', lambda: plot_dashboard(df, framework, output_dir)),
    ]
    for name, gen in generators:
        try:
            p = gen()
            if p:
                paths.append(p)
        except Exception as e:
            print(f"  Warning: {name} failed: {e}")
            import traceback
            traceback.print_exc()
    return paths
