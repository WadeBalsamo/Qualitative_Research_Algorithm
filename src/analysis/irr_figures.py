"""
analysis/irr_figures.py
-----------------------
Matplotlib (Agg) figures for the inter-rater-reliability analysis.

Written into ``04_validation/irr/`` alongside the data artifacts:
  * irr_confusion_human_vs_llm.png   — human-consensus vs LLM confusion matrix
  * irr_confusion_human_vs_gnn.png   — human-consensus vs GNN confusion matrix
  * irr_rater_agreement_heatmap.png  — pairwise Cohen's κ between raters per test-set

Written into ``06_reports/01_reliability/`` (the trust-instruments tier):
  * reliability_forest.png           — every agreement coefficient on one forest plot,
                                       with Landis–Koch bands + the human-ceiling marker

All figures degrade gracefully: a family with no usable data is skipped.
"""

import os
from typing import List, Optional

from process import output_paths as _paths


def _out_dir(output_dir: str) -> str:
    d = _paths.irr_validation_dir(output_dir)
    os.makedirs(d, exist_ok=True)
    return d


def _plot_confusion(results: dict, key: str, title: str, fname: str,
                    output_dir: str) -> Optional[str]:
    block = results.get(key, {})
    conf = block.get('confusion') or {}
    matrix = conf.get('matrix')
    names = conf.get('label_names')
    if not matrix or not names:
        return None

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(1.2 + 0.8 * len(names), 1.2 + 0.8 * len(names)))
    im = ax.imshow(arr, cmap='Blues')
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('machine')
    ax.set_ylabel('human consensus')
    k = block.get('cohen_kappa')
    ksuf = f"  (κ={k:+.3f}, n={block.get('n', 0)})" if k is not None else ""
    ax.set_title(title + ksuf, fontsize=10)
    thresh = arr.max() / 2 if arr.size else 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, int(arr[i, j]), ha='center', va='center',
                    color='white' if arr[i, j] > thresh else 'black', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = os.path.join(_out_dir(output_dir), fname)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_rater_heatmap(results: dict, output_dir: str) -> Optional[str]:
    """One subplot per test-set: symmetric pairwise Cohen's κ matrix (primary)."""
    hh = results.get('human_human', {})
    panels = []
    for ws in sorted(hh, key=lambda x: int(x)):
        pri = hh[ws].get('primary')
        if pri and pri.get('pairwise'):
            panels.append((ws, hh[ws].get('raters', []), pri['pairwise']))
    if not panels:
        return None

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4),
                             squeeze=False)
    for ax, (ws, raters, pairwise) in zip(axes[0], panels):
        n = len(raters)
        idx = {r: i for i, r in enumerate(raters)}
        mat = np.full((n, n), np.nan)
        for i in range(n):
            mat[i, i] = 1.0
        for pr in pairwise:
            a, b, k = pr['rater_a'], pr['rater_b'], pr['cohen_kappa']
            if a in idx and b in idx and k is not None:
                mat[idx[a], idx[b]] = k
                mat[idx[b], idx[a]] = k
        im = ax.imshow(mat, cmap='RdYlGn', vmin=-0.2, vmax=1.0)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(raters, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(raters, fontsize=8)
        ax.set_title(f"test-set {ws}", fontsize=10)
        for i in range(n):
            for j in range(n):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha='center', va='center', fontsize=7)
    fig.suptitle("Pairwise Cohen's κ between raters (primary VAAMR)", fontsize=11)
    path = os.path.join(_out_dir(output_dir), 'irr_rater_agreement_heatmap.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return path


# Landis & Koch (1977) bands as (lo, hi, label) for the shaded background.
_LK_BANDS = [
    (-0.20, 0.20, 'slight'),
    (0.20, 0.40, 'fair'),
    (0.40, 0.60, 'moderate'),
    (0.60, 0.80, 'substantial'),
    (0.80, 1.00, 'almost\nperfect'),
]


def _forest_rows(results: dict):
    """Collect every agreement coefficient as (group, label, value, lo, hi) tuples.

    Returns the rows in bottom-to-top plot order grouped into:
      1. Human↔Human per-test-set Krippendorff α (+ bootstrap CI)
      2. pairwise human Cohen κ
      3. Human↔LLM consensus (overall + per-test-set) (+ bootstrap CI)
      4. per-model Human↔LLM κ (+ bootstrap CI)
    """
    rows = []  # (group_title, label, value, lo, hi)
    hh = results.get('human_human', {})

    # Group 1 — Human↔Human α.
    for ws in sorted(hh, key=lambda x: int(x)):
        pri = hh[ws].get('primary')
        if not pri or pri.get('krippendorff_alpha') is None:
            continue
        ci = pri.get('alpha_ci') or {}
        rows.append(('Human ↔ Human (Krippendorff α)',
                     f"test-set {ws}", pri['krippendorff_alpha'],
                     ci.get('lo'), ci.get('hi')))

    # Group 2 — pairwise human κ.
    for ws in sorted(hh, key=lambda x: int(x)):
        pri = hh[ws].get('primary') or {}
        for pr in pri.get('pairwise', []):
            if pr.get('cohen_kappa') is None:
                continue
            rows.append(('Pairwise human (Cohen κ)',
                         f"T{ws}: {pr['rater_a']}↔{pr['rater_b']}",
                         pr['cohen_kappa'], None, None))

    # Group 3 — Human↔LLM consensus.
    llm = results.get('human_vs_llm', {})
    if llm.get('n') and llm.get('cohen_kappa') is not None:
        ci = llm.get('kappa_ci') or {}
        rows.append(('Human ↔ LLM consensus (Cohen κ)',
                     'overall', llm['cohen_kappa'], ci.get('lo'), ci.get('hi')))
        for ws, sub in sorted((llm.get('per_worksheet') or {}).items(),
                              key=lambda x: int(x[0])):
            if sub.get('cohen_kappa') is None:
                continue
            ci = sub.get('kappa_ci') or {}
            rows.append(('Human ↔ LLM consensus (Cohen κ)',
                         f"test-set {ws}", sub['cohen_kappa'],
                         ci.get('lo'), ci.get('hi')))

    # Group 4 — per-model κ.
    for rid, sub in sorted((llm.get('per_llm_rater') or {}).items(),
                           key=lambda x: -(x[1].get('cohen_kappa') or 0)):
        if sub.get('cohen_kappa') is None:
            continue
        ci = sub.get('kappa_ci') or {}
        rows.append(('Human ↔ LLM per model (Cohen κ)',
                     rid.split('/')[-1], sub['cohen_kappa'],
                     ci.get('lo'), ci.get('hi')))
    return rows


def _human_band(results: dict):
    """(min, max) of per-test-set Human↔Human Krippendorff α — the machine ceiling."""
    hh = results.get('human_human', {})
    a = [v['primary']['krippendorff_alpha'] for v in hh.values()
         if v.get('primary') and v['primary'].get('krippendorff_alpha') is not None]
    return (min(a), max(a)) if a else None


def plot_reliability_forest(results: dict, out_path: str) -> Optional[str]:
    """Forest plot of every IRR agreement coefficient, with Landis–Koch bands and
    the human↔human ceiling shaded. Writes ``out_path`` (PNG, Agg, dpi 200)."""
    rows = _forest_rows(results)
    if not rows:
        return None

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    # Assign each row a y position; insert a small gap + header between groups.
    groups = []
    for grp, *_ in rows:
        if not groups or groups[-1] != grp:
            groups.append(grp)
    group_order = list(dict.fromkeys(groups))
    group_colors = {
        'Human ↔ Human (Krippendorff α)': '#1f4e8c',
        'Pairwise human (Cohen κ)': '#5b8bd0',
        'Human ↔ LLM consensus (Cohen κ)': '#b8460f',
        'Human ↔ LLM per model (Cohen κ)': '#d98a5a',
    }

    # Build y-layout top-to-bottom: first group at top.
    ylabels, yvals, ylos, yhis, ycolors = [], [], [], [], []
    yticks, ytick_labels = [], []
    y = 0.0
    for grp in group_order:
        grp_rows = [r for r in rows if r[0] == grp]
        for (_g, label, val, lo, hi) in grp_rows:
            yvals.append(y)
            ylabels.append(val)
            ylos.append(lo)
            yhis.append(hi)
            ycolors.append(group_colors.get(grp, '#444444'))
            yticks.append(y)
            ytick_labels.append(label)
            y -= 1.0
        y -= 1.2  # gap between groups

    n_rows = len(yvals)
    fig_h = max(4.5, 0.42 * n_rows + 2.2)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))

    top_y = max(yvals) + 0.9
    # Shaded Landis–Koch bands.
    band_greys = ['#f2f2f2', '#e6e6e6', '#dadada', '#cfcfcf', '#c4c4c4']
    for (lo, hi, lab), grey in zip(_LK_BANDS, band_greys):
        ax.axvspan(lo, hi, color=grey, zorder=0)
        ax.text((lo + hi) / 2, top_y, lab, ha='center', va='bottom',
                fontsize=7.5, color='#666666', zorder=1)

    # Human↔human ceiling band (vertical shaded span over the α range).
    band = _human_band(results)
    if band:
        ax.axvspan(band[0], band[1], color='#ffd24d', alpha=0.35, zorder=1)
        ax.axvline(band[0], color='#c79100', lw=0.8, ls='--', zorder=2)
        ax.axvline(band[1], color='#c79100', lw=0.8, ls='--', zorder=2)
        ax.annotate('human↔human band\n(ceiling for machine raters)',
                    xy=((band[0] + band[1]) / 2, min(yvals) - 0.4),
                    xytext=(0.62, min(yvals) - 0.7),
                    fontsize=7.5, color='#8a6500', va='center', ha='left',
                    arrowprops=dict(arrowstyle='->', color='#c79100', lw=1.0))

    # Zero reference.
    ax.axvline(0.0, color='#999999', lw=0.8, zorder=2)

    # Points + CI whiskers.
    for yy, val, lo, hi, col in zip(yvals, ylabels, ylos, yhis, ycolors):
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [yy, yy], color=col, lw=1.6, zorder=4,
                    solid_capstyle='round')
            ax.plot([lo, lo], [yy - 0.12, yy + 0.12], color=col, lw=1.2, zorder=4)
            ax.plot([hi, hi], [yy - 0.12, yy + 0.12], color=col, lw=1.2, zorder=4)
        ax.scatter([val], [yy], color=col, s=42, zorder=5, edgecolor='white',
                   linewidth=0.6)
        ax.text(val, yy + 0.28, f"{val:+.2f}", ha='center', va='bottom',
                fontsize=7, color=col, zorder=6)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=8)
    ax.set_ylim(min(yvals) - 1.3, max(yvals) + 1.7)
    ax.set_xlim(-0.25, 1.0)
    ax.set_xlabel("agreement coefficient (Cohen κ / Krippendorff α)", fontsize=9)
    ax.set_title("VAAMR inter-rater reliability — forest of agreement coefficients",
                 fontsize=11, fontweight='bold')

    # Legend for groups.
    handles = [Patch(color=group_colors.get(g, '#444'), label=g) for g in group_order]
    handles.append(Patch(color='#ffd24d', alpha=0.35,
                         label='human↔human ceiling band'))
    ax.legend(handles=handles, loc='lower right', fontsize=7.2, framealpha=0.9)

    ax.grid(axis='x', color='white', lw=0.6, zorder=1)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path


def write_irr_figures(results: dict, output_dir: str) -> List[str]:
    """Generate every IRR figure that has data; return the written paths."""
    paths = []
    for key, title, fname in (
        ('human_vs_llm', 'Human consensus vs LLM', 'irr_confusion_human_vs_llm.png'),
        ('human_vs_gnn', 'Human consensus vs GNN', 'irr_confusion_human_vs_gnn.png'),
    ):
        p = _plot_confusion(results, key, title, fname, output_dir)
        if p:
            paths.append(p)
    p = _plot_rater_heatmap(results, output_dir)
    if p:
        paths.append(p)
    # Reliability forest — lives in the trust-instruments report tier (01_reliability/).
    forest_path = os.path.join(_paths.reports_reliability_dir(output_dir),
                               'reliability_forest.png')
    p = plot_reliability_forest(results, forest_path)
    if p:
        paths.append(p)
    return paths
