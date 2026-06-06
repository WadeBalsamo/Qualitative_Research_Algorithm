"""
analysis/irr_figures.py
-----------------------
Matplotlib (Agg) figures for the inter-rater-reliability analysis, written into
``04_validation/irr/`` alongside the data artifacts.

  * irr_confusion_human_vs_llm.png   — human-consensus vs LLM confusion matrix
  * irr_confusion_human_vs_gnn.png   — human-consensus vs GNN confusion matrix
  * irr_rater_agreement_heatmap.png  — pairwise Cohen's κ between raters per test-set

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
    return paths
