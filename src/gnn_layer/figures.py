"""
gnn_layer/figures.py
--------------------
Visualizations for the GNN representation-and-discovery layer. Previously the
layer produced only CSVs and text reports; these figures surface its three most
decision-relevant outputs so a human reviewer can see them at a glance:

  • gnn_validation_kappa_by_stage.png — the reliability gate (per VAAMR stage and
    per PURER move κ vs LLM consensus, with the irr_target line). This is the
    safeguard against rare-stage collapse (§5.3) made visible.
  • gnn_motif_influence.png — discovered therapist-language motifs by forward
    influence vs PURER purity; low-purity/high-influence = emergent candidates.
  • gnn_coupling_factors.png — latent coupling factors by correlation with
    subsequent participant forward movement.

All read the CSVs the layer already wrote and degrade gracefully if absent.
matplotlib uses the non-interactive Agg backend.
"""

import os
from typing import List

from process import output_paths as _paths


def _figs_dir(output_dir: str) -> str:
    d = _paths.figures_dir(output_dir)
    os.makedirs(d, exist_ok=True)
    return d


def plot_validation_kappa(output_dir: str, irr_target: float = 0.70):
    """Per-class κ bars for VAAMR & PURER from gnn_validation.csv."""
    import pandas as pd
    csv = os.path.join(_paths.gnn_data_dir(output_dir), 'gnn_validation.csv')
    if not os.path.isfile(csv):
        return None
    df = pd.read_csv(csv)
    per = df[df['scope'] == 'per_class'].copy()
    if per.empty:
        return None
    per['kappa'] = pd.to_numeric(per['kappa'], errors='coerce')
    per = per.dropna(subset=['kappa'])
    if per.empty:
        return None

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    frameworks = [fw for fw in ('vaamr', 'purer') if (per['framework'] == fw).any()]
    fig, axes = plt.subplots(1, len(frameworks), figsize=(6 * len(frameworks), 4.2), squeeze=False)
    for ax, fw in zip(axes[0], frameworks):
        sub = per[per['framework'] == fw]
        labels = [str(x) for x in sub['class_name']]
        vals = list(sub['kappa'])
        colors = ['#d62728' if v < irr_target else '#2ca02c' for v in vals]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.axhline(irr_target, color='black', ls='--', lw=1, label=f'irr_target={irr_target:.2f}')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
        ax.set_ylim(min(0, min(vals + [0])), 1.0)
        ax.set_ylabel("Cohen's κ vs LLM consensus")
        ax.set_title(f'{fw.upper()} per-class reliability')
        ax.legend(fontsize=8)
    fig.suptitle('GNN reliability gate — red = below target (watch rare-stage collapse)', fontsize=10)
    fig.tight_layout()
    path = os.path.join(_figs_dir(output_dir), 'gnn_validation_kappa_by_stage.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_motif_influence(output_dir: str):
    """Scatter of motif influence vs PURER purity; emergent motifs highlighted."""
    import pandas as pd
    csv = os.path.join(_paths.gnn_data_dir(output_dir), 'cue_motifs.csv')
    if not os.path.isfile(csv):
        return None
    df = pd.read_csv(csv)
    if df.empty or 'influence' not in df.columns:
        return None

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    inf = pd.to_numeric(df['influence'], errors='coerce').fillna(0)
    pur = pd.to_numeric(df.get('purer_purity', 0), errors='coerce').fillna(0)
    emergent = (inf >= 1.2) & (pur < 0.5)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pur[~emergent], inf[~emergent], c='#1f77b4', label='explained by PURER', alpha=0.7)
    ax.scatter(pur[emergent], inf[emergent], c='#d62728', label='emergent (candidate new construct)', alpha=0.9)
    for _, r in df[emergent].iterrows():
        ax.annotate(f"motif {int(r['motif_id'])}", (r.get('purer_purity', 0), r.get('influence', 0)),
                    fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.axhline(1.0, color='gray', ls=':', lw=1)
    ax.set_xlabel('PURER purity (1 = fully explained by an existing move)')
    ax.set_ylabel('forward-transition influence')
    ax.set_title('GNN cue motifs — influence vs explainability')
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(_figs_dir(output_dir), 'gnn_motif_influence.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_coupling_factors(output_dir: str):
    """Bar chart of latent coupling-factor correlation with forward movement."""
    import pandas as pd
    csv = os.path.join(_paths.gnn_data_dir(output_dir), 'coupling_factors.csv')
    if not os.path.isfile(csv):
        return None
    df = pd.read_csv(csv)
    if df.empty or 'forward_corr' not in df.columns:
        return None
    df = df.copy()
    df['forward_corr'] = pd.to_numeric(df['forward_corr'], errors='coerce')
    df = df.dropna(subset=['forward_corr'])
    if df.empty:
        return None

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = [str(r.get('nearest_cf_ic') or f"factor {int(r['factor'])}") for _, r in df.iterrows()]
    vals = list(df['forward_corr'])
    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in vals]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(vals) + 1)))
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('correlation with subsequent participant forward movement')
    ax.set_title('GNN latent coupling factors (alliance-like structure, discovered)')
    ax.invert_yaxis()
    fig.tight_layout()
    path = os.path.join(_figs_dir(output_dir), 'gnn_coupling_factors.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return path


def generate_gnn_figures(output_dir: str, irr_target: float = 0.70) -> List[str]:
    """Render every GNN figure that has data. Each plotter is independently guarded."""
    paths = []
    for fn in (lambda: plot_validation_kappa(output_dir, irr_target),
               lambda: plot_motif_influence(output_dir),
               lambda: plot_coupling_factors(output_dir)):
        try:
            p = fn()
            if p:
                paths.append(p)
        except Exception as e:
            print(f"  Warning: a GNN figure failed: {e}")
    return paths
