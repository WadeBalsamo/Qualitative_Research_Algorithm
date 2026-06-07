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


def plot_discriminant_validity(output_dir: str):
    """H6: PCA-2D of segments coloured by VAAMR stage (no stage separation along content axes)
    beside the recoverability-vs-similarity κ bars with participant-clustered CIs."""
    import pandas as pd
    gnn = _paths.gnn_data_dir(output_dir)
    coords_csv = os.path.join(gnn, 'discriminant_pca_coords.csv')
    arms_csv = os.path.join(gnn, 'discriminant_validity.csv')
    if not os.path.isfile(coords_csv) and not os.path.isfile(arms_csv):
        return None

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: PCA scatter coloured by stage (the "content axes don't separate stage" panel).
    ax = axes[0]
    if os.path.isfile(coords_csv):
        c = pd.read_csv(coords_csv)
        names = {0: 'Vigilance', 1: 'Avoidance', 2: 'AttentionReg', 3: 'Metacognition', 4: 'Reappraisal'}
        cmap = plt.get_cmap('viridis', 5)
        for s in sorted(c['stage'].unique()):
            sub = c[c['stage'] == s]
            ax.scatter(sub['pc1'], sub['pc2'], s=14, alpha=0.6, color=cmap(int(s)),
                       label=names.get(int(s), str(s)))
        ax.set_xlabel('content PC1'); ax.set_ylabel('content PC2')
        ax.set_title('Segments in content-PC space (colour = VAAMR stage)\nstages are intermixed → stage ≠ topic')
        ax.legend(fontsize=7, markerscale=1.5)
    else:
        ax.set_visible(False)

    # Right: recoverability vs similarity vs chance, human-axis κ with CIs.
    ax = axes[1]
    if os.path.isfile(arms_csv):
        a = pd.read_csv(arms_csv)
        arms = a[a['section'] == 'arm'].copy() if 'section' in a.columns else a.iloc[0:0]
        pretty = {'H6-probe': 'probe\n(supervised)', 'H6-content': 'content\n(C&S)',
                  'H6-chance-mode': 'chance\n(modal)', 'H6-chance-strat': 'chance\n(strat.)'}
        order = ['H6-probe', 'H6-content', 'H6-chance-mode', 'H6-chance-strat']
        arms = arms[arms['name'].isin(order)]
        arms['__o'] = arms['name'].map({n: i for i, n in enumerate(order)})
        arms = arms.sort_values('__o')
        if not arms.empty:
            ks = pd.to_numeric(arms['human_kappa'], errors='coerce').to_numpy()
            lo = pd.to_numeric(arms['human_lo'], errors='coerce').to_numpy()
            hi = pd.to_numeric(arms['human_hi'], errors='coerce').to_numpy()
            x = np.arange(len(arms))
            yerr = np.vstack([np.clip(ks - lo, 0, None), np.clip(hi - ks, 0, None)])
            colors = ['#2ca02c', '#d62728', '#7f7f7f', '#7f7f7f'][:len(arms)]
            ax.bar(x, ks, color=colors)
            ax.errorbar(x, ks, yerr=yerr, fmt='none', ecolor='black', capsize=4, lw=1)
            ax.axhline(0, color='black', lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([pretty.get(n, n) for n in arms['name']], fontsize=8)
            ax.set_ylabel("Cohen's κ vs human consensus")
            ax.set_title('H6: stage is recoverable by supervision,\nnot by content similarity')
    else:
        ax.set_visible(False)

    fig.suptitle('H6 — VAAMR is developmental, not topical (hypothesis-generating; n≈32)', fontsize=11)
    fig.tight_layout()
    path = os.path.join(_figs_dir(output_dir), 'gnn_discriminant_validity.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_transition_influence(output_dir: str):
    """Dyadic transition model: per-PURER-move learned ΔE[stage] (counterfactual) with
    participant-clustered CIs; thin-support moves flagged (extrapolated)."""
    import pandas as pd
    csv = os.path.join(_paths.gnn_data_dir(output_dir), 'transition_per_move.csv')
    if not os.path.isfile(csv):
        return None
    df = pd.read_csv(csv)
    if df.empty or 'mean_influence' not in df.columns:
        return None

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    df = df.sort_values('mean_influence', ascending=True)
    y = np.arange(len(df))
    vals = pd.to_numeric(df['mean_influence'], errors='coerce').to_numpy()
    lo = pd.to_numeric(df.get('ci_lo'), errors='coerce').to_numpy()
    hi = pd.to_numeric(df.get('ci_hi'), errors='coerce').to_numpy()
    xerr = np.vstack([np.clip(vals - lo, 0, None), np.clip(hi - vals, 0, None)])
    support = pd.to_numeric(df.get('centroid_support', 0), errors='coerce').fillna(0).to_numpy()
    colors = ['#bbbbbb' if s < 10 else ('#2ca02c' if v >= 0 else '#d62728')
              for s, v in zip(support, vals)]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.6 * len(df) + 1)))
    ax.barh(y, vals, color=colors)
    ax.errorbar(vals, y, xerr=xerr, fmt='none', ecolor='black', capsize=4, lw=1)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}{' (thin)' if s < 10 else ''}"
                        for n, s in zip(df['move_name'], support)], fontsize=8)
    ax.set_xlabel('learned ΔE[stage] if cue were this move (vs neutral)')
    ax.set_title('Transition model — learned counterfactual influence per PURER move\n'
                 '(hypothesis-generating; directions not magnitudes; n≈32)')
    fig.tight_layout()
    path = os.path.join(_figs_dir(output_dir), 'gnn_transition_influence.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_confound_localization(output_dir: str):
    """WS3: signed-divergence heatmap (observed − counterfactual) over from_stage × PURER move —
    where responsiveness most distorts the observed mechanism table."""
    import pandas as pd
    csv = os.path.join(_paths.gnn_data_dir(output_dir), 'confound_localization.csv')
    if not os.path.isfile(csv):
        return None
    df = pd.read_csv(csv)
    if df.empty or 'divergence' not in df.columns:
        return None

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    moves = ['Phenomenology', 'Utilization', 'Reframing', 'Education', 'Reinforcement']
    stages = ['Vigilance', 'Avoidance', 'AttentionReg', 'Metacognition', 'Reappraisal']
    M = np.full((len(stages), len(moves)), np.nan)
    for _, r in df.iterrows():
        try:
            si = stages.index(str(r['from_stage_name'])); mi = moves.index(str(r['move_name']))
        except ValueError:
            continue
        M[si, mi] = r['divergence']
    if np.all(np.isnan(M)):
        return None

    vmax = float(np.nanmax(np.abs(M))) or 1.0
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(M, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(moves))); ax.set_xticklabels(moves, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(len(stages))); ax.set_yticklabels(stages, fontsize=8)
    for si in range(len(stages)):
        for mi in range(len(moves)):
            if not np.isnan(M[si, mi]):
                ax.text(mi, si, f"{M[si, mi]:+.2f}", ha='center', va='center', fontsize=7,
                        color='black')
    ax.set_xlabel('therapist PURER move'); ax.set_ylabel('participant FROM stage')
    ax.set_title('Confound localization: observed − counterfactual Δprogression\n'
                 '(red = observed > cue warrants; blue = observed < cue warrants; caveat map, n≈32)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='signed divergence')
    fig.tight_layout()
    path = os.path.join(_figs_dir(output_dir), 'gnn_confound_localization.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return path


def generate_gnn_figures(output_dir: str, irr_target: float = 0.70) -> List[str]:
    """Render every GNN figure that has data. Each plotter is independently guarded."""
    paths = []
    for fn in (lambda: plot_validation_kappa(output_dir, irr_target),
               lambda: plot_motif_influence(output_dir),
               lambda: plot_coupling_factors(output_dir),
               lambda: plot_discriminant_validity(output_dir),
               lambda: plot_transition_influence(output_dir),
               lambda: plot_confound_localization(output_dir)):
        try:
            p = fn()
            if p:
                paths.append(p)
        except Exception as e:
            print(f"  Warning: a GNN figure failed: {e}")
    return paths
