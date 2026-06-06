"""
analysis/reports/superposition_report.py
-----------------------------------------
Human-readable superposition report (06_reports/02_mechanism/superposition.txt).

Surfaces the VAAMR stage-mixture signal the hard reports collapse: the corpus
superposition rate, the stage co-occurrence ("cusp") matrix, how liminality
(mixture entropy) shifts across the program, the Avoidance↔Attention-Regulation
cusp-density trend, and the most-liminal exemplar quotes. All quantities are
directional / hypothesis-generating; hard labels remain the labeler-of-record.
"""

import os

import numpy as np
import pandas as pd

from process import output_paths as _paths
from ..loader import sort_session_ids
from ..superposition import stage_cooccurrence_matrix, dominant_source
from ._formatting import _wrap_quote


_SOURCE_NOTE = {
    'gnn': "GNN geometry (learned 5-way mixture) — the strongest substrate.",
    'ballots': "reconstructed from multi-run LLM ballots — a STABILITY signal "
               "(single model, repeated), not independent evidence.",
    'secondary': "reconstructed from final_label + secondary_stage — a coarse "
                 "two-point fallback.",
    'none': "no usable mixture signal (uniform) — interpret with caution.",
}


def generate_superposition_report(df: pd.DataFrame, framework: dict, output_dir: str) -> str:
    """Write 02_mechanism/superposition.txt. Returns the path (or '' if no mixtures)."""
    if 'mixture' not in df.columns or len(df) == 0:
        return ''

    stage_ids = sorted(framework.keys())
    n_stages = len(stage_ids)
    names = [framework[s].get('short_name', str(s)) for s in stage_ids]

    L = []
    L.append("=" * 78)
    L.append("VAAMR SUPERPOSITION REPORT")
    L.append("=" * 78)
    L.append("")
    L.append("Participant segments express a BLEND of VAAMR stages, not a single one.")
    L.append("This report surfaces that mixture. It is DIRECTIONAL / hypothesis-generating;")
    L.append("the hard final_label remains the labeler-of-record.")
    L.append("")

    source = dominant_source(df)
    L.append(f"Mixture source (dominant): {source} — {_SOURCE_NOTE.get(source, '')}")
    L.append("")

    # ---- Corpus superposition rate -------------------------------------
    liminal_pct = float(df['is_liminal'].mean()) * 100 if 'is_liminal' in df.columns else 0.0
    mean_ent = float(df['mixture_entropy'].mean()) if df['mixture_entropy'].notna().any() else 0.0
    mean_active = float(df['n_active_stages'].mean()) if 'n_active_stages' in df.columns else 0.0
    L.append("-" * 78)
    L.append("1. CORPUS SUPERPOSITION")
    L.append("-" * 78)
    L.append(f"  Segments:                    {len(df)}")
    L.append(f"  Liminal (mixed-stage):       {liminal_pct:.1f}%")
    L.append(f"  Mean mixture entropy:        {mean_ent:.3f}  (0 = pure stage, 1 = uniform)")
    L.append(f"  Mean active stages/segment:  {mean_active:.2f}")
    L.append("")

    # ---- Stage co-occurrence (cusp) matrix -----------------------------
    mat = np.array(stage_cooccurrence_matrix(df, n_stages))
    L.append("-" * 78)
    L.append("2. STAGE CO-OCCURRENCE (CUSP) MATRIX")
    L.append("-" * 78)
    L.append("  Mass that stage pairs co-express (which stages live together at the boundary).")
    header = "           " + "".join(f"{nm[:8]:>10}" for nm in names)
    L.append(header)
    for i, nm in enumerate(names):
        row = f"  {nm[:9]:<9}" + "".join(f"{mat[i, j]:>10.3f}" for j in range(n_stages))
        L.append(row)
    # Highlight the strongest off-diagonal pair.
    off = [(i, j, mat[i, j]) for i in range(n_stages) for j in range(n_stages) if i < j]
    if off:
        i, j, v = max(off, key=lambda t: t[2])
        L.append(f"\n  Strongest cusp: {names[i]} ↔ {names[j]} ({v:.3f}).")
    L.append("")

    # ---- Liminality across the program ---------------------------------
    L.append("-" * 78)
    L.append("3. LIMINALITY ACROSS THE PROGRAM (mean entropy by session)")
    L.append("-" * 78)
    sids = sort_session_ids(df['session_id'].unique().tolist())
    ent_series = []
    for sid in sids:
        e = df[df['session_id'] == sid]['mixture_entropy'].dropna()
        if len(e):
            ent_series.append((sid, float(e.mean())))
    for sid, e in ent_series:
        bar = '#' * int(round(e * 40))
        L.append(f"  {sid:<10} {e:.3f} {bar}")
    if len(ent_series) >= 2:
        trend = ent_series[-1][1] - ent_series[0][1]
        direction = "fell" if trend < -0.02 else ("rose" if trend > 0.02 else "held steady")
        L.append(f"\n  Liminality {direction} across the program ({trend:+.3f}); "
                 "falling liminality is consistent with stage stabilization.")
    L.append("")

    # ---- Avoidance↔Attention-Regulation cusp density -------------------
    L.append("-" * 78)
    L.append("4. AVOIDANCE ↔ ATTENTION-REGULATION CUSP DENSITY (the clinical barrier)")
    L.append("-" * 78)
    cusp_path = os.path.join(_paths.mechanism_dir(output_dir), 'avoidance_cusp_density_by_session.csv')
    if os.path.isfile(cusp_path):
        try:
            cdf = pd.read_csv(cusp_path)
            for _, r in cdf.iterrows():
                bar = '#' * int(round(float(r['cusp_density']) * 40))
                L.append(f"  {str(r['session_id']):<10} {float(r['cusp_density'])*100:5.1f}% {bar}")
        except Exception:
            L.append("  (cusp density unavailable)")
    else:
        L.append("  (run with mechanism analysis enabled to populate this section)")
    L.append("")

    # ---- Most-liminal exemplars ----------------------------------------
    L.append("-" * 78)
    L.append("5. MOST-LIMINAL EXEMPLARS (genuine mixed-stage expression)")
    L.append("-" * 78)
    cand = df[(df.get('word_count', 0) >= 15) & (df.get('word_count', 0) <= 200)] \
        if 'word_count' in df.columns else df
    cand = cand.sort_values('mixture_entropy', ascending=False).head(6)
    for _, r in cand.iterrows():
        mix = np.asarray(r['mixture'], dtype=float)
        order = np.argsort(mix)[::-1]
        a, b = int(order[0]), int(order[1])
        blend = (f"{framework.get(stage_ids[a] if a < len(stage_ids) else a, {}).get('short_name', a)} "
                 f"{mix[a]:.2f} / "
                 f"{framework.get(stage_ids[b] if b < len(stage_ids) else b, {}).get('short_name', b)} "
                 f"{mix[b]:.2f}")
        L.append(f"  [{r.get('session_id', '')}] entropy={float(r['mixture_entropy']):.2f}  {blend}")
        L.append(_wrap_quote(str(r.get('text', ''))[:400], indent=4))
        L.append("")

    rep_dir = _paths.reports_mechanism_dir(output_dir)
    os.makedirs(rep_dir, exist_ok=True)
    path = os.path.join(rep_dir, 'superposition.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path
