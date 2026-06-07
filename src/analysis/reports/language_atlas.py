"""
analysis/reports/language_atlas.py
----------------------------------
Therapeutic language atlas — the readable "key language patterns" deliverable.

The mechanism dossier ranks therapist moves by their association with participant
progression, but a curriculum designer needs to read the actual language. This
renders, for the top-ranked PURER moves / emergent motifs / named
coupling factors, the FROM participant quote → therapist cue text → TO participant
quote, with stage mixtures — so the influential patterns are concrete and
teachable. Emergent motifs (high influence, low PURER purity) are flagged as
candidate new constructs. Directional/associational, like the dossier it draws on.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from process import output_paths as _paths
from ._formatting import _wrap_quote


def _text_lookup(df_all: pd.DataFrame) -> Dict[str, dict]:
    out = {}
    for _, r in df_all.iterrows():
        out[str(r.get('segment_id', ''))] = {
            'text': str(r.get('text', '')),
            'mixture': r.get('mixture'),
            'speaker': r.get('speaker'),
        }
    return out


def _fmt_mixture(mix, framework) -> str:
    if mix is None:
        return ''
    vec = np.asarray(mix, dtype=float)
    order = np.argsort(vec)[::-1]
    stage_ids = sorted(framework.keys())
    parts = []
    for k in order[:2]:
        nm = framework.get(stage_ids[k] if k < len(stage_ids) else int(k), {}).get('short_name', str(k))
        parts.append(f"{nm} {vec[k]:.2f}")
    return ' / '.join(parts)


def _render_block(b, tlu, framework, L, indent='    '):
    """Render one FROM→CUE→TO block with text + mixtures into line list L."""
    fr = tlu.get(b['from_seg_id'], {})
    to = tlu.get(b['to_seg_id'], {})
    cue_text = ' '.join(tlu.get(s, {}).get('text', '') for s in b.get('therapist_seg_ids', [])).strip()
    L.append(f"{indent}FROM [{_fmt_mixture(fr.get('mixture'), framework)}]:")
    L.append(_wrap_quote(fr.get('text', '')[:300], indent=len(indent) + 2))
    L.append(f"{indent}CUE (therapist):")
    L.append(_wrap_quote(cue_text[:400] or '(no therapist speech captured)', indent=len(indent) + 2))
    L.append(f"{indent}TO   [{_fmt_mixture(to.get('mixture'), framework)}]:")
    L.append(_wrap_quote(to.get('text', '')[:300], indent=len(indent) + 2))
    L.append("")


def generate_language_atlas(df, df_all, framework, output_dir) -> Optional[str]:
    """Write 02_mechanism/language_atlas.txt. Returns path, or None if inputs are missing."""
    from gnn_layer.cue_features import build_cue_blocks_with_segments
    from ..mechanism import _seg_lookup, _load_block_motifs, _enrich_blocks

    blocks = build_cue_blocks_with_segments(df_all)
    if not blocks:
        return None
    lookup = _seg_lookup(df_all)
    enriched = _enrich_blocks(blocks, lookup, _load_block_motifs(output_dir))
    if not enriched:
        return None
    tlu = _text_lookup(df_all)

    L = []
    L.append("=" * 78)
    L.append("THERAPEUTIC LANGUAGE ATLAS")
    L.append("=" * 78)
    L.append("")
    L.append("The actual language behind the mechanism statistics: FROM participant quote →")
    L.append("therapist CUE → TO participant quote, for the therapist moves most associated")
    L.append("with progression — BOTH forward-moving and backward/stalling patterns. ")
    L.append("Directional/associational (see 02_mechanism/mechanism.txt for CIs and")
    L.append("significance). Read as candidate teachable patterns, not proof.")
    L.append("")

    # ---- Forward AND backward movers (from the inferential mechanism table) --
    mech_csv = os.path.join(_paths.mechanism_dir(output_dir), 'mechanism_delta_progression.csv')
    forward_specs = []   # (grouping, from_stage, behavior, mean_delta) — most positive
    backward_specs = []  # (grouping, from_stage, behavior, mean_delta) — most negative
    if os.path.isfile(mech_csv):
        try:
            mdf = pd.read_csv(mech_csv)
            for grouping in ('purer', 'motif'):
                g = mdf[mdf['grouping'] == grouping]
                if g.empty:
                    continue
                if 'fdr_significant' in g.columns and g['fdr_significant'].any():
                    g = g[g['fdr_significant']]
                fwd = g.sort_values('mean_delta_prog', ascending=False).head(3)
                bwd = g[g['mean_delta_prog'] < 0].sort_values('mean_delta_prog', ascending=True).head(3)
                for _, r in fwd.iterrows():
                    forward_specs.append((grouping, r['from_stage'], r['behavior'], r['mean_delta_prog']))
                for _, r in bwd.iterrows():
                    backward_specs.append((grouping, r['from_stage'], r['behavior'], r['mean_delta_prog']))
        except Exception:
            pass

    key_of = {'purer': 'dominant_purer', 'motif': 'cue_motif'}

    def _emit_specs(specs, most_negative_first):
        if not specs:
            L.append("  (run the mechanism analysis first to populate ranked movers)")
            return
        for grouping, from_stage, behavior, delta in specs:
            kf = key_of[grouping]
            examples = [b for b in enriched
                        if b['from_stage'] == from_stage and str(b.get(kf)) == str(behavior)]
            examples.sort(key=lambda b: b['delta_prog'] if most_negative_first else -b['delta_prog'])
            if not examples:
                continue
            fname = framework.get(int(from_stage), {}).get('short_name', str(from_stage))
            L.append(f"\n  [{grouping}] {behavior}  —  from {fname}  (mean Δ={delta:+.3f})")
            for b in examples[:2]:
                _render_block(b, tlu, framework, L)

    L.append("-" * 78)
    L.append("1a. TOP FORWARD-MOVING THERAPIST LANGUAGE (Δprogression > 0)")
    L.append("-" * 78)
    _emit_specs(forward_specs, most_negative_first=False)

    L.append("")
    L.append("-" * 78)
    L.append("1b. BACKWARD-MOVING / STALLING THERAPIST LANGUAGE (Δprogression < 0)")
    L.append("    Read as 'patterns to notice and avoid', not as blame — same caveats apply.")
    L.append("-" * 78)
    if backward_specs:
        _emit_specs(backward_specs, most_negative_first=True)
    else:
        L.append("  (no behaviors with negative mean Δprogression met the threshold)")

    # ---- Emergent motifs (high influence, low PURER purity) -----------------
    L.append("-" * 78)
    L.append("2. EMERGENT MOTIFS — influential language not captured by PURER")
    L.append("-" * 78)
    motif_csv = os.path.join(_paths.gnn_data_dir(output_dir), 'cue_motifs.csv')
    if os.path.isfile(motif_csv):
        try:
            cm = pd.read_csv(motif_csv)
            emergent = cm[(cm.get('influence', 0) >= 1.2)]
            if 'purer_purity' in cm.columns:
                emergent = emergent[emergent['purer_purity'] < 0.5]
            emergent = emergent.sort_values('influence', ascending=False).head(3)
            if emergent.empty:
                L.append("  None flagged (no high-influence, low-purity motifs).")
            for _, r in emergent.iterrows():
                mid = int(r['motif_id'])
                L.append(f"\n  Motif {mid}: influence={r.get('influence'):.2f}, "
                         f"dominant PURER={r.get('dominant_purer')} (purity={r.get('purer_purity')})")
                ex = [b for b in enriched if b.get('cue_motif') == mid]
                ex.sort(key=lambda b: -b['delta_prog'])
                for b in ex[:1]:
                    _render_block(b, tlu, framework, L)
        except Exception:
            L.append("  (cue_motifs.csv unreadable)")
    else:
        L.append("  (GNN motif discovery not available — enable the GNN layer)")

    # ---- Named coupling / alliance factors ----------------------------------
    L.append("-" * 78)
    L.append("3. ALLIANCE / COUPLING FACTORS (GNN latent factors, named)")
    L.append("-" * 78)
    coup_csv = os.path.join(_paths.gnn_data_dir(output_dir), 'coupling_factors.csv')
    if os.path.isfile(coup_csv):
        try:
            cf = pd.read_csv(coup_csv)
            for _, r in cf.iterrows():
                name = r.get('nearest_cf_ic') or f"factor {int(r.get('factor', 0))}"
                corr = r.get('forward_corr')
                corr_s = f"{corr:+.3f}" if isinstance(corr, (int, float)) and corr == corr else 'n/a'
                L.append(f"  {str(name):<24} forward-corr={corr_s}  "
                         f"(var explained={r.get('explained_variance_ratio')})")
        except Exception:
            L.append("  (coupling_factors.csv unreadable)")
    else:
        L.append("  (GNN coupling not available — enable the GNN layer)")
    L.append("")

    rep_dir = _paths.reports_mechanism_dir(output_dir)
    os.makedirs(rep_dir, exist_ok=True)
    path = os.path.join(rep_dir, 'language_atlas.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path
