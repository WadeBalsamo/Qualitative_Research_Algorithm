"""
analysis/mechanism.py
---------------------
Mechanistic FROM→CUE→TO analysis on the continuous progression coordinate.

The hard pipeline already reports therapist-cue transitions as discrete
forward/backward/lateral moves. With per-segment stage *mixtures* attached
(analysis.superposition), we can ask the deeper question the methodology is
built around: *what therapist language moves participants, by how much, and
where does it have the most leverage?*

For each cue block (a participant FROM segment, the therapist language between,
and the participant TO segment) we compute a continuous

    Δprogression = progression_coord(TO) − progression_coord(FROM)

and aggregate it by the therapist behaviour in the block (dominant PURER move,
dominant microskill, and — when the GNN ran — emergent cue motif), conditioned
on the FROM stage. Three lenses follow:

  1. Δprogression mechanism table — which behaviours move participants furthest,
     given where they start.
  2. Liminality leverage — do cues bite harder when the participant is at a
     stage cusp (high mixture entropy)?
  3. Avoidance barrier — restricted to Avoidance, what crosses participants
     toward Attention-Regulation, and does the cusp move earlier across sessions?

Plus a rule-based participant trajectory typology and a GNN↔LLM construct-
validity summary. Everything here is exploratory / directional (methodology
§5.4): it generates hypotheses for the next cohort, it does not confirm them.
"""

import os
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from process import output_paths as _paths
from .loader import sort_session_ids

# VAAMR stage indices (0=Vigilance .. 4=Reappraisal).
AVOIDANCE = 1
ATTENTION_REGULATION = 2

# Canonical PURER construct names (for readable behaviour labels; falls back to id).
_PURER_NAMES = {0: 'Phenomenology', 1: 'Utilization', 2: 'Reframing',
                3: 'Education', 4: 'Reinforcement'}


def _purer_label(val):
    """Readable PURER label 'Reframing(2)' from a numeric id; pass through otherwise."""
    if val is None:
        return None
    try:
        k = int(val)
    except (ValueError, TypeError):
        return str(val)
    name = _PURER_NAMES.get(k)
    return f'{name}({k})' if name else str(k)


def _slope(ys: List[float]) -> float:
    """Least-squares slope of ys against position index. 0.0 if < 2 points."""
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return round(num / den, 4) if den != 0 else 0.0


def _seg_lookup(df_all: pd.DataFrame) -> Dict[str, dict]:
    """segment_id → {progression_coord, mixture, mixture_entropy, purer, microskills}."""
    out: Dict[str, dict] = {}
    has_purer = 'purer_primary' in df_all.columns
    has_micro = 'microskill_labels_ensemble' in df_all.columns
    for _, r in df_all.iterrows():
        sid = str(r.get('segment_id', ''))
        if not sid:
            continue
        micro = r.get('microskill_labels_ensemble') if has_micro else None
        out[sid] = {
            'progression_coord': float(r['progression_coord']) if pd.notna(r.get('progression_coord')) else None,
            'mixture': r.get('mixture'),
            'mixture_entropy': float(r['mixture_entropy']) if pd.notna(r.get('mixture_entropy')) else None,
            'purer': r.get('purer_primary') if has_purer else None,
            'microskills': micro if isinstance(micro, list) else [],
        }
    return out


def _load_block_motifs(output_dir: str) -> Dict[tuple, int]:
    """(from_seg_id, to_seg_id) → motif_id, from cue_block_assignments.csv (GNN, optional)."""
    path = os.path.join(_paths.gnn_data_dir(output_dir), 'cue_block_assignments.csv')
    if not os.path.isfile(path):
        return {}
    try:
        mdf = pd.read_csv(path)
    except Exception:
        return {}
    if not {'from_seg_id', 'to_seg_id', 'motif_id'}.issubset(mdf.columns):
        return {}
    return {
        (str(r['from_seg_id']), str(r['to_seg_id'])): int(r['motif_id'])
        for _, r in mdf.iterrows()
    }


def _enrich_blocks(blocks: List[dict], lookup: Dict[str, dict],
                   block_motifs: Dict[tuple, int]) -> List[dict]:
    """Attach Δprogression, FROM entropy, and dominant therapist behaviours to blocks."""
    enriched = []
    for b in blocks:
        fl = lookup.get(b['from_seg_id'])
        tl = lookup.get(b['to_seg_id'])
        if not fl or not tl or fl['progression_coord'] is None or tl['progression_coord'] is None:
            continue
        delta = tl['progression_coord'] - fl['progression_coord']

        # Dominant therapist PURER move + microskill across the block's therapist segments.
        purers = [lookup[s]['purer'] for s in b['therapist_seg_ids']
                  if s in lookup and lookup[s]['purer'] not in (None, '') and not (isinstance(lookup[s]['purer'], float) and pd.isna(lookup[s]['purer']))]
        micros = []
        for s in b['therapist_seg_ids']:
            if s in lookup:
                micros.extend([m for m in lookup[s]['microskills'] if m])
        dom_purer = _purer_label(Counter(purers).most_common(1)[0][0]) if purers else None
        dom_micro = Counter(micros).most_common(1)[0][0] if micros else None

        enriched.append({
            **b,
            'delta_prog': round(float(delta), 4),
            'from_entropy': fl['mixture_entropy'],
            'from_mixture': fl['mixture'],
            'dominant_purer': dom_purer,
            'dominant_microskill': dom_micro,
            'cue_motif': block_motifs.get((b['from_seg_id'], b['to_seg_id'])),
            'n_therapist_segments': len(b['therapist_seg_ids']),
        })
    return enriched


def _agg_delta(blocks: List[dict], framework: dict, key_field: str, grouping: str,
               min_n: int = 2) -> List[dict]:
    """Aggregate mean Δprogression by (from_stage, blocks[key_field])."""
    rows = []
    buckets: Dict[tuple, List[dict]] = {}
    for b in blocks:
        kv = b.get(key_field)
        if kv is None:
            continue
        buckets.setdefault((b['from_stage'], kv), []).append(b)
    for (from_stage, kv), bs in buckets.items():
        if len(bs) < min_n:
            continue
        deltas = [x['delta_prog'] for x in bs]
        ents = [x['from_entropy'] for x in bs if x['from_entropy'] is not None]
        rows.append({
            'grouping': grouping,
            'from_stage': from_stage,
            'from_stage_name': framework.get(from_stage, {}).get('short_name', str(from_stage)),
            'behavior': kv,
            'n': len(bs),
            'mean_delta_prog': round(float(np.mean(deltas)), 4),
            'sd_delta_prog': round(float(np.std(deltas)), 4),
            'mean_from_entropy': round(float(np.mean(ents)), 4) if ents else None,
        })
    rows.sort(key=lambda r: (r['from_stage'], -r['mean_delta_prog']))
    return rows


def _liminality_leverage(blocks: List[dict]) -> dict:
    """Bin FROM segments by entropy; report |Δprog| leverage at the cusp."""
    bins = {'low': [], 'medium': [], 'high': []}
    for b in blocks:
        e = b.get('from_entropy')
        if e is None:
            continue
        tier = 'low' if e < 0.33 else ('medium' if e < 0.66 else 'high')
        bins[tier].append(b)

    summary = {}
    for tier, bs in bins.items():
        if not bs:
            summary[tier] = {'n': 0}
            continue
        adeltas = [abs(x['delta_prog']) for x in bs]
        deltas = [x['delta_prog'] for x in bs]
        purers = [x['dominant_purer'] for x in bs if x['dominant_purer']]
        summary[tier] = {
            'n': len(bs),
            'mean_abs_delta_prog': round(float(np.mean(adeltas)), 4),
            'mean_delta_prog': round(float(np.mean(deltas)), 4),
            'top_purer': Counter(purers).most_common(3),
        }

    # Correlation between FROM entropy and |Δprog| (leverage-at-the-cusp test).
    es = [b['from_entropy'] for b in blocks if b['from_entropy'] is not None]
    ds = [abs(b['delta_prog']) for b in blocks if b['from_entropy'] is not None]
    corr = None
    if len(es) >= 3 and np.std(es) > 0 and np.std(ds) > 0:
        corr = round(float(np.corrcoef(es, ds)[0, 1]), 4)
    summary['entropy_abs_delta_correlation'] = corr
    return summary


def _avoidance_barrier(blocks: List[dict], framework: dict) -> dict:
    """What therapist language crosses participants out of Avoidance, ranked by Δprog."""
    av_blocks = [b for b in blocks if b['from_stage'] == AVOIDANCE]
    by_purer = _agg_delta(av_blocks, framework, 'dominant_purer', 'avoidance_purer', min_n=2)
    by_micro = _agg_delta(av_blocks, framework, 'dominant_microskill', 'avoidance_microskill', min_n=2)
    by_motif = _agg_delta(av_blocks, framework, 'cue_motif', 'avoidance_motif', min_n=2)
    return {
        'n_avoidance_blocks': len(av_blocks),
        'mean_delta_from_avoidance': round(float(np.mean([b['delta_prog'] for b in av_blocks])), 4) if av_blocks else None,
        'by_purer': by_purer,
        'by_microskill': by_micro,
        'by_motif': by_motif,
    }


def _cusp_density_by_session(df: pd.DataFrame) -> List[dict]:
    """Fraction of participant segments on the Avoidance↔Attention-Regulation cusp, per session.

    A segment is on the barrier cusp when its top-2 mixture stages are exactly
    {Avoidance, Attention-Regulation}. Ordered longitudinally to show whether the
    barrier is crossed earlier across the program.
    """
    if 'mixture' not in df.columns:
        return []
    rows = []
    for sid in sort_session_ids(df['session_id'].unique().tolist()):
        sdf = df[df['session_id'] == sid]
        n = len(sdf)
        if n == 0:
            continue
        cusp = 0
        for mix in sdf['mixture']:
            vec = np.asarray(mix, dtype=np.float64)
            order = np.argsort(vec)[::-1]
            top2 = {int(order[0]), int(order[1])}
            if top2 == {AVOIDANCE, ATTENTION_REGULATION}:
                cusp += 1
        rows.append({
            'session_id': sid,
            'n_segments': n,
            'cusp_count': cusp,
            'cusp_density': round(cusp / n, 4),
        })
    return rows


def _trajectory_typology(df: pd.DataFrame, framework: dict) -> List[dict]:
    """Rule-based participant trajectory types from the continuous progression series.

    KMeans is inappropriate at cohort sizes of 1–2, so we use interpretable
    thresholds on slope / volatility / level. Returns one row per participant.
    """
    if 'progression_coord' not in df.columns:
        return []
    rows = []
    for pid in sorted(df['participant_id'].unique().tolist()):
        pdf = df[df['participant_id'] == pid]
        session_means = []
        within_sds = []
        for sid in sort_session_ids(pdf['session_id'].unique().tolist()):
            vals = pdf[pdf['session_id'] == sid]['progression_coord'].dropna().tolist()
            if vals:
                session_means.append(float(np.mean(vals)))
                within_sds.append(float(np.std(vals)))
        if not session_means:
            continue
        slope = _slope(session_means)
        volatility = round(float(np.mean(within_sds)), 4) if within_sds else 0.0
        level_max = round(float(np.max(session_means)), 4)
        end_minus_start = round(session_means[-1] - session_means[0], 4)
        mean_level = float(np.mean(session_means))

        if slope > 0.1 and end_minus_start > 0.2:
            ttype = 'climber'
        elif mean_level <= AVOIDANCE + 0.3 and slope <= 0.1:
            ttype = 'stuck_at_avoidance'
        elif volatility >= 0.9:
            ttype = 'oscillator'
        elif mean_level >= ATTENTION_REGULATION + 0.5:
            ttype = 'advanced'
        else:
            ttype = 'stable'

        rows.append({
            'participant_id': pid,
            'n_sessions': len(session_means),
            'progression_slope': slope,
            'within_session_volatility': volatility,
            'max_progression': level_max,
            'end_minus_start': end_minus_start,
            'trajectory_type': ttype,
        })
    return rows


def _construct_validity(output_dir: str) -> Optional[dict]:
    """Summarize GNN↔LLM lift convergence from gnn_vs_llm_lift.csv (if present)."""
    path = os.path.join(_paths.gnn_data_dir(output_dir), 'gnn_vs_llm_lift.csv')
    if not os.path.isfile(path):
        return None
    try:
        cdf = pd.read_csv(path)
    except Exception:
        return None
    cols = set(cdf.columns)
    summary = {'n_pairs': int(len(cdf))}
    if 'both_elevated' in cols:
        summary['n_both_elevated'] = int(cdf['both_elevated'].astype(bool).sum())
    elif {'lift_gnn', 'lift_llm'}.issubset(cols):
        both = ((cdf['lift_gnn'] >= 1.5) & (cdf['lift_llm'] >= 1.5)).sum()
        summary['n_both_elevated'] = int(both)
    return summary


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_csv(rows: List[dict], path: str) -> Optional[str]:
    if not rows:
        return None
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_mechanism_report(delta_rows, liminality, avoidance, cusp_density,
                            trajectories, construct, framework, path) -> str:
    L = []
    L.append("=" * 78)
    L.append("MECHANISTIC ANALYSIS — therapist language × continuous Δprogression")
    L.append("=" * 78)
    L.append("")
    L.append("DIRECTIONAL / HYPOTHESIS-GENERATING. Δprogression is the change in the")
    L.append("continuous VAAMR progression coordinate (E[stage], 0–4) from the FROM to")
    L.append("the TO participant segment of each therapist cue block. Positive = movement")
    L.append("toward later stages. These are observational associations, not causal effects.")
    L.append("")

    L.append("-" * 78)
    L.append("1. ΔPROGRESSION BY THERAPIST BEHAVIOUR × FROM-STAGE (top movers)")
    L.append("-" * 78)
    if delta_rows:
        for grouping in ('purer', 'microskill', 'motif'):
            grows = [r for r in delta_rows if r['grouping'] == grouping]
            if not grows:
                continue
            L.append(f"\n  [{grouping.upper()}]")
            for r in sorted(grows, key=lambda r: -r['mean_delta_prog'])[:12]:
                L.append(f"    {r['from_stage_name']:<22} {str(r['behavior'])[:28]:<28} "
                         f"Δ={r['mean_delta_prog']:+.3f} (±{r['sd_delta_prog']:.2f}, n={r['n']})")
    else:
        L.append("  No behaviour-labelled cue blocks available.")

    L.append("")
    L.append("-" * 78)
    L.append("2. LIMINALITY LEVERAGE — do cues bite harder at the stage cusp?")
    L.append("-" * 78)
    corr = liminality.get('entropy_abs_delta_correlation')
    L.append(f"  Correlation(FROM entropy, |Δprogression|): "
             f"{corr if corr is not None else 'n/a'}")
    for tier in ('low', 'medium', 'high'):
        t = liminality.get(tier, {})
        if t.get('n'):
            L.append(f"    {tier:<7} entropy: n={t['n']:<4} "
                     f"mean|Δ|={t['mean_abs_delta_prog']:+.3f}  meanΔ={t['mean_delta_prog']:+.3f}")
    L.append("  (Higher mean|Δ| at high entropy ⇒ therapist language has maximal leverage at the cusp.)")

    L.append("")
    L.append("-" * 78)
    L.append("3. AVOIDANCE BARRIER — what crosses participants toward Attention-Regulation")
    L.append("-" * 78)
    L.append(f"  Avoidance cue blocks: {avoidance['n_avoidance_blocks']}  "
             f"(mean Δprogression from Avoidance: {avoidance['mean_delta_from_avoidance']})")
    for label, key in (('PURER', 'by_purer'), ('Microskill', 'by_microskill'), ('Motif', 'by_motif')):
        ranked = [r for r in avoidance[key] if r['mean_delta_prog'] > 0][:6]
        if ranked:
            L.append(f"\n  Top {label} movers out of Avoidance:")
            for r in ranked:
                L.append(f"    {str(r['behavior'])[:30]:<30} Δ={r['mean_delta_prog']:+.3f} (n={r['n']})")
    if cusp_density:
        L.append("\n  Avoidance↔Attention-Regulation cusp density by session (longitudinal):")
        for r in cusp_density:
            L.append(f"    {r['session_id']:<10} {r['cusp_density']*100:5.1f}%  "
                     f"({r['cusp_count']}/{r['n_segments']})")
        L.append("  (Falling density / earlier sessions crossing ⇒ the barrier is being worked through.)")

    L.append("")
    L.append("-" * 78)
    L.append("4. PARTICIPANT TRAJECTORY TYPOLOGY (rule-based)")
    L.append("-" * 78)
    if trajectories:
        for r in trajectories:
            L.append(f"    {r['participant_id']:<16} {r['trajectory_type']:<20} "
                     f"slope={r['progression_slope']:+.3f}  vol={r['within_session_volatility']:.2f}  "
                     f"max={r['max_progression']:.2f}")
    else:
        L.append("  No trajectory data.")

    L.append("")
    L.append("-" * 78)
    L.append("5. CONSTRUCT VALIDITY — GNN ↔ LLM convergence")
    L.append("-" * 78)
    if construct:
        nb = construct.get('n_both_elevated')
        L.append(f"  (stage, code) pairs compared: {construct['n_pairs']}")
        if nb is not None:
            L.append(f"  Pairs elevated under BOTH the GNN geometry and LLM labels: {nb}")
            L.append("  Convergence across independent substrates is stronger evidence than")
            L.append("  LLM↔LLM agreement — but remains directional pending human validation.")
    else:
        L.append("  GNN lift comparison not available (GNN layer did not run).")
        L.append("  Mixtures here are sourced from LLM ballots / secondary_stage — a stability")
        L.append("  signal, not an independent substrate. Interpret accordingly.")
    L.append("")

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path


def _write_avoidance_report(avoidance, cusp_density, framework, path) -> str:
    L = []
    L.append("=" * 78)
    L.append("AVOIDANCE BARRIER REPORT")
    L.append("=" * 78)
    L.append("")
    L.append("The Avoidance→Attention-Regulation crossing is the clinically central barrier.")
    L.append("This report ranks therapist language by the continuous Δprogression it is")
    L.append("associated with when participants begin in Avoidance, and tracks how the")
    L.append("Avoidance↔Attention-Regulation cusp density shifts across the program.")
    L.append("DIRECTIONAL — associational, curriculum-actionable hypotheses, not causal claims.")
    L.append("")
    L.append(f"Avoidance cue blocks analysed: {avoidance['n_avoidance_blocks']}")
    L.append(f"Mean Δprogression out of Avoidance: {avoidance['mean_delta_from_avoidance']}")
    L.append("")
    for label, key in (('PURER move', 'by_purer'), ('Microskill', 'by_microskill'), ('Cue motif', 'by_motif')):
        ranked = avoidance[key]
        if not ranked:
            continue
        L.append("-" * 78)
        L.append(f"{label} ranked by Δprogression from Avoidance")
        L.append("-" * 78)
        for r in ranked:
            L.append(f"  {str(r['behavior'])[:34]:<34} Δ={r['mean_delta_prog']:+.3f} "
                     f"(±{r['sd_delta_prog']:.2f}, n={r['n']})")
        L.append("")
    if cusp_density:
        L.append("-" * 78)
        L.append("Avoidance↔Attention-Regulation cusp density by session")
        L.append("-" * 78)
        for r in cusp_density:
            bar = '#' * int(round(r['cusp_density'] * 40))
            L.append(f"  {r['session_id']:<10} {r['cusp_density']*100:5.1f}% {bar}")
        L.append("")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_mechanism_analysis(df: pd.DataFrame, df_all: pd.DataFrame,
                           output_dir: str, framework: dict) -> dict:
    """Run the full mechanistic analysis. Returns {files_written, n_blocks}."""
    from gnn_layer.inference import build_cue_blocks_with_segments

    files_written: List[str] = []
    mech_dir = _paths.mechanism_dir(output_dir)
    os.makedirs(mech_dir, exist_ok=True)

    blocks = build_cue_blocks_with_segments(df_all)
    if not blocks:
        return {'files_written': files_written, 'n_blocks': 0}

    lookup = _seg_lookup(df_all)
    block_motifs = _load_block_motifs(output_dir)
    enriched = _enrich_blocks(blocks, lookup, block_motifs)
    if not enriched:
        return {'files_written': files_written, 'n_blocks': 0}

    # 1. Δprogression mechanism table.
    delta_rows = []
    delta_rows += _agg_delta(enriched, framework, 'dominant_purer', 'purer')
    delta_rows += _agg_delta(enriched, framework, 'dominant_microskill', 'microskill')
    delta_rows += _agg_delta(enriched, framework, 'cue_motif', 'motif')
    p = _write_csv(delta_rows, os.path.join(mech_dir, 'mechanism_delta_progression.csv'))
    if p:
        files_written.append(p)

    # 2. Liminality leverage.
    liminality = _liminality_leverage(enriched)
    lim_rows = [{'entropy_bin': t, **{k: v for k, v in liminality.get(t, {}).items() if k != 'top_purer'}}
                for t in ('low', 'medium', 'high') if liminality.get(t, {}).get('n')]
    p = _write_csv(lim_rows, os.path.join(mech_dir, 'mechanism_liminality.csv'))
    if p:
        files_written.append(p)

    # 3. Avoidance barrier + cusp density.
    avoidance = _avoidance_barrier(enriched, framework)
    cusp_density = _cusp_density_by_session(df)
    av_rows = avoidance['by_purer'] + avoidance['by_microskill'] + avoidance['by_motif']
    p = _write_csv(av_rows, os.path.join(mech_dir, 'mechanism_avoidance_barrier.csv'))
    if p:
        files_written.append(p)
    p = _write_csv(cusp_density, os.path.join(mech_dir, 'avoidance_cusp_density_by_session.csv'))
    if p:
        files_written.append(p)

    # 4. Trajectory typology.
    trajectories = _trajectory_typology(df, framework)
    p = _write_csv(trajectories, os.path.join(mech_dir, 'participant_trajectory_types.csv'))
    if p:
        files_written.append(p)

    # 5. Construct validity.
    construct = _construct_validity(output_dir)

    # Reports.
    rep_dir = _paths.human_reports_dir(output_dir)
    os.makedirs(rep_dir, exist_ok=True)
    files_written.append(_write_mechanism_report(
        delta_rows, liminality, avoidance, cusp_density, trajectories, construct,
        framework, os.path.join(rep_dir, 'report_mechanism.txt'),
    ))
    files_written.append(_write_avoidance_report(
        avoidance, cusp_density, framework,
        os.path.join(rep_dir, 'report_avoidance_barrier.txt'),
    ))

    # Figures that depend on the CSVs just written.
    try:
        from .figures import generate_mechanism_figures
        files_written.extend(generate_mechanism_figures(output_dir, framework))
    except Exception:
        pass

    return {'files_written': files_written, 'n_blocks': len(enriched)}
