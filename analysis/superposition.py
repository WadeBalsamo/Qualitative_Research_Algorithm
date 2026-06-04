"""
analysis/superposition.py
-------------------------
Unified VAAMR stage-mixture (superposition) provider for the analysis layer.

The methodology holds that participant segments exist *in superposition* — they
express a blend of VAAMR stages rather than a single one. That signal is already
present in the data three ways, in descending fidelity:

  1. GNN geometry  — ``03_analysis_data/gnn/segment_positions.csv`` (when the GNN
     layer has run): a learned 5-way mixture per segment.
  2. LLM ballots   — ``rater_votes`` in the master CSV: multi-run votes whose
     spread encodes stage ambiguity (reconstructed via gnn_layer.soft_labels).
  3. secondary_stage — the dual-coding fallback: a two-point mixture from
     ``final_label`` + ``secondary_stage`` weighted by their confidences.

``attach_superposition`` adds mixture columns to any analysis DataFrame so every
downstream report can show the superposition instead of a hard argmax. It is
purely additive — existing ``final_label`` columns are untouched — and degrades
gracefully (GNN → ballots → secondary) so it works on cohorts that never ran the
GNN. All mixture-derived quantities are directional / hypothesis-generating;
hard labels remain the labeler-of-record.
"""

import math
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from process import output_paths as _paths
from gnn_layer import soft_labels as _sl

N_VAAMR_STAGES = 5

# Column names attached by attach_superposition().
SUPERPOSITION_COLUMNS = (
    'mixture', 'progression_coord', 'mixture_entropy', 'max_stage',
    'second_stage', 'n_active_stages', 'is_liminal', 'mixture_source',
)


def mixture_entropy(mixture, n_stages: int = N_VAAMR_STAGES) -> float:
    """Normalized Shannon entropy H = -Σ p·ln p / ln K  ∈ [0, 1] (liminality)."""
    m = np.asarray(mixture, dtype=np.float64)
    s = m.sum()
    if s <= 0:
        return 0.0
    m = m / s
    h = 0.0
    for p in m:
        if p > 0:
            h -= p * math.log(p)
    denom = math.log(n_stages) if n_stages > 1 else 1.0
    return float(h / denom) if denom > 0 else 0.0


def _secondary_mixture(row, n_stages: int) -> Optional[np.ndarray]:
    """Two-point mixture from final_label + secondary_stage weighted by confidence."""
    fl = row.get('final_label')
    if fl is None or (isinstance(fl, float) and pd.isna(fl)):
        return None
    try:
        fl = int(fl)
    except (ValueError, TypeError):
        return None
    if not (0 <= fl < n_stages):
        return None
    acc = np.zeros(n_stages, dtype=np.float64)
    cp = row.get('llm_confidence_primary')
    acc[fl] += float(cp) if isinstance(cp, (int, float)) and not pd.isna(cp) else 1.0
    sec = row.get('secondary_stage')
    if sec is not None and not (isinstance(sec, float) and pd.isna(sec)):
        try:
            sec = int(sec)
        except (ValueError, TypeError):
            sec = None
        if sec is not None and 0 <= sec < n_stages:
            cs = row.get('llm_confidence_secondary')
            acc[sec] += float(cs) if isinstance(cs, (int, float)) and not pd.isna(cs) else 0.5
    total = acc.sum()
    return acc / total if total > 0 else _sl.one_hot(fl, n_stages)


def _load_gnn_mixtures(output_dir: str, n_stages: int) -> dict:
    """Return {segment_id: np.ndarray[n_stages]} from segment_positions.csv, or {}."""
    path = os.path.join(_paths.gnn_data_dir(output_dir), 'segment_positions.csv')
    if not os.path.isfile(path):
        return {}
    try:
        gdf = pd.read_csv(path)
    except Exception:
        return {}
    mix_cols = [f'vaamr_mix_{k}' for k in range(n_stages)]
    if 'segment_id' not in gdf.columns or not all(c in gdf.columns for c in mix_cols):
        return {}
    out = {}
    for _, r in gdf.iterrows():
        vec = np.array([float(r[c]) for c in mix_cols], dtype=np.float64)
        s = vec.sum()
        if s > 0:
            out[str(r['segment_id'])] = vec / s
    return out


def _row_mixture(row, gnn_lookup: dict, mode: str, n_stages: int):
    """Resolve one row's mixture under the source-priority policy. Returns (vec, source)."""
    sid = str(row.get('segment_id', ''))

    if mode in ('auto', 'gnn') and sid in gnn_lookup:
        return gnn_lookup[sid], 'gnn'
    if mode == 'gnn':  # forced gnn but missing → fall through to ballots/secondary
        pass

    if mode in ('auto', 'gnn', 'ballots'):
        votes = _sl._parse_votes(row.get('rater_votes'))
        if votes:
            mix = _sl.ballots_to_mixture(votes, n_stages=n_stages)
            # ballots_to_mixture returns a uniform vector when nothing usable was found
            if not np.allclose(mix, 1.0 / n_stages):
                return mix, 'ballots'
        if mode == 'ballots' and votes:
            return _sl.ballots_to_mixture(votes, n_stages=n_stages), 'ballots'

    sec = _secondary_mixture(row, n_stages)
    if sec is not None:
        return sec, 'secondary'

    # Last resort: uniform (no information).
    return np.full(n_stages, 1.0 / n_stages, dtype=np.float64), 'none'


def attach_superposition(
    df: pd.DataFrame,
    output_dir: str,
    config=None,
    n_stages: int = N_VAAMR_STAGES,
) -> pd.DataFrame:
    """Attach stage-mixture / liminality columns to ``df`` in place and return it.

    Parameters
    ----------
    df : DataFrame
        Any analysis DataFrame (participant or all-speaker). Rows without a
        usable signal receive a uniform mixture with source ``'none'``.
    output_dir : str
        Pipeline output dir (used to locate the GNN segment_positions.csv).
    config : SuperpositionConfig | None
        Thresholds + ``mixture_source`` mode. Falls back to defaults when None.
    n_stages : int
        Number of VAAMR stages (default 5).
    """
    if df is None or len(df) == 0:
        return df

    # Resolve config thresholds (graceful defaults).
    ent_thr = getattr(config, 'liminal_entropy_threshold', 0.6) if config else 0.6
    gap_thr = getattr(config, 'liminal_gap_threshold', 0.25) if config else 0.25
    active_thr = getattr(config, 'active_stage_threshold', 0.15) if config else 0.15
    mode = getattr(config, 'mixture_source', 'auto') if config else 'auto'

    gnn_lookup = _load_gnn_mixtures(output_dir, n_stages) if mode in ('auto', 'gnn') else {}

    mixtures, progs, ents = [], [], []
    maxs, seconds, nactives, liminals, sources = [], [], [], [], []

    for _, row in df.iterrows():
        vec, source = _row_mixture(row, gnn_lookup, mode, n_stages)
        order = np.argsort(vec)[::-1]
        p1 = float(vec[order[0]])
        p2 = float(vec[order[1]]) if n_stages > 1 else 0.0
        ent = mixture_entropy(vec, n_stages)

        mixtures.append([round(float(x), 4) for x in vec])
        progs.append(round(_sl.mixture_to_progression(vec), 4))
        ents.append(round(ent, 4))
        maxs.append(int(order[0]))
        seconds.append(int(order[1]) if n_stages > 1 else int(order[0]))
        nactives.append(int((vec >= active_thr).sum()))
        liminals.append(bool(ent >= ent_thr or (p1 - p2) < gap_thr))
        sources.append(source)

    df['mixture'] = mixtures
    df['progression_coord'] = progs
    df['mixture_entropy'] = ents
    df['max_stage'] = maxs
    df['second_stage'] = seconds
    df['n_active_stages'] = nactives
    df['is_liminal'] = liminals
    df['mixture_source'] = sources
    return df


def dominant_source(df: pd.DataFrame) -> str:
    """Return the most common mixture_source in df (for report provenance lines)."""
    if 'mixture_source' not in df.columns or len(df) == 0:
        return 'none'
    counts = df['mixture_source'].value_counts()
    return str(counts.index[0]) if len(counts) else 'none'


def stage_cooccurrence_matrix(df: pd.DataFrame, n_stages: int = N_VAAMR_STAGES) -> List[List[float]]:
    """Symmetric n×n cusp matrix: mass that stage pairs co-express across segments.

    For each segment we take its top-2 mixture stages and add the product of their
    probabilities to the off-diagonal (and p² to the diagonal). Normalized by the
    number of segments. Reveals which stages live together at the boundary.
    """
    mat = np.zeros((n_stages, n_stages), dtype=np.float64)
    if 'mixture' not in df.columns or len(df) == 0:
        return mat.tolist()
    n = 0
    for mix in df['mixture']:
        vec = np.asarray(mix, dtype=np.float64)
        if vec.shape[0] != n_stages:
            continue
        order = np.argsort(vec)[::-1]
        a, b = int(order[0]), int(order[1])
        mat[a][a] += vec[a] * vec[a]
        mat[a][b] += vec[a] * vec[b]
        mat[b][a] += vec[a] * vec[b]
        n += 1
    if n > 0:
        mat /= n
    return np.round(mat, 4).tolist()
