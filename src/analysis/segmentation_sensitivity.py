"""
analysis/segmentation_sensitivity.py
------------------------------------
Segmentation-Sensitivity Check — a trust instrument (Feature 2).

The headline H1 progression slope is computed on ONE frozen segmentation. This
module asks: does that result survive reasonable perturbations of the
segmentation parameters? It re-segments the raw transcripts under a grid of
`SegmentationConfig` settings and PROJECTS the existing per-segment values onto
the new units (token-overlap-weighted — NO re-classification, the values of
record are reused):
  - the continuous superposition PROGRESSION COORDINATE (the ACTUAL headline H1
    statistic, recomputed via `superposition.attach_superposition` exactly as the
    headline is) by overlap-weighted MEAN — so the instrument perturbs the real
    mixture slope, not the hard-label E[stage] proxy; and
  - the hard VAAMR label (for barrier/occupancy) by overlap-weighted majority.
It then recomputes the H1 group slope + barrier rate under each arm. It reports
how much the boundary placement moved (per-session E[stage] Spearman + boundary
Jaccard) vs a CANONICAL-IN-SAME-EMBEDDER baseline (see below).

This converts a frozen researcher choice into a STABILITY statement, scoped:
"the headline progression direction is stable / sensitive across the grid —
boundary-placement only, labels/coords projected, not re-classified." Aggregate
stability is partly STRUCTURAL (values are re-grouped, not re-derived); a
re-classifying arm is the stronger, more expensive test (named as future work).

Cost discipline: `use_llm_refinement` is forced OFF in every arm, so the check is
frontier-LLM-cost-free. When the configured (Qwen) embedder is unavailable it
falls back to all-MiniLM — and CRUCIALLY the canonical baseline every arm is
compared to is re-segmented with that SAME embedder (default params, LLM off), so
the only thing varying across arms is the parameter under test (a param-only
perturbation, not an embedder/refinement swap). The report states the embedder
actually used; the honest claim is "robustness within the MiniLM embedding space;
Qwen-space pending the embedder fix."

OPT-IN: gated by `config.validation.run_segmentation_sensitivity` (default False)
and the `qra analyze --segmentation-sensitivity` flag. Default analyze is
unchanged.

Outputs
-------
  03_analysis_data/segmentation_sensitivity.csv
  06_reports/01_outcomes/segmentation_sensitivity.txt
"""

import itertools
import os
from dataclasses import replace
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from process import output_paths as _paths
from .loader import load_segments, sort_session_ids
from . import efficacy as _eff


# ── Perturbation grid ───────────────────────────────────────────────────────

# Each entry: SegmentationConfig field -> candidate values. The canonical value
# (the project default) is included so its arm reproduces the frozen result.
_DEFAULT_GRID: Dict[str, list] = {
    'semantic_shift_percentile': [15, 25, 35],
    'min_segment_words_conversational': [40, 60, 80],
    'max_segment_words_conversational': [350, 500, 650],
    'broad_window_size': [5, 7, 9],
    'use_adaptive_threshold': [True, False],
}


def _one_factor_at_a_time(grid: Dict[str, list], base) -> List[dict]:
    """Build settings: the canonical arm + one-factor sweeps off it.

    A full Cartesian product (3·3·3·3·2 = 162 arms) would re-segment the whole
    corpus 162×. A one-factor-at-a-time (OFAT) design around the canonical config
    keeps the corpus re-segmentation count to ~1 + Σ(|values|−1) while still
    probing every axis. The canonical arm is emitted first and labeled.
    """
    settings: List[dict] = []
    base_vals = {k: getattr(base, k, grid[k][len(grid[k]) // 2]) for k in grid}
    settings.append({'__label__': 'canonical', **base_vals})
    seen = {tuple(sorted(base_vals.items()))}
    for field, values in grid.items():
        for v in values:
            cand = dict(base_vals)
            cand[field] = v
            key = tuple(sorted(cand.items()))
            if key in seen:
                continue
            seen.add(key)
            settings.append({'__label__': f'{field}={v}', **cand})
    return settings


# ── Raw transcript loading (reuse the orchestrator's loaders) ───────────────

def _load_raw_sessions(config) -> List[dict]:
    """Discover + load raw transcript sessions as [{session_id, sentences, metadata}].

    Reuses transcript_ingestion's discovery + VTT/JSON loaders so this matches the
    real ingest path. Returns [] when no transcripts are found.
    """
    from process.transcript_ingestion import (
        discover_session_files, load_vtt_session, load_diarized_session,
        parse_session_id_metadata,
    )
    import glob as _glob

    tdir = config.transcript_dir
    files = discover_session_files(tdir)
    if not files:
        files = sorted(
            set(_glob.glob(os.path.join(tdir, '**/*.json'), recursive=True))
            | set(_glob.glob(os.path.join(tdir, '**/*.vtt'), recursive=True))
        )

    out = []
    for f in files:
        if f.lower().endswith('.vtt'):
            data = load_vtt_session(f)
            stem = os.path.splitext(os.path.basename(f))[0]
        else:
            data = load_diarized_session(f)
            stem = os.path.basename(os.path.dirname(f))
        parsed = parse_session_id_metadata(stem)
        meta = data.get('metadata', {})
        meta.setdefault('trial_id', getattr(config, 'trial_id', ''))
        meta.setdefault('session_id', stem)
        meta.setdefault('session_number', parsed['session_number'])
        meta.setdefault('cohort_id', parsed['cohort_id'])
        meta.setdefault('session_variant', parsed['session_variant'])
        out.append({'session_id': meta['session_id'],
                    'sentences': data.get('sentences', []),
                    'metadata': meta})
    return out


def _build_seg_config(config, setting: dict) -> dict:
    """Build the ConversationalSegmenter config dict for one perturbation setting.

    Forces `use_llm_refinement` semantics OFF (we never construct a refiner) and
    carries the speaker-filter so therapist turns are excluded exactly as in the
    real participant segmentation.
    """
    seg = config.segmentation
    sf = getattr(config, 'speaker_filter', None)
    excluded = sf.speakers if (sf is not None and sf.mode == 'exclude') else []

    # Perturbed fields: take the swept value when this setting carries it, else fall
    # back to the segmentation-config default — so any GRID SUBSET works (a grid need
    # not include every perturbable axis).
    def _val(field, default):
        return setting[field] if field in setting else getattr(seg, field, default)

    return {
        'embedding_model': getattr(seg, 'embedding_model', 'Qwen/Qwen3-Embedding-8B'),
        'silence_threshold_ms': getattr(seg, 'silence_threshold_ms', 1500),
        'semantic_shift_percentile': _val('semantic_shift_percentile', 25),
        'min_segment_words_conversational': _val('min_segment_words_conversational', 60),
        'max_segment_words_conversational': _val('max_segment_words_conversational', 500),
        'max_gap_seconds': getattr(seg, 'max_gap_seconds', 30.0),
        'min_words_per_sentence': getattr(seg, 'min_words_per_sentence', 10),
        'max_segment_duration_seconds': getattr(seg, 'max_segment_duration_seconds', 300.0),
        'excluded_speakers': excluded,
        'speaker_filter_mode': (sf.mode if sf is not None else 'none'),
        'use_adaptive_threshold': _val('use_adaptive_threshold', True),
        'min_prominence': getattr(seg, 'min_prominence', 0.05),
        'broad_window_size': _val('broad_window_size', 7),
        'use_topic_clustering': getattr(seg, 'use_topic_clustering', True),
    }


# ── Label projection (token-overlap-weighted majority, NO re-classification) ─

_WORD_RE = __import__('re').compile(r"[a-z0-9']+")


def _tokens(s: str) -> List[str]:
    return _WORD_RE.findall(str(s).casefold())


def _ensure_progression_coord(canon_df: pd.DataFrame, output_dir: str, config, log):
    """Ensure canon_df carries a per-segment continuous progression coordinate.

    The headline H1 statistic is the superposition mixture coordinate, attached by
    ``superposition.attach_superposition`` (GNN mixture → LLM ballots → secondary) —
    NOT the hard-label E[stage]. We recompute it here on the canonical participant rows
    exactly as the headline does, so the canonical arm reproduces the real mixture slope
    (a stale persisted ``progression_coord`` column, if any, is overwritten).

    Returns ``(coord_col, note)`` where ``coord_col`` is ``'progression_coord'`` when a
    usable continuous coordinate is available (some rows finite) and ``None`` when it is
    NOT — in which case projection falls back to the hard label and the report says so.
    ``note`` is a short human string describing what happened (mixture source or fallback
    reason).
    """
    try:
        from .superposition import attach_superposition
        superpos_cfg = getattr(config, 'superposition', None) if config is not None else None
        if superpos_cfg is None or getattr(superpos_cfg, 'enabled', True):
            # Drop any stale persisted coordinate so we recompute from the live signal.
            canon_df.drop(columns=['progression_coord', 'mixture', 'stage_mixture'],
                          errors='ignore', inplace=True)
            attach_superposition(canon_df, output_dir, config=superpos_cfg)
    except Exception as e:                              # pragma: no cover - defensive
        log(f"    superposition attach failed ({type(e).__name__}); "
            f"projecting hard labels instead.")

    if 'progression_coord' in canon_df.columns and canon_df['progression_coord'].notna().any():
        try:
            from .superposition import dominant_source
            src = dominant_source(canon_df)
        except Exception:
            src = 'unknown'
        cov = float(canon_df['progression_coord'].notna().mean())
        note = (f"projecting the continuous superposition coordinate "
                f"(mixture source: {src}; {cov*100:.0f}% of canonical segments).")
        log(f"    {note}")
        return 'progression_coord', note
    note = ("progression_coord unavailable — FALLING BACK to the hard-label E[stage] "
            "(the instrument then perturbs the hard-label proxy, not the headline mixture).")
    log(f"    {note}")
    return None, note


def _canon_units(canon_df: pd.DataFrame, coord_col: Optional[str]) -> List[dict]:
    """Pre-tokenize canonical participant rows once (label + optional continuous coord).

    Each unit carries its hard `label`, its content-token set, time bounds, and — when
    `coord_col` is present and finite — the continuous mixture `coord` (the actual H1
    statistic). Rows without a usable `final_label` are skipped.
    """
    units = []
    for _, r in canon_df.iterrows():
        lab = r.get('final_label')
        if lab is None or (isinstance(lab, float) and pd.isna(lab)):
            continue
        coord = None
        if coord_col is not None:
            cv = r.get(coord_col)
            if cv is not None and not (isinstance(cv, float) and pd.isna(cv)):
                try:
                    coord = float(cv)
                except (TypeError, ValueError):
                    coord = None
        units.append({
            'start': int(r.get('start_time_ms', 0) or 0),
            'end': int(r.get('end_time_ms', 0) or 0),
            'label': int(lab),
            'coord': coord,
            'tokens': set(_tokens(r.get('text', ''))),
        })
    return units


def _overlap_weights(seg, canon: List[dict]):
    """Token-overlap (within temporal overlap) weights of each canonical unit for one new seg.

    Returns (weighted_units, total_weight) where weighted_units is [(unit, w), ...] over
    the canonical units that TEMPORALLY overlap the new segment, w = token-Jaccard +
    a tiny temporal-overlap floor. Empty when nothing overlaps.
    """
    s_start = int(getattr(seg, 'start_time_ms', 0) or 0)
    s_end = int(getattr(seg, 'end_time_ms', 0) or 0)
    s_tokens = set(_tokens(getattr(seg, 'text', '')))
    weighted = []
    for c in canon:
        overlaps = (c['start'] < s_end and c['end'] > s_start) if (s_end > s_start) else False
        if not overlaps:
            continue
        jac = (len(s_tokens & c['tokens']) / len(s_tokens | c['tokens'])
               if (s_tokens or c['tokens']) else 0.0)
        inter_ms = max(0, min(s_end, c['end']) - max(s_start, c['start']))
        w = jac + 1e-6 * inter_ms
        weighted.append((c, w))
    return weighted


def _nearest_unit(seg, canon: List[dict]) -> Optional[dict]:
    """Temporally-nearest canonical unit (by midpoint distance), or None."""
    s_start = int(getattr(seg, 'start_time_ms', 0) or 0)
    s_end = int(getattr(seg, 'end_time_ms', 0) or 0)
    if not canon or s_end <= 0:
        return None
    mid = (s_start + s_end) / 2.0
    return min(canon, key=lambda c: abs((c['start'] + c['end']) / 2.0 - mid))


def _project_labels_and_coords(new_segs, canon: List[dict]
                               ) -> tuple:
    """Project frozen VAAMR labels AND the continuous progression coordinate onto new segs.

    For each new segment, candidate canonical units are weighted by content-token
    overlap (within temporal overlap):
      - the hard LABEL with the greatest summed overlap weight wins (argmax — used for
        occupancy/barrier, the labeler-of-record), and
      - the continuous COORD is the overlap-WEIGHTED MEAN of the canonical coords (the
        ACTUAL H1 statistic — the superposition mixture coordinate), so the instrument
        perturbs the real headline rather than the hard-label E[stage] proxy.
    Both fall back to the temporally-nearest canonical unit when no token overlap is
    found. Returns (labels, coords), each aligned to `new_segs` order; an entry is None
    when no canonical unit is reachable (label) / when no coord is available (coord).
    """
    labels: List[Optional[int]] = []
    coords: List[Optional[float]] = []
    for seg in new_segs:
        weighted = _overlap_weights(seg, canon)
        if weighted:
            # Hard label: argmax of summed per-label weight.
            lab_w: Dict[int, float] = {}
            for c, w in weighted:
                lab_w[c['label']] = lab_w.get(c['label'], 0.0) + w
            labels.append(max(lab_w.items(), key=lambda kv: kv[1])[0])
            # Continuous coord: overlap-weighted mean over units that carry a coord.
            num = den = 0.0
            for c, w in weighted:
                if c['coord'] is not None and w > 0:
                    num += w * c['coord']
                    den += w
            coords.append(round(num / den, 6) if den > 0 else None)
            continue

        # Fallback: nearest canonical unit by midpoint distance.
        nearest = _nearest_unit(seg, canon)
        if nearest is not None:
            labels.append(nearest['label'])
            coords.append(round(nearest['coord'], 6) if nearest['coord'] is not None else None)
        else:
            labels.append(None)
            coords.append(None)
    return labels, coords


# ── Boundary comparison metrics ─────────────────────────────────────────────

def _boundary_set(segs) -> set:
    """Set of segment END times (ms) — the boundary positions for a session."""
    return {int(getattr(s, 'end_time_ms', 0) or 0) for s in segs
            if int(getattr(s, 'end_time_ms', 0) or 0) > 0}


def _boundary_jaccard_sets(new_ends: set, canon_ends: set, tol_ms: int = 1000) -> float:
    """Jaccard of two boundary-position SETS (end-times), matching within tol_ms."""
    if not new_ends and not canon_ends:
        return 1.0
    if not new_ends or not canon_ends:
        return 0.0
    matched_canon = set()
    for ne in new_ends:
        for ce in canon_ends:
            if ce in matched_canon:
                continue
            if abs(ne - ce) <= tol_ms:
                matched_canon.add(ce)
                break
    inter = len(matched_canon)
    union = len(new_ends) + len(canon_ends) - inter
    return inter / union if union else 1.0


def _boundary_jaccard(new_segs, canon_ends: set, tol_ms: int = 1000) -> float:
    """Jaccard of boundary positions (canonical vs new segments), matching within tol_ms."""
    return _boundary_jaccard_sets(_boundary_set(new_segs), canon_ends, tol_ms)


def _spearman_vs_canonical(per_session_new: Dict[str, float],
                           per_session_canon: Dict[str, float]) -> Optional[float]:
    """Spearman ρ of per-session E[stage]: new arm vs canonical (paired by session)."""
    common = [s for s in per_session_new if s in per_session_canon]
    pairs = [(per_session_new[s], per_session_canon[s]) for s in common
             if per_session_new[s] == per_session_new[s]
             and per_session_canon[s] == per_session_canon[s]]
    if len(pairs) < 3:
        return None
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    if np.all(a == a[0]) or np.all(b == b[0]):
        return None
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(a, b)
        return float(rho) if rho == rho else None
    except Exception:
        return None


# ── One-arm evaluation ──────────────────────────────────────────────────────

def _new_segs_to_ps_frame(new_by_session: Dict[str, list],
                          labels_by_session: Dict[str, List[Optional[int]]],
                          coords_by_session: Dict[str, List[Optional[float]]],
                          part_meta: Dict[str, dict]) -> pd.DataFrame:
    """Assemble a participant-segment df (the shape efficacy.compute_* expect).

    Carries the projected continuous ``progression_coord`` (the superposition mixture
    coordinate — the ACTUAL H1 statistic) alongside the hard ``final_label``. When a
    projected coord is available for a segment, ``compute_participant_session_outcomes``
    uses it (its ``has_prog`` branch); otherwise that segment's coord is NaN and the
    per-session mean falls back to hard labels (disclosed in the report).
    """
    rows = []
    for sid, segs in new_by_session.items():
        labs = labels_by_session.get(sid, [])
        crds = coords_by_session.get(sid, [])
        meta = part_meta.get(sid, {})
        for seg, lab, crd in zip(segs, labs, crds):
            if lab is None:
                continue
            rows.append({
                'segment_id': getattr(seg, 'segment_id', '') or f'{sid}_{getattr(seg,"segment_index",0)}',
                'participant_id': getattr(seg, 'participant_id', '') or meta.get('participant_id', sid),
                'session_id': sid,
                'session_number': int(meta.get('session_number', 0) or 0),
                'final_label': int(lab),
                'progression_coord': (float(crd) if crd is not None else float('nan')),
                'start_time_ms': int(getattr(seg, 'start_time_ms', 0) or 0),
                'end_time_ms': int(getattr(seg, 'end_time_ms', 0) or 0),
            })
    return pd.DataFrame(rows)


def _evaluate_arm(ps_df: pd.DataFrame, config) -> dict:
    """Compute the H1 group slope + barrier-crossing rate for one re-segmented arm.

    H1 is the mixed-effects slope of the **progression coordinate** over sessions. The
    coordinate is the projected continuous superposition mixture (the ACTUAL headline
    statistic) for every (participant, session) whose segments carried a projected
    coord; where coord coverage is missing, ``compute_participant_session_outcomes``
    falls back to the hard-label E[stage] for that cell. ``coord_coverage`` reports the
    fraction of per-segment rows that carried a projected coordinate, ``trend_method``
    the estimator actually used (mixedlm/ols/none), and ``converged`` whether the
    mixed-effects optimiser reached an interior optimum (False ⇒ "MLE on the boundary"
    — surfaced, not hidden).
    """
    res = {'h1_slope': None, 'h1_ci_lo': None, 'h1_ci_hi': None,
           'h1_p': None, 'barrier_rate': None, 'n_labeled': int(len(ps_df)),
           'coord_coverage': None, 'trend_method': None, 'converged': None}
    if ps_df.empty:
        return res

    # Fraction of per-segment rows carrying a projected continuous coordinate.
    if 'progression_coord' in ps_df.columns and len(ps_df):
        res['coord_coverage'] = round(float(ps_df['progression_coord'].notna().mean()), 4)

    eff_cfg = getattr(config, 'efficacy', None)
    try:
        ps = _eff.compute_participant_session_outcomes(ps_df, eff_cfg)
    except Exception:
        ps = pd.DataFrame()
    if ps.empty:
        return res

    # H1: mixed-effects slope of the progression coordinate over sessions. Capture any
    # ConvergenceWarning ("MLE on the boundary") so it is reported rather than hidden.
    try:
        import warnings as _warnings
        from . import stats as S
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            trend = S.mixedlm_trend(ps, 'progression_coord', 'session_number', 'participant_id')
        res['trend_method'] = trend.get('method')
        boundary = any('boundary' in str(w.message).lower()
                       or 'convergence' in str(w.category.__name__).lower()
                       for w in caught)
        # mixedlm that warned on the boundary did not cleanly converge; ols/none have
        # no random-effects optimiser to converge.
        if trend.get('method') == 'mixedlm':
            res['converged'] = (not boundary)
        res['h1_slope'] = (round(float(trend['slope']), 4)
                           if trend.get('slope') == trend.get('slope') else None)
        res['h1_ci_lo'] = (round(float(trend['ci_lo']), 4)
                           if trend.get('ci_lo') == trend.get('ci_lo') else None)
        res['h1_ci_hi'] = (round(float(trend['ci_hi']), 4)
                           if trend.get('ci_hi') == trend.get('ci_hi') else None)
        res['h1_p'] = (round(float(trend['p_value']), 4)
                       if trend.get('p_value') == trend.get('p_value') else None)
    except Exception:
        pass

    # Barrier crossing rate (fraction of participants who cross to Attn-Reg).
    try:
        barrier = _eff.compute_barrier_crossing(ps_df, eff_cfg)
        if len(barrier):
            res['barrier_rate'] = round(
                float(barrier['crossed_to_attention_regulation'].mean()), 4)
    except Exception:
        pass
    return res


# ── Orchestrator ────────────────────────────────────────────────────────────

def run_segmentation_sensitivity(
    output_dir: str,
    config,
    grid: Optional[Dict[str, list]] = None,
    *,
    segmenter_factory: Optional[Callable[[dict], object]] = None,
    verbose: bool = True,
) -> dict:
    """Re-segment under a perturbation grid, reuse frozen labels, recompute H1.

    Parameters
    ----------
    output_dir : str
        Pipeline output directory (must hold raw transcripts + frozen labels).
    config : PipelineConfig
        Provides segmentation defaults, transcript_dir, speaker_filter, efficacy.
    grid : dict, optional
        {SegmentationConfig field -> [values]}. Defaults to `_DEFAULT_GRID`.
    segmenter_factory : callable, optional
        `seg_config_dict -> segmenter`; the segmenter must expose
        `segment_session(sentences, metadata) -> [Segment]`. Defaults to the real
        `ConversationalSegmenter`. Injected by the hermetic unit test (no embeddings).
    verbose : bool
        Print progress.

    Returns {'files_written': [...], 'table': DataFrame, 'verdict': str}.
    """
    def log(m):
        if verbose:
            print(f"  {m}")

    grid = grid or _DEFAULT_GRID

    # Canonical (frozen) participant labels — the labels of record we project.
    try:
        canon_df = load_segments(output_dir, speaker_filter='participant', require_labeled=True)
    except FileNotFoundError:
        log("segmentation-sensitivity: no master_segments found — skipped.")
        return {'files_written': [], 'table': pd.DataFrame(), 'verdict': 'skipped (no data)'}
    if canon_df.empty:
        return {'files_written': [], 'table': pd.DataFrame(), 'verdict': 'skipped (no labels)'}

    # F2-A — give the canonical rows their continuous progression coordinate (the REAL
    # headline H1 statistic), recomputed via attach_superposition exactly as the headline
    # is (GNN mixture → LLM ballots → secondary), so the canonical arm reproduces the
    # real mixture slope rather than a stale persisted column. Track whether a coordinate
    # is actually available; if not, every arm falls back to the hard-label E[stage] and
    # the report SAYS SO.
    coord_col, coord_note = _ensure_progression_coord(canon_df, output_dir, config, log)

    canon_by_session = {sid: g for sid, g in canon_df.groupby('session_id')}
    part_meta = {}
    for sid, g in canon_by_session.items():
        row = g.iloc[0]
        part_meta[sid] = {'participant_id': str(row.get('participant_id', sid)),
                          'session_number': int(row.get('session_number', 0) or 0)}
    # Canonical UNITS per session (label + content tokens + continuous coord) — the
    # projection source. NOT used as the boundary/Spearman baseline (see F2-C below).
    canon_units = {sid: _canon_units(g, coord_col) for sid, g in canon_by_session.items()}

    raw_sessions = _load_raw_sessions(config)
    if not raw_sessions:
        log("segmentation-sensitivity: no raw transcripts discoverable — skipped.")
        return {'files_written': [], 'table': pd.DataFrame(),
                'verdict': 'skipped (no raw transcripts)'}

    settings = _one_factor_at_a_time(grid, config.segmentation)

    # Segmenter factory: real ConversationalSegmenter unless injected. Embedding
    # fallback (Qwen unavailable -> all-MiniLM) handled here, gracefully. We record the
    # embedder ACTUALLY used so the report can state it (F2-C).
    embed_state = {'fallback': False, 'model': None}
    if segmenter_factory is None:
        def _default_factory(seg_cfg: dict):
            from process.transcript_ingestion import ConversationalSegmenter
            try:
                seg = ConversationalSegmenter(seg_cfg)
                embed_state['model'] = embed_state['model'] or seg_cfg.get('embedding_model')
                return seg
            except Exception as e:
                log(f"    embedder {seg_cfg.get('embedding_model')!r} unavailable "
                    f"({type(e).__name__}); falling back to all-MiniLM.")
                seg_cfg = dict(seg_cfg)
                seg_cfg['embedding_model'] = 'sentence-transformers/all-MiniLM-L6-v2'
                embed_state['fallback'] = True
                embed_state['model'] = 'sentence-transformers/all-MiniLM-L6-v2'
                return ConversationalSegmenter(seg_cfg)
        segmenter_factory = _default_factory

    def _segment_arm(seg_cfg: dict):
        """Re-segment every session under one config; project labels+coords; collect
        per-session participant segments, projected labels/coords, E[stage], bounds."""
        try:
            segmenter = segmenter_factory(seg_cfg)
        except Exception as e:
            log(f"      segmenter construction failed: {e} — arm skipped.")
            return None
        new_by_session: Dict[str, list] = {}
        labels_by_session: Dict[str, List[Optional[int]]] = {}
        coords_by_session: Dict[str, List[Optional[float]]] = {}
        estage_by_session: Dict[str, float] = {}
        bounds_by_session: Dict[str, set] = {}
        for sess in raw_sessions:
            sid = sess['session_id']
            if sid not in canon_units:
                continue   # only sessions we have frozen labels for
            try:
                segs = segmenter.segment_session(sess['sentences'], sess['metadata'])
            except Exception as e:
                log(f"      segment_session failed for {sid}: {e}")
                continue
            part_segs = [s for s in segs if getattr(s, 'speaker', '') == 'participant']
            if not part_segs:
                part_segs = [s for s in segs if getattr(s, 'speaker', '') != 'therapist']
            labels, coords = _project_labels_and_coords(part_segs, canon_units[sid])
            new_by_session[sid] = part_segs
            labels_by_session[sid] = labels
            coords_by_session[sid] = coords
            lab_vals = [l for l in labels if l is not None]
            estage_by_session[sid] = float(np.mean(lab_vals)) if lab_vals else float('nan')
            bounds_by_session[sid] = _boundary_set(part_segs)
        return {'segs': new_by_session, 'labels': labels_by_session,
                'coords': coords_by_session, 'estage': estage_by_session,
                'bounds': bounds_by_session}

    def _arm_row(label: str, seg: dict, base_estage: dict, base_bounds: dict) -> dict:
        """Evaluate one segmented arm + compare to the (MiniLM-)canonical baseline."""
        ps_df = _new_segs_to_ps_frame(seg['segs'], seg['labels'], seg['coords'], part_meta)
        arm = _evaluate_arm(ps_df, config)
        jacs = [_boundary_jaccard_sets(seg['bounds'].get(sid, set()), base_bounds.get(sid, set()))
                for sid in seg['bounds']]
        rho = _spearman_vs_canonical(seg['estage'], base_estage)
        return {
            'setting': label,
            'n_segments': int(sum(len(v) for v in seg['segs'].values())),
            'n_labeled': arm['n_labeled'],
            'coord_coverage': arm['coord_coverage'],
            'h1_slope': arm['h1_slope'],
            'h1_ci_lo': arm['h1_ci_lo'],
            'h1_ci_hi': arm['h1_ci_hi'],
            'h1_p': arm['h1_p'],
            'trend_method': arm['trend_method'],
            'converged': arm['converged'],
            'barrier_rate': arm['barrier_rate'],
            'estage_spearman_vs_canon': (round(rho, 4) if rho is not None else None),
            'boundary_jaccard_vs_canon': (round(float(np.mean(jacs)), 4) if jacs else None),
        }

    # ── F2-C: canonical-in-MiniLM baseline ──────────────────────────────────
    # Re-segment ONCE at default params with the SAME embedder the arms use (MiniLM
    # fallback when Qwen won't load) and LLM refinement off. THIS — not the original
    # Qwen+LLM-refined frozen segmentation — is the baseline every arm is compared to,
    # so the ONLY thing varying across arms is the parameter under test (a true
    # param-only perturbation, not a wholesale embedder/refinement swap).
    canonical_setting = next((s for s in settings if s['__label__'] == 'canonical'), settings[0])
    base_seg = _segment_arm(_build_seg_config(config, canonical_setting))
    if base_seg is None:
        return {'files_written': [], 'table': pd.DataFrame(),
                'verdict': 'skipped (canonical baseline segmentation failed)'}
    base_estage, base_bounds = base_seg['estage'], base_seg['bounds']

    table_rows: List[dict] = []
    canon_slope = None
    n_attempted = 0
    for setting in settings:
        label = setting['__label__']
        log(f"    arm: {label}")
        n_attempted += 1
        # Reuse the already-segmented canonical baseline rather than re-segmenting it.
        seg = base_seg if label == canonical_setting['__label__'] else \
            _segment_arm(_build_seg_config(config, setting))
        if seg is None:
            # Failed arm: record it (F2-D) so it is COUNTED, not silently dropped.
            table_rows.append({'setting': label, 'n_segments': 0, 'n_labeled': 0,
                               'coord_coverage': None, 'h1_slope': None,
                               'h1_ci_lo': None, 'h1_ci_hi': None, 'h1_p': None,
                               'trend_method': None, 'converged': None,
                               'barrier_rate': None, 'estage_spearman_vs_canon': None,
                               'boundary_jaccard_vs_canon': None})
            continue
        row = _arm_row(label, seg, base_estage, base_bounds)
        if label == canonical_setting['__label__']:
            canon_slope = row['h1_slope']
        table_rows.append(row)

    table = pd.DataFrame(table_rows)
    verdict = _verdict(table, canon_slope)

    embedder_used = embed_state['model'] or getattr(
        getattr(config, 'segmentation', None), 'embedding_model', 'unknown')
    files = _write_outputs(table, verdict, embed_state['fallback'], embedder_used,
                           coord_col is not None, coord_note, output_dir)
    return {'files_written': files, 'table': table, 'verdict': verdict,
            'embedder_used': embedder_used, 'coord_used': coord_col is not None,
            'n_arms_total': len(table), 'n_arms_with_slope': int(table['h1_slope'].notna().sum())
            if not table.empty else 0}


def _verdict(table: pd.DataFrame, canon_slope) -> str:
    """One-line stability call: do all arms agree on the H1 slope SIGN?

    F2-B: the verdict is scope-disclosing — labels/coords are PROJECTED (re-grouped),
    not re-classified, so aggregate stability is partly invariant-by-construction. The
    word "STABLE" is therefore qualified, never bare.
    F2-D: the verdict reports K/N — how many of the N attempted arms actually produced a
    slope (failed/non-converged arms are counted, not silently dropped).
    """
    if table.empty:
        return "indeterminate — no arms evaluated"
    n_total = len(table)
    slopes = [s for s in table['h1_slope'].tolist() if s is not None]
    k = len(slopes)
    kn = f"based on {k}/{n_total} arms that produced a slope"
    if k < 2:
        return f"indeterminate — too few arms produced a slope ({kn})"
    signs = {('+' if s > 0 else ('-' if s < 0 else '0')) for s in slopes}
    lo, hi = min(slopes), max(slopes)
    if len(signs) == 1 and '0' not in signs:
        return ("STABLE (boundary-placement; labels/coords projected, not re-classified) "
                f"— every evaluated arm's H1 slope shares the same sign "
                f"(range {lo:+.4f} … {hi:+.4f}; {kn}).")
    return (f"SENSITIVE — H1 slope sign is NOT constant across the grid "
            f"(range {lo:+.4f} … {hi:+.4f}; {kn}); interpret the headline with caution.")


def _write_outputs(table: pd.DataFrame, verdict: str, embed_fallback: bool,
                   embedder_used: str, coord_used: bool, coord_note: str,
                   output_dir: str) -> List[str]:
    files: List[str] = []
    data_dir = _paths.analysis_data_dir(output_dir)
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'segmentation_sensitivity.csv')
    table.to_csv(csv_path, index=False)
    files.append(csv_path)

    # F2-D bookkeeping: how many of the attempted arms actually produced a slope, and
    # which mixed-effects fits did NOT cleanly converge ("MLE on the boundary").
    n_total = len(table)
    n_slope = int(table['h1_slope'].notna().sum()) if (not table.empty and 'h1_slope' in table) else 0
    n_failed = int((table['n_segments'] == 0).sum()) if (not table.empty and 'n_segments' in table) else 0
    nonconv = ([str(r['setting']) for _, r in table.iterrows()
                if r.get('converged') is False] if not table.empty else [])
    ols_arms = ([str(r['setting']) for _, r in table.iterrows()
                 if r.get('trend_method') == 'ols'] if not table.empty else [])

    short = 'all-MiniLM-L6-v2' if 'MiniLM' in str(embedder_used) else str(embedder_used)

    L: List[str] = []
    L.append("=" * 78)
    L.append("SEGMENTATION-SENSITIVITY CHECK")
    L.append("=" * 78)
    L.append("")
    # F2-B — structural-caveat one-liner at the very top.
    L.append("SCOPE: aggregate stability here is PARTLY STRUCTURAL — labels/coords are held")
    L.append("fixed and only RE-GROUPED onto new boundaries (not re-classified), so the")
    L.append("trajectory is partly invariant-by-construction to regrouping. A re-classifying")
    L.append("arm (re-running the LLM/probe on re-segmented text) is the stronger, more")
    L.append("expensive test — named as future work below.")
    L.append("")
    L.append("Does the headline H1 progression result survive reasonable perturbations of")
    L.append("the segmentation parameters? Each arm RE-SEGMENTS the raw transcripts, then")
    if coord_used:
        L.append("PROJECTS the existing per-segment values onto the new units by token-overlap-")
        L.append("weighted projection (NO re-classification): the continuous superposition")
        L.append("PROGRESSION COORDINATE — the ACTUAL headline H1 statistic — by weighted MEAN,")
        L.append("and the hard VAAMR label (for barrier/occupancy) by weighted majority. It then")
        L.append("recomputes the H1 mixed-effects slope of that coordinate + the barrier rate.")
    else:
        L.append("PROJECTS the frozen VAAMR labels onto the new units by token-overlap-weighted")
        L.append("majority (NO re-classification). NOTE: the continuous progression coordinate")
        L.append("was UNAVAILABLE, so H1 here is the HARD-LABEL E[stage] proxy, NOT the headline")
        L.append("mixture coordinate — " + coord_note)
    L.append("")
    L.append("LLM segmentation refinement is DISABLED in every arm (frontier-cost-free).")
    L.append("")
    # F2-C — embedder disclosure + param-only-perturbation framing.
    L.append(f"EMBEDDER: {short}.")
    if embed_fallback:
        L.append("  The configured (Qwen3) embedder would not load under the pinned transformers,")
        L.append("  so all-MiniLM was used for EVERY arm INCLUDING the canonical baseline. The")
        L.append("  baseline every arm is compared to is therefore a canonical-in-MiniLM re-")
        L.append("  segmentation (default params, LLM off) — so the only thing varying across")
        L.append("  arms is the parameter under test (a param-only perturbation, not an")
        L.append("  embedder/refinement swap). Honest scope: robustness WITHIN the MiniLM")
        L.append("  embedding space; Qwen-space robustness pending the embedder fix.")
    else:
        L.append("  The canonical baseline every arm is compared to was re-segmented at default")
        L.append("  params with this SAME embedder (LLM off), so arms differ only by the")
        L.append("  parameter under test (a true param-only perturbation).")
    L.append("")
    L.append("-" * 78)
    L.append(f"VERDICT: {verdict}")
    L.append("-" * 78)
    L.append("")

    if table.empty:
        L.append("  (no arms evaluated)")
    else:
        # Per-arm table. 'cov' = projected-coordinate coverage; '!' marks a non-converged
        # mixed-effects fit, 'ols' marks an OLS fallback (F2-D, surfaced not hidden).
        hdr = (f"  {'setting':<34}{'n_seg':>7}{'cov':>6}{'H1 slope':>10}{'[95% CI]':>18}"
               f"{'barr':>6}{'ρ':>6}{'bJac':>6}{'fit':>6}")
        L.append(hdr)
        L.append("  " + "-" * (len(hdr) - 2))
        for _, r in table.iterrows():
            slope = '  n/a' if r['h1_slope'] is None else f"{r['h1_slope']:+.4f}"
            if r['h1_ci_lo'] is None or r['h1_ci_hi'] is None:
                ci = '[     n/a      ]'
            else:
                ci = f"[{r['h1_ci_lo']:+.3f},{r['h1_ci_hi']:+.3f}]"
            cov = '  n/a' if r.get('coord_coverage') is None else f"{r['coord_coverage']*100:.0f}%"
            barrier = ' n/a' if r['barrier_rate'] is None else f"{r['barrier_rate']:.2f}"
            rho = ' n/a' if r['estage_spearman_vs_canon'] is None else f"{r['estage_spearman_vs_canon']:+.2f}"
            jac = ' n/a' if r['boundary_jaccard_vs_canon'] is None else f"{r['boundary_jaccard_vs_canon']:.2f}"
            meth = r.get('trend_method')
            if int(r['n_segments']) == 0:
                fit = 'FAIL'
            elif meth == 'ols':
                fit = 'ols'
            elif r.get('converged') is False:
                fit = 'bdry!'
            elif meth == 'mixedlm':
                fit = 'mix'
            else:
                fit = '  -'
            L.append(f"  {str(r['setting'])[:33]:<34}{int(r['n_segments']):>7}{cov:>6}"
                     f"{slope:>10}{ci:>18}{barrier:>6}{rho:>6}{jac:>6}{fit:>6}")
        L.append("")
        coord_word = ("continuous superposition progression coordinate"
                      if coord_used else "HARD-LABEL E[stage] proxy (coordinate unavailable)")
        L.append(f"  H1 slope = mixed-effects slope of the {coord_word}")
        L.append("             over sessions (random participant).")
        L.append("  cov      = fraction of this arm's projected segments that carried a continuous")
        L.append("             coordinate (rest fall back to hard label for that segment).")
        L.append("  barr     = fraction of participants who cross to Attention-Regulation.")
        L.append("  ρ        = per-session E[stage] Spearman of this arm vs the canonical baseline.")
        L.append("  bJac     = boundary-position Jaccard vs the canonical baseline (<=1s match).")
        L.append("  fit      = mix(ed-effects) | bdry! (MLE on the boundary; did NOT converge) |")
        L.append("             ols (mixedlm unavailable, OLS fallback) | FAIL (arm produced no slope).")
        L.append("")
        # F2-D — explicit arm accounting.
        L.append(f"  Arms: {n_slope}/{n_total} produced a slope"
                 + (f"; {n_failed} failed outright" if n_failed else "")
                 + (f"; {len(nonconv)} non-converged (boundary): {', '.join(nonconv)}" if nonconv else "")
                 + (f"; {len(ols_arms)} used OLS fallback: {', '.join(ols_arms)}" if ols_arms else "")
                 + ".")
    L.append("")
    L.append("CAVEAT: this isolates the BOUNDARY-PLACEMENT contribution to H1 — projection")
    L.append("reuses the canonical labels/coordinate and does NOT re-classify, so aggregate")
    L.append("stability is partly structural (it survives even heavy re-grouping). The")
    L.append("stronger, more expensive test — a re-classifying arm that re-runs the LLM/probe")
    L.append("on each re-segmented unit — is FUTURE WORK. Associational/robustness, not causal.")

    report_dir = _paths.reports_outcomes_dir(output_dir)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'segmentation_sensitivity.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    files.append(report_path)
    return files
