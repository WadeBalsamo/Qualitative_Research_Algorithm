"""
gnn_layer/soft_labels.py
------------------------
Reconstruct soft VAAMR supervision targets from multi-run ballots (Capability A).

The superposition signal already exists per participant segment in ``rater_votes``
(schema: each ballot dict carries ``stage`` / ``vote`` and ``confidence``, optional
``secondary_stage`` / ``secondary_confidence``). This module turns those ballots into
a normalized 5-vector mixture over VAAMR stages — the signal majority-vote discards —
and a scalar progression coordinate E[stage] = sum_k k * p_k.

numpy only; no torch.
"""

import json
from typing import Dict, List, Optional

N_VAAMR_STAGES = 5  # 0=Vigilance .. 4=Reappraisal


def _coerce_stage(v, n_stages):
    """Return an int stage in [0, n_stages) or None (handles 'ABSTAIN', floats, str ints)."""
    if v is None:
        return None
    try:
        iv = int(v)
    except (ValueError, TypeError):
        return None
    return iv if 0 <= iv < n_stages else None


def ballots_to_mixture(rater_votes: Optional[List[dict]],
                       n_stages: int = N_VAAMR_STAGES,
                       secondary_weight: float = 0.5):
    """Convert a segment's ``rater_votes`` into a normalized stage-mixture vector.

    Each ballot contributes its primary stage (weight = confidence, default 1.0) and,
    if present, its secondary stage (weight = secondary_weight * secondary_confidence).
    Returns a uniform vector when no usable ballots exist (caller may substitute a
    final_label one-hot).
    """
    import numpy as np
    acc = np.zeros(n_stages, dtype=np.float64)
    if rater_votes:
        for b in rater_votes:
            if not isinstance(b, dict):
                continue
            stage = _coerce_stage(b.get('stage', b.get('vote')), n_stages)
            if stage is not None:
                conf = b.get('confidence')
                acc[stage] += float(conf) if isinstance(conf, (int, float)) else 1.0
            sec = _coerce_stage(b.get('secondary_stage'), n_stages)
            if sec is not None:
                sconf = b.get('secondary_confidence')
                acc[sec] += secondary_weight * (float(sconf) if isinstance(sconf, (int, float)) else 1.0)
    total = acc.sum()
    if total <= 0:
        return np.full(n_stages, 1.0 / n_stages, dtype=np.float64)
    return acc / total


def one_hot(stage: int, n_stages: int = N_VAAMR_STAGES):
    import numpy as np
    v = np.zeros(n_stages, dtype=np.float64)
    if stage is not None and 0 <= int(stage) < n_stages:
        v[int(stage)] = 1.0
    else:
        v[:] = 1.0 / n_stages
    return v


def mixture_to_progression(mixture) -> float:
    """Expected stage value sum_k k * p_k — the continuous progression coordinate."""
    import numpy as np
    m = np.asarray(mixture, dtype=np.float64)
    return float(np.dot(np.arange(m.shape[0]), m))


def _parse_votes(raw):
    """rater_votes may be a list (in-memory) or a JSON string (master CSV)."""
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            v = json.loads(raw)
            return v if isinstance(v, list) else None
        except (ValueError, TypeError):
            return None
    return None


def ballot_coverage(df_all) -> dict:
    """Summarize multi-run LLM ballot coverage across participant segments.

    The GNN's soft consensus targets are learned from per-segment ``rater_votes``
    (the multi-run LLM ballots).  When a segment carries <2 ballots,
    :func:`build_soft_targets` degrades to a one-hot of the final label — a weak
    training signal that makes the reliability gate κ optimistic/unreliable.  This
    quantifies coverage so ``analysis.runner`` / ``qra gnn train`` can warn when a
    project was classified with a single LLM run.

    Returns ``{'n_participant', 'n_with_multirun_ballots', 'multirun_fraction'}``.
    A missing ``rater_votes`` column counts as zero coverage (triggers the warning).
    """
    empty = {'n_participant': 0, 'n_with_multirun_ballots': 0, 'multirun_fraction': 0.0}
    if df_all is None or len(df_all) == 0:
        return empty
    part = df_all[df_all['speaker'] == 'participant'] if 'speaker' in df_all.columns else df_all
    n_part = len(part)
    if n_part == 0:
        return empty
    n_multi = 0
    if 'rater_votes' in part.columns:
        for _, row in part.iterrows():
            votes = _parse_votes(row.get('rater_votes'))
            if votes is not None and len(votes) >= 2:
                n_multi += 1
    return {
        'n_participant': int(n_part),
        'n_with_multirun_ballots': int(n_multi),
        'multirun_fraction': (n_multi / n_part) if n_part else 0.0,
    }


def build_soft_targets(df_all, label_mode: str = 'weak',
                       n_stages: int = N_VAAMR_STAGES) -> Dict[str, "object"]:
    """Return {segment_id: mixture_vector} for participant segments.

    label_mode:
      'weak'            — from rater_votes (fallback: one-hot of final_label)
      'human'           — only rows with in_human_coded_subset; one-hot of human_label
      'self_supervised' — {} (no label targets; link-prediction only)

    ``n_stages``:
      5 (default)       — the five VAAMR stages; behaviour is byte-identical to before.
      6                 — adds a "No code" class at index 5. In 'weak' mode a participant
                          row whose ``final_label`` is null/NaN gets a one-hot on class 5
                          (replacing the old uniform-noise fallback that design_decisions.md
                          §4 #4 flags); labeled rows keep their 5-stage ballot mixture in
                          dims 0..4 with dim 5 = 0.
    """
    if label_mode == 'self_supervised':
        return {}
    import numpy as np

    if 'speaker' in df_all.columns:
        part = df_all[df_all['speaker'] == 'participant']
    else:
        part = df_all

    six_class = int(n_stages) > N_VAAMR_STAGES
    targets: Dict[str, object] = {}
    for _, row in part.iterrows():
        sid = str(row.get('segment_id'))
        if label_mode == 'human':
            if not bool(row.get('in_human_coded_subset', False)):
                continue
            hl = row.get('human_label')
            stage = _coerce_stage(hl, n_stages)
            if stage is None:
                continue
            targets[sid] = one_hot(stage, n_stages)
            continue
        # weak
        if six_class:
            # 6-class "No code" handling: key off final_label. Null/NaN → a clean
            # one-hot on the No-code class (index N_VAAMR_STAGES); labeled rows keep
            # their 5-stage ballot mixture padded into dims 0..4 (dim 5 stays 0).
            stage = _coerce_stage(row.get('final_label', row.get('primary_stage')),
                                  N_VAAMR_STAGES)
            if stage is None:
                targets[sid] = one_hot(N_VAAMR_STAGES, n_stages)
                continue
            votes = _parse_votes(row.get('rater_votes'))
            mix5 = ballots_to_mixture(votes, n_stages=N_VAAMR_STAGES)
            if votes is None or np.allclose(mix5, 1.0 / N_VAAMR_STAGES):
                mix5 = one_hot(stage, N_VAAMR_STAGES)
            mix = np.zeros(int(n_stages), dtype=np.float64)
            mix[:N_VAAMR_STAGES] = mix5
            targets[sid] = mix
            continue
        votes = _parse_votes(row.get('rater_votes'))
        mix = ballots_to_mixture(votes, n_stages=n_stages)
        # If ballots were unusable, fall back to a final_label one-hot when available.
        if votes is None or np.allclose(mix, 1.0 / n_stages):
            stage = _coerce_stage(row.get('final_label', row.get('primary_stage')), n_stages)
            if stage is not None:
                mix = one_hot(stage, n_stages)
        targets[sid] = mix
    return targets
