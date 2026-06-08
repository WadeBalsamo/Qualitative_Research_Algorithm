"""
classification_tools/probe_classifier.py
-----------------------------------------
QRA's LLM-free, gated, abstention-aware VAAMR scaler — the production realization of the
distillation campaign winner (methodology §8.6; spec in scalable_classification_master_plan.md).

The multi-run LLM consensus stays the label of record (human-level, κ=0.537 vs human). This
probe distils that consensus to a cheap student that *fills* segments the LLM has not
labeled, abstains where unsure, and writes a ``probe_consensus`` provenance tier ranked
BELOW ``llm_zero_shot`` — never overriding it.

WINNER (campaign §0): a PER-RATER ENSEMBLE — one 6-class class-weighted L2-LogReg probe per
LLM rater, ensembled by mean ``predict_proba`` then argmax (C=4). On data/Meta this
reproduces classifier↔LLM grouped κ ≈ 0.36 / classifier↔human κ ≈ 0.45 (dominates the
single collapsed-consensus probe "A1n" at 0.283/0.365). When a project's ballots carry
fewer than two distinct raters, the model falls back to the single A1n probe on
``final_label``.

Pure sklearn + numpy. Features reuse the GNN/codebook Qwen embedding path
(``gnn_layer.embeddings.load_or_build_segment_embeddings``, shared npz cache); calibration
reuses ``gnn_layer.classifier.calibration``; folds + κ-CI reuse ``analysis.grouped_cv``;
the human axis reuses ``analysis.irr_join`` / ``analysis.irr_stats``. Ports the validated
experimental code (``experiments/gnn_reliability/baselines.py``,
``experiments/classification_scaler/rater_distill.py``) into production — no torch, no graph.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

PROBE_CODE_VERSION = '1.0'

# VAAMR stage names + the 6th "No code" class the human axis uses (ABSTAIN == -1).
PROBE_VAAMR_NAMES = {0: 'Vigilance', 1: 'Avoidance', 2: 'AttentionReg',
                     3: 'Metacognition', 4: 'Reappraisal', 5: 'No code'}
# The two binding rare stages (campaign §0): recall here gates promotion.
RARE_STAGES = (1, 3)
GATE_FILENAME = 'probe_gate.json'
MODEL_FILENAME = 'probe_model.joblib'
MANIFEST_FILENAME = 'probe_manifest.json'


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class ProbeConfig:
    """Serialized into ``PipelineConfig.probe``. Default OFF — the LLM stays primary."""
    enabled: bool = False
    # ---- features: reuse the GNN/codebook Qwen embedding substrate ----
    embedding_backend: str = 'openai'        # Qwen via LM Studio /v1/embeddings
    embedding_base_url: Optional[str] = 'http://10.0.0.58:1234/v1'
    embedding_model: str = 'text-embedding-qwen3-embedding-8b'
    use_query_prefix: bool = True
    embedding_batch_size: int = 16
    # ---- model: PER-RATER ENSEMBLE (campaign winner) ----
    ensemble: bool = True                    # True → one probe per LLM rater, mean proba (WINNER)
    raters: Tuple[str, ...] = ('google/gemma-4-31b', 'nvidia/nemotron-3-nano-30b',
                               'qwen/qwen3-next-80b')   # filtered to those present in the data
    n_classes: int = 6                       # 5 VAAMR stages + "No code"
    class_weight: Optional[str] = 'balanced'  # recovers rare stages
    C: float = 4.0                           # LLM-axis optimum (campaign C-sweep)
    max_iter: int = 3000
    # ---- calibration / abstention (don't poison the corpus) ----
    calibrate: bool = True
    abstain_threshold: Optional[float] = None             # global max-prob floor
    abstain_per_stage: Optional[Dict[int, float]] = None  # per-stage floors
    abstain_calibrate: bool = True                        # derive floors from held-out precision
    abstain_target_precision: float = 0.80
    # ---- gate (per-project trust) ----
    irr_human_band_floor: float = 0.33       # probe↔human κ must reach the human band to scale
    rare_stage_recall_floor: float = 0.20
    validation_folds: int = 5
    seed: int = 42


# ===========================================================================
# Probe primitives (ported from experiments/gnn_reliability/baselines.py)
# ===========================================================================
def _stack_l2(embeddings: Dict[str, "object"], ids: List[str]):
    """Stack the given segment embeddings into an [n, D] float64 matrix, L2-normalized."""
    from sklearn.preprocessing import normalize
    mat = np.stack([np.asarray(embeddings[s], dtype=np.float64) for s in ids], axis=0)
    return normalize(mat, norm='l2', axis=1)


class _ConstantClf:
    """Degenerate fallback when a training fold has <2 classes (predicts that class)."""
    def __init__(self, cls: int):
        self.classes_ = np.array([int(cls)])

    def predict_proba(self, X):
        return np.ones((len(X), 1), dtype=np.float64)


def _fit_probe(X_train, y_train, config):
    """Fit one logistic-regression probe on a set of rows. class_weight + C from config."""
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(y_train)) < 2:
        return _ConstantClf(int(np.unique(y_train)[0]))
    clf = LogisticRegression(max_iter=int(config.max_iter), C=float(config.C),
                             class_weight=(config.class_weight or None))
    clf.fit(X_train, y_train)
    return clf


def _predict_proba_full(clf, X, n_classes: int):
    """predict_proba expanded to the full [n, n_classes] space (zeros for unseen classes)."""
    proba = clf.predict_proba(X)
    full = np.zeros((len(X), n_classes), dtype=np.float64)
    for j, c in enumerate(clf.classes_):
        ci = int(c)
        if 0 <= ci < n_classes:
            full[:, ci] = proba[:, j]
    return full


def _resolve_folds(seg_ids: List[str], df_all, folds: Dict[str, int]
                   ) -> Tuple[Dict[str, int], List[int]]:
    """Map EVERY target seg_id → a fold index (participant-pure; ported from baselines)."""
    part_of: Dict[str, str] = {
        str(r.get('segment_id')): str(r.get('participant_id'))
        for _, r in df_all.iterrows()
    }
    fold_list = sorted({int(v) for v in folds.values()}) or [0]
    p2f: Dict[str, int] = {}
    for sid, fi in folds.items():
        p = part_of.get(str(sid))
        if p is not None:
            p2f.setdefault(p, int(fi))
    fold_of: Dict[str, int] = {}
    orphan_parts: List[str] = []
    for s in seg_ids:
        if s in folds:
            fold_of[s] = int(folds[s])
        else:
            p = part_of.get(s)
            if p in p2f:
                fold_of[s] = p2f[p]
            else:
                orphan_parts.append(p)
    for i, p in enumerate(sorted({x for x in orphan_parts if x is not None})):
        p2f[p] = fold_list[i % len(fold_list)]
    for s in seg_ids:
        if s not in fold_of:
            fold_of[s] = p2f.get(part_of.get(s), fold_list[0])
    return fold_of, fold_list


def _iter_folds(seg_ids: List[str], fold_of: Dict[str, int], fold_list: List[int]):
    """Yield (fold_idx, train_ids, test_ids) for the grouped CV."""
    sid_set = list(seg_ids)
    for f in fold_list:
        test = [s for s in sid_set if fold_of[s] == f]
        if not test:
            continue
        train = [s for s in sid_set if fold_of[s] != f]
        yield f, train, test


# ===========================================================================
# Per-rater label extraction + ensemble (ported from rater_distill.py)
# ===========================================================================
def _vote_stage(rv: dict) -> Optional[int]:
    """One rater's VAAMR ballot: 0..4 (CODED), -1 (ABSTAIN), or None (ERROR/unparseable)."""
    vote = rv.get('vote')
    if vote == 'ABSTAIN':
        return -1
    if vote == 'CODED':
        return rv.get('stage')
    if vote is None and rv.get('stage') is not None:
        return rv.get('stage')
    return None


def _raters_present(df_all, configured: Tuple[str, ...]) -> List[str]:
    """The distinct rater ids that actually appear in participant ``rater_votes`` ballots,
    intersected with the configured raters (or all present when none configured)."""
    present: set = set()
    for _, r in df_all.iterrows():
        if str(r.get('speaker', '') or '') != 'participant':
            continue
        rv_raw = r.get('rater_votes')
        if not isinstance(rv_raw, str) or not rv_raw.strip() or rv_raw.strip().lower() == 'nan':
            continue
        try:
            ballots = json.loads(rv_raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(ballots, list):
            continue
        for rv in ballots:
            if isinstance(rv, dict) and rv.get('rater'):
                present.add(rv['rater'])
    if configured:
        ordered = [r for r in configured if r in present]
        return ordered or sorted(present)
    return sorted(present)


def participant_rater_labels(df_all, raters: List[str], n_classes: int = 6
                             ) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    """Return (seg_ids, per_rater_label). ABSTAIN → class n_classes-1; ERROR → dropped."""
    no_code = n_classes - 1
    seg_ids: List[str] = []
    per_rater: Dict[str, Dict[str, int]] = {r: {} for r in raters}
    rater_set = set(raters)
    for _, r in df_all.iterrows():
        if str(r.get('speaker', '') or '') != 'participant':
            continue
        sid = str(r.get('segment_id'))
        rv_raw = r.get('rater_votes')
        if not isinstance(rv_raw, str) or not rv_raw.strip() or rv_raw.strip().lower() == 'nan':
            continue
        try:
            ballots = json.loads(rv_raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(ballots, list):
            continue
        seg_ids.append(sid)
        for rv in ballots:
            if not isinstance(rv, dict):
                continue
            rid = rv.get('rater')
            if rid not in rater_set:
                continue
            st = _vote_stage(rv)
            if st is None:
                continue
            per_rater[rid][sid] = no_code if st < 0 else int(st)
    return seg_ids, per_rater


def _fold_probe_proba(embeddings, train_ids, label_of, test_ids, config, n_classes):
    """Fit one balanced probe on the train rows that HAVE a label; proba on test."""
    tr = [s for s in train_ids if s in label_of and s in embeddings]
    te = list(test_ids)
    if not tr:
        return np.full((len(te), n_classes), 1.0 / n_classes)
    clf = _fit_probe(_stack_l2(embeddings, tr), [label_of[s] for s in tr], config)
    return _predict_proba_full(clf, _stack_l2(embeddings, te), n_classes)


def per_rater_oof_proba(df_all, embeddings, folds, raters, config
                        ) -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]]]:
    """OOF predict_proba [C] per rater per segment (the probe fits done once)."""
    n_classes = int(config.n_classes)
    seg_ids, per_rater = participant_rater_labels(df_all, raters, n_classes)
    seg_ids = [s for s in seg_ids if s in embeddings]
    fold_of, fold_list = _resolve_folds(seg_ids, df_all, folds)
    proba: Dict[str, Dict[str, np.ndarray]] = {r: {} for r in raters}
    for _f, train_ids, test_ids in _iter_folds(seg_ids, fold_of, fold_list):
        for r in raters:
            P = _fold_probe_proba(embeddings, train_ids, per_rater[r], test_ids, config, n_classes)
            for s, row in zip(test_ids, P):
                proba[r][s] = row
    return seg_ids, proba


def ensemble_proba(seg_ids, proba, raters, n_classes) -> Dict[str, np.ndarray]:
    """Mean of the per-rater OOF probas per segment (rows present for every rater)."""
    out: Dict[str, np.ndarray] = {}
    for s in seg_ids:
        if not all(s in proba[r] for r in raters):
            continue
        out[s] = np.mean([proba[r][s] for r in raters], axis=0)
    return out


# ---- single-probe (A1n) OOF, the fallback when <2 raters ----
def _prepare_labeled(df_all, embeddings, n_classes) -> Tuple[List[str], Dict[str, int]]:
    """Labeled participant target set: 0..4 from final_label, plus No-code (n-1) when 6-class."""
    no_code = n_classes - 1
    label_of: Dict[str, int] = {}
    for _, r in df_all.iterrows():
        if str(r.get('speaker', '') or '') != 'participant':
            continue
        sid = str(r.get('segment_id'))
        if sid not in embeddings:
            continue
        lab = r.get('final_label')
        labeled = lab is not None and not (isinstance(lab, float) and np.isnan(lab))
        if labeled:
            label_of[sid] = int(round(float(lab)))
        elif n_classes >= 6:
            label_of[sid] = no_code
    return list(label_of.keys()), label_of


def _single_probe_oof_proba(df_all, embeddings, folds, config) -> Dict[str, np.ndarray]:
    """A1n OOF mean-proba (single probe on final_label) — the <2-rater fallback."""
    n_classes = int(config.n_classes)
    seg_ids, label_of = _prepare_labeled(df_all, embeddings, n_classes)
    fold_of, fold_list = _resolve_folds(seg_ids, df_all, folds)
    out: Dict[str, np.ndarray] = {}
    for _f, train_ids, test_ids in _iter_folds(seg_ids, fold_of, fold_list):
        P = _fold_probe_proba(embeddings, train_ids, label_of, test_ids, config, n_classes)
        for s, row in zip(test_ids, P):
            out[s] = row
    return out


# ===========================================================================
# Fitted model (for inference on new/unlabeled segments)
# ===========================================================================
@dataclass
class ProbeModel:
    """A trained probe (per-rater ensemble or single A1n) + calibration/abstention state."""
    rater_probes: List = field(default_factory=list)   # list of fitted sklearn clf (one per rater / one A1n)
    raters: List[str] = field(default_factory=list)     # rater ids aligned to rater_probes ([] for A1n)
    n_classes: int = 6
    temperature: float = 1.0
    abstain_threshold: Optional[float] = None
    abstain_per_stage: Optional[Dict[int, float]] = None

    def _ensemble_proba(self, X) -> np.ndarray:
        """Mean predict_proba across the probes → [N, n_classes]."""
        Xn = X if isinstance(X, np.ndarray) else np.asarray(X, dtype=np.float64)
        probs = [_predict_proba_full(clf, Xn, self.n_classes) for clf in self.rater_probes]
        return np.mean(probs, axis=0)

    def predict(self, embeddings_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (pred[0..n-1], conf, abstain[bool], stage_mixture[N,5]).

        ``conf`` is the calibrated max-softmax confidence; class n-1 ("No code") or a
        sub-floor confidence sets ``abstain=True`` (defer to the LLM). ``stage_mixture`` is
        the ensemble-averaged 5-stage posterior (the No-code mass dropped, renormalized).
        """
        from sklearn.preprocessing import normalize
        X = normalize(np.asarray(embeddings_matrix, dtype=np.float64), norm='l2', axis=1)
        soft = self._ensemble_proba(X)                      # [N, C]
        pred = soft.argmax(axis=1).astype(int)
        # calibrated confidence — the SAME transform that derived the abstention floors
        # (_calibrate_abstain_floors), so a floor and the conf it gates share one scale.
        conf = _calibrated_conf(soft, self.temperature)
        # 5-stage mixture (drop No-code mass, renormalize)
        mix5 = soft[:, :5]
        denom = mix5.sum(axis=1, keepdims=True)
        mix5 = np.divide(mix5, denom, out=np.full_like(mix5, 0.2), where=denom > 0)
        abstain = np.zeros(len(pred), dtype=bool)
        no_code = self.n_classes - 1
        for i in range(len(pred)):
            if pred[i] == no_code:
                abstain[i] = True
                continue
            floor = None
            if self.abstain_per_stage:
                floor = self.abstain_per_stage.get(int(pred[i]))
            if floor is None:
                floor = self.abstain_threshold
            if floor is not None and conf[i] < float(floor):
                abstain[i] = True
        return pred, conf, abstain, mix5


# ===========================================================================
# Embedding access (reuse the project's Qwen npz cache)
# ===========================================================================
def _load_embeddings(df_all, output_dir, config) -> Dict[str, "np.ndarray"]:
    from gnn_layer.config import GnnLayerConfig
    from gnn_layer import embeddings as _emb
    from process import output_paths as _paths
    gnn_dir = _paths.gnn_model_dir(output_dir)
    if config.embedding_backend == 'openai':
        cache = os.path.join(gnn_dir, 'segment_embeddings_qwen3_8b.npz')
    else:
        cache = os.path.join(gnn_dir, 'segment_embeddings.npz')
    emb_cfg = GnnLayerConfig(
        embedding_backend=config.embedding_backend,
        embedding_base_url=config.embedding_base_url,
        embedding_model=config.embedding_model,
        use_query_prefix=config.use_query_prefix,
        embedding_batch_size=config.embedding_batch_size,
        cache_embeddings=True,
    )
    return _emb.load_or_build_segment_embeddings(df_all, emb_cfg, cache_path=cache)


# ===========================================================================
# Calibration + abstention-floor fitting (held-out)
# ===========================================================================
def _calibrated_conf(soft: np.ndarray, temperature: float) -> np.ndarray:
    """Max calibrated-softmax confidence for ensemble proba rows ``soft`` ([N, C] or [C]).

    Temperature scaling on log-proba pseudo-logits — the SINGLE definition of the probe's
    confidence, shared by training-time abstention-floor calibration
    (``_calibrate_abstain_floors``) and inference (``ProbeModel.predict``) so a floor and the
    conf it gates always live on the same scale.
    """
    from gnn_layer.classifier import calibration as _cal
    arr = np.atleast_2d(np.asarray(soft, dtype=np.float64))
    logits = np.log(np.clip(arr, 1e-9, 1.0))
    cal = _cal.apply_temperature(logits, float(temperature))
    e = np.exp(cal - cal.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).max(axis=1)


def _fit_temperature_oof(ens_proba: Dict[str, np.ndarray], final_of: Dict[str, int]) -> float:
    from gnn_layer.classifier import calibration as _cal
    L, y = [], []
    for sid, p in ens_proba.items():
        if sid in final_of:
            L.append(np.log(np.clip(p, 1e-9, 1.0)))
            y.append(int(final_of[sid]))
    if len(L) < 2:
        return 1.0
    return _cal.fit_temperature(np.stack(L), np.asarray(y, dtype=np.int64))


def _calibrate_abstain_floors(ens_proba, final_of, n_classes, target_precision,
                              temperature: float = 1.0) -> Dict[int, float]:
    """Per-stage confidence floor: the SMALLEST floor whose KEPT held-out precision ≥ target.

    Keep as many confident-correct predictions as possible — a higher floor only abstains
    more. Mirrors the GNN ``calibrate_abstain_floors`` (classifier/train.py): walk candidate
    floors ASCENDING and take the first that meets the target; if the target is unreachable
    for a stage, fall back to the floor with the best achievable precision. Confidence is the
    calibrated max-softmax (``_calibrated_conf`` with the SAME temperature ``predict`` uses),
    so the floors gate exactly the conf seen at inference. Only the real stages (0..4) get
    floors; No-code (n-1) is an explicit abstain class already.
    """
    items = [(p, int(final_of[s])) for s, p in ens_proba.items() if s in final_of]
    if not items:
        return {}
    soft = np.stack([p for p, _ in items], axis=0)
    conf = _calibrated_conf(soft, temperature)
    preds = soft.argmax(axis=1).astype(int)
    refs = np.asarray([ref for _, ref in items], dtype=int)
    rows = list(zip(preds.tolist(), conf.tolist(), refs.tolist()))

    floors: Dict[int, float] = {}
    for stage in range(min(5, n_classes)):
        cand = sorted({c for pr, c, _ in rows if pr == stage})
        chosen = None
        best = None  # (floor, precision) — highest precision seen; used iff target unreachable
        for floor in cand:
            kept = [ref == stage for pr, c, ref in rows if pr == stage and c >= floor]
            if not kept:
                continue
            prec = sum(kept) / len(kept)
            if best is None or prec > best[1]:
                best = (floor, prec)
            if prec >= target_precision:
                chosen = floor
                break
        if chosen is None and best is not None:
            chosen = best[0]
        if chosen is not None:
            floors[stage] = float(chosen)
    return floors


# ===========================================================================
# Train / evaluate / classify
# ===========================================================================
def _resolve_mode(df_all, config) -> Tuple[bool, List[str]]:
    """(use_ensemble, raters). Ensemble needs ≥2 distinct raters present; else A1n fallback."""
    if not config.ensemble:
        return False, []
    raters = _raters_present(df_all, tuple(config.raters))
    if len(raters) < 2:
        return False, raters
    return True, raters


def _has_llm_judgment(row) -> bool:
    """True if the LLM already judged this segment — a stage label OR a ballot of any kind
    (a "No code" ABSTAIN is a judgment, not a gap). Used to exclude such segments from the
    probe's scaling target (it only fills segments the LLM never balloted on)."""
    for key in ('final_label', 'primary_stage'):
        v = row.get(key)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return True
    rv = row.get('rater_votes')
    return isinstance(rv, str) and bool(rv.strip()) and rv.strip().lower() != 'nan'


def _final_label_map(df_all) -> Dict[str, int]:
    """{segment_id: int(final_label)} for labeled participant segments (0..4)."""
    out: Dict[str, int] = {}
    for _, r in df_all.iterrows():
        if str(r.get('speaker', '') or '') != 'participant':
            continue
        lab = r.get('final_label')
        if lab is None or (isinstance(lab, float) and np.isnan(lab)):
            continue
        try:
            out[str(r.get('segment_id'))] = int(round(float(lab)))
        except (TypeError, ValueError):
            continue
    return out


def _oof_ensemble_proba(df_all, emb, config) -> Dict[str, np.ndarray]:
    """Participant-grouped OOF mean-proba per segment (ensemble or A1n fallback)."""
    from analysis import grouped_cv as gcv
    use_ensemble, raters = _resolve_mode(df_all, config)
    folds = gcv.build_folds(df_all, config.validation_folds, config.seed)
    if use_ensemble:
        seg_ids, proba = per_rater_oof_proba(df_all, emb, folds, raters, config)
        return ensemble_proba(seg_ids, proba, raters, config.n_classes)
    return _single_probe_oof_proba(df_all, emb, folds, config)


def train_probe(df_all, output_dir: str, config: ProbeConfig, embeddings=None) -> ProbeModel:
    """Fit the per-rater ensemble (or A1n fallback) on the LLM/human label of record.

    Trains on participant segments with a label of record ONLY — never on prior probe
    labels (no model-distilling-model). Calibration temperature + abstention floors are fit
    on participant-grouped OOF. Persists the bundle + manifest under ``02_meta/probe/``.
    ``embeddings`` (``{segment_id: vec}``) may be supplied to bypass the encoder (tests).
    """
    import datetime
    import joblib
    from process import output_paths as _paths

    emb = embeddings if embeddings is not None else _load_embeddings(df_all, output_dir, config)
    if not emb:
        raise RuntimeError("probe: no segment embeddings available (is the embedding "
                           "backend reachable / cache present?)")
    n_classes = int(config.n_classes)
    use_ensemble, raters = _resolve_mode(df_all, config)

    rater_probes: List = []
    rater_ids: List[str] = []
    class_counts: Dict[str, Dict[int, int]] = {}
    if use_ensemble:
        seg_ids, per_rater = participant_rater_labels(df_all, raters, n_classes)
        for r in raters:
            label_of = {s: c for s, c in per_rater[r].items() if s in emb}
            if len(set(label_of.values())) < 1:
                continue
            clf = _fit_probe(_stack_l2(emb, list(label_of)),
                             [label_of[s] for s in label_of], config)
            rater_probes.append(clf)
            rater_ids.append(r)
            vals, cnts = np.unique(list(label_of.values()), return_counts=True)
            class_counts[r] = {int(k): int(v) for k, v in zip(vals, cnts)}
        provenance = 'per_rater_ensemble'
    if not rater_probes:  # A1n fallback (no/insufficient ballots)
        _ids, label_of = _prepare_labeled(df_all, emb, n_classes)
        if not label_of:
            raise RuntimeError("probe: no LLM/human-labeled participant segments to train on.")
        clf = _fit_probe(_stack_l2(emb, list(label_of)),
                         [label_of[s] for s in label_of], config)
        rater_probes = [clf]
        rater_ids = []
        vals, cnts = np.unique(list(label_of.values()), return_counts=True)
        class_counts = {'final_label': {int(k): int(v) for k, v in zip(vals, cnts)}}
        provenance = 'single_probe_a1n'

    # ---- calibration + abstention floors on held-out OOF ----
    temperature = 1.0
    floors = config.abstain_per_stage
    if config.calibrate or config.abstain_calibrate:
        ens = _oof_ensemble_proba(df_all, emb, config)
        final_of = _final_label_map(df_all)
        if config.calibrate:
            temperature = _fit_temperature_oof(ens, final_of)
        if config.abstain_calibrate:
            floors = _calibrate_abstain_floors(ens, final_of, n_classes,
                                               config.abstain_target_precision, temperature)

    model = ProbeModel(rater_probes=rater_probes, raters=rater_ids, n_classes=n_classes,
                       temperature=float(temperature),
                       abstain_threshold=config.abstain_threshold,
                       abstain_per_stage=floors)

    # ---- persist ----
    model_dir = _paths.probe_model_dir(output_dir)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, MODEL_FILENAME))
    sample = next(iter(emb.values()))
    manifest = {
        'code_version': PROBE_CODE_VERSION,
        'embedding_model': config.embedding_model,
        'embedding_backend': config.embedding_backend,
        'embedding_dim': int(np.asarray(sample).shape[0]),
        'ensemble': bool(use_ensemble and provenance == 'per_rater_ensemble'),
        'raters': rater_ids,
        'n_classes': n_classes,
        'class_weight': config.class_weight,
        'C': config.C,
        'temperature': float(temperature),
        'abstain_per_stage': floors,
        'per_rater_class_counts': class_counts,
        'training_label_provenance': provenance,
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(os.path.join(model_dir, MANIFEST_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    return model


def evaluate_probe(df_all, output_dir: str, config: ProbeConfig, embeddings=None) -> dict:
    """Participant-grouped gate: probe↔LLM κ and probe↔human κ, per-class recall, CIs.

    Writes ``06_reports/06_classifier/probe_validation.txt`` and
    ``03_analysis_data/probe/probe_gate.json``. Verdict ``ready_for_scaling`` iff
    probe↔human κ ≥ ``irr_human_band_floor`` AND no rare-stage recall < floor.
    """
    from analysis import grouped_cv as gcv
    from analysis import irr_join

    emb = embeddings if embeddings is not None else _load_embeddings(df_all, output_dir, config)
    use_ensemble, raters = _resolve_mode(df_all, config)
    ens = _oof_ensemble_proba(df_all, emb, config)
    ens_pred = {s: int(p.argmax()) for s, p in ens.items()}

    part_of = {str(r.get('segment_id')): str(r.get('participant_id'))
               for _, r in df_all.iterrows()}

    # ---- LLM axis (vs final_label over labeled segments) ----
    final_of = _final_label_map(df_all)
    lp, lr, lg = [], [], []
    for sid, ref in final_of.items():
        if sid in ens_pred:
            lp.append(ens_pred[sid]); lr.append(ref); lg.append(part_of.get(sid))
    llm_ci = gcv.kappa_cluster_ci(lp, lr, lg, seed=config.seed)
    per_class = gcv.per_class_recall_precision(list(zip(lp, lr)), PROBE_VAAMR_NAMES)
    recall_by_class = {r['class_id']: r['recall'] for r in per_class}

    # ---- human axis (vs human consensus; predicted No-code 5 → ABSTAIN -1) ----
    df_h = irr_join.populate_human_columns(df_all, output_dir)
    hp, hr, hg = [], [], []
    for _, r in df_h.iterrows():
        sid = str(r.get('segment_id'))
        if not r.get('in_human_coded_subset') or sid not in ens_pred:
            continue
        ref = r.get('human_label')
        if ref is None or (isinstance(ref, float) and np.isnan(ref)):
            continue
        pred = ens_pred[sid]
        hp.append(-1 if pred == 5 else pred); hr.append(int(ref)); hg.append(part_of.get(sid))
    human_ci = gcv.kappa_cluster_ci(hp, hr, hg, seed=config.seed)

    # ---- verdict ----
    rare_notes = []
    for c in RARE_STAGES:
        rec = recall_by_class.get(c)
        if rec is not None and rec < config.rare_stage_recall_floor:
            rare_notes.append(f"{PROBE_VAAMR_NAMES[c]} recall {rec:.2f} < {config.rare_stage_recall_floor}")
    human_k = human_ci.get('point')
    rare_ok = not rare_notes
    ready = bool(human_k is not None and human_k >= config.irr_human_band_floor and rare_ok)

    verdict = {
        'ready_for_scaling': ready,
        'mode': 'per_rater_ensemble' if use_ensemble else 'single_probe_a1n',
        'raters': raters if use_ensemble else [],
        'probe_human_kappa': human_k,
        'probe_human_ci': [human_ci.get('lo'), human_ci.get('hi')],
        'probe_human_n': human_ci.get('n'),
        'probe_llm_kappa': llm_ci.get('point'),
        'probe_llm_ci': [llm_ci.get('lo'), llm_ci.get('hi')],
        'probe_llm_n': llm_ci.get('n'),
        'per_class_recall': {PROBE_VAAMR_NAMES[k]: v for k, v in recall_by_class.items()},
        'rare_ok': rare_ok,
        'rare_stage_notes': rare_notes,
        'irr_human_band_floor': config.irr_human_band_floor,
        'rare_stage_recall_floor': config.rare_stage_recall_floor,
        'n_classes': config.n_classes,
    }
    _write_probe_gate(verdict, output_dir)
    _write_validation_report(verdict, per_class, output_dir)
    return verdict


def classify_with_probe(df_all, output_dir: str, config: ProbeConfig,
                        only_session_ids=None, embeddings=None) -> int:
    """LLM-free label participant segments WITHOUT a label of record; write probe_labels.

    Targets participant segments whose ``final_label`` AND ``primary_stage`` are both null
    (the scaling target). Applies abstention. Returns the number of overlay rows written.
    LLM-labeled segments are never touched.
    """
    from process import classifications_io as _cio
    from classification_tools.data_structures import Segment

    model = load_probe_model(output_dir)
    if model is None:
        raise RuntimeError("probe: no persisted model — run `qra probe train` first.")
    emb = embeddings if embeddings is not None else _load_embeddings(df_all, output_dir, config)

    targets = []
    for _, r in df_all.iterrows():
        if str(r.get('speaker', '') or '') != 'participant':
            continue
        sid = str(r.get('segment_id'))
        if sid not in emb:
            continue
        # Skip any segment the LLM already JUDGED: a stage label OR a "No code" ABSTAIN
        # (No-code carries ballots but a null final_label — it is a judgment, not a gap).
        # The scaling target is segments the LLM never balloted on (genuinely unlabeled).
        if _has_llm_judgment(r):
            continue
        if only_session_ids is not None and str(r.get('session_id')) not in set(map(str, only_session_ids)):
            continue
        targets.append(sid)
    if not targets:
        return 0

    X = np.stack([np.asarray(emb[s]) for s in targets], axis=0)
    pred, conf, abstain, _mix = model.predict(X)
    segs = []
    for i, sid in enumerate(targets):
        s = Segment(segment_id=sid)
        s.probe_pred = int(pred[i])
        s.probe_conf = float(conf[i])
        s.probe_abstain = bool(abstain[i])
        segs.append(s)
    _cio.merge_probe_overlay(output_dir, segs)
    return len(segs)


# ===========================================================================
# Gate + model persistence helpers
# ===========================================================================
def _write_probe_gate(verdict: dict, output_dir: str) -> str:
    from process import output_paths as _paths
    d = _paths.probe_data_dir(output_dir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, GATE_FILENAME)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(verdict, f, indent=2)
    return path


def read_probe_gate(output_dir: str) -> Optional[dict]:
    from process import output_paths as _paths
    path = os.path.join(_paths.probe_data_dir(output_dir), GATE_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def probe_gate_ready(output_dir: str) -> bool:
    """True only if a persisted probe gate exists AND reports ready_for_scaling."""
    v = read_probe_gate(output_dir)
    return bool(v and v.get('ready_for_scaling'))


def load_probe_model(output_dir: str) -> Optional[ProbeModel]:
    import joblib
    from process import output_paths as _paths
    path = os.path.join(_paths.probe_model_dir(output_dir), MODEL_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def _write_validation_report(verdict: dict, per_class: List[dict], output_dir: str) -> str:
    from process import output_paths as _paths
    d = _paths.reports_classifier_dir(output_dir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, 'probe_validation.txt')

    def _fmt(x, nd=3):
        return f"{x:.{nd}f}" if isinstance(x, (int, float)) else 'n/a'

    lines = [
        "PROBE SCALER — RELIABILITY GATE (participant-grouped, leakage-free)",
        "=" * 70,
        "",
        f"Mode: {verdict.get('mode')}   raters: {', '.join(verdict.get('raters') or []) or '(A1n)'}",
        "",
        "The multi-run LLM consensus is the label of record (human-level, κ≈0.537 vs human;",
        "human↔human α≈0.33–0.52). This probe is an ASSISTIVE, gated, abstention-aware",
        "pre-labeler that fills UNLABELED segments only; it ranks BELOW the LLM and never",
        "overrides it.",
        "",
        "HEADLINE κ (point [95% cluster-bootstrap CI], n):",
        f"  probe ↔ human consensus : {_fmt(verdict.get('probe_human_kappa'))} "
        f"[{_fmt(verdict.get('probe_human_ci',[None,None])[0])}, "
        f"{_fmt(verdict.get('probe_human_ci',[None,None])[1])}]  n={verdict.get('probe_human_n')}",
        f"  probe ↔ LLM consensus   : {_fmt(verdict.get('probe_llm_kappa'))} "
        f"[{_fmt(verdict.get('probe_llm_ci',[None,None])[0])}, "
        f"{_fmt(verdict.get('probe_llm_ci',[None,None])[1])}]  n={verdict.get('probe_llm_n')}",
        "",
        "Per-stage recall/precision (LLM axis):",
    ]
    for r in per_class:
        lines.append(f"  {str(r['class_name']):<14} support={r['support']:>3}  "
                     f"recall={_fmt(r['recall'],2)}  precision={_fmt(r['precision'],2)}")
    lines += [
        "",
        f"Gate floors: probe↔human κ ≥ {verdict.get('irr_human_band_floor')}  AND  "
        f"rare-stage recall ≥ {verdict.get('rare_stage_recall_floor')} (Avoidance, Metacognition)",
        f"Rare-stage check: {'OK' if verdict.get('rare_ok') else 'FAIL — ' + '; '.join(verdict.get('rare_stage_notes', []))}",
        "",
        f"VERDICT — ready for LLM-free scaling: {'YES' if verdict.get('ready_for_scaling') else 'NO'}",
        "",
        "Interpretation: a NO does not mean the probe is broken — at pilot scale the data",
        "ceiling (human↔human α≈0.33–0.52, rare-stage scarcity) bounds any VAAMR labeler.",
        "Re-gate as labeled participants accumulate (methodology §8.6).",
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    return path
