"""
experiments/gnn_reliability/rater_distill.py  (scaler/ensemble branch — NOT committed)
--------------------------------------------------------------------------------------
Per-rater distillation / ensembling vs the collapsed consensus probe (A1n).

The shipped consensus collapses the 3 LLM raters (gemma-4-31b, nemotron-3-nano-30b,
qwen3-next-80b) into ``final_label`` by stage-majority vote.  Their *disagreement
structure* may carry signal the majority throws away.  This module reuses the GNN
reliability harness (identical Qwen embeddings, identical participant-grouped folds,
identical two-axis scorer) and asks: does modeling each rater separately and then
ensembling beat the single probe trained on the collapsed majority?

Variants (all 6-class, class-weighted, L2-normed Qwen features, same folds as A1n):
  A1n            single LogReg probe on final_label (the harness baseline; reproduced)
  per-rater_R    single LogReg probe on rater R's own labels (diagnostic)
  ens_majority   3 per-rater probes, ensemble by argmax majority vote
  ens_softavg    3 per-rater probes, ensemble by mean predict_proba then argmax
  ens_softavg_w  ens_softavg with raters weighted by human-subset kappa (exploratory; leak-flagged)
  mlp_hard       MLP on the hard argmax of the rater distribution (CE)        [arch control]
  mlp_soft_kl    MLP on the soft per-segment rater distribution (KL)          [the (b) variant]

ABSTAIN -> No-code class (n_classes-1 == 5); ERROR/unparseable -> MISSING for that
rater (dropped from that rater's training rows, never forced to No-code).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.gnn_reliability import baselines as B

RATERS = ['google/gemma-4-31b', 'nvidia/nemotron-3-nano-30b', 'qwen/qwen3-next-80b']


# ---------------------------------------------------------------------------
# per-rater label extraction (reuses irr_analysis._rater_vote_stage semantics)
# ---------------------------------------------------------------------------
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


def participant_rater_labels(df_all, n_classes: int = 6
                             ) -> Tuple[List[str], Dict[str, Dict[str, int]], Dict[str, np.ndarray]]:
    """Return (seg_ids, per_rater_label, soft_target).

    seg_ids        : every participant segment that has a rater_votes row.
    per_rater_label: {rater: {seg_id: int_label}} with ABSTAIN->n_classes-1, ERROR dropped.
    soft_target    : {seg_id: np.ndarray[n_classes]} empirical distribution over the
                     non-missing raters' one-hot labels (rows summing to 1; all-missing -> None).
    """
    no_code = n_classes - 1
    seg_ids: List[str] = []
    per_rater: Dict[str, Dict[str, int]] = {r: {} for r in RATERS}
    soft: Dict[str, np.ndarray] = {}
    for _, r in df_all.iterrows():
        if str(r.get('speaker', '') or '') != 'participant':
            continue
        sid = str(r.get('segment_id'))
        rv_raw = r.get('rater_votes')
        if not isinstance(rv_raw, str) or not rv_raw.strip() or rv_raw.strip().lower() == 'nan':
            continue
        seg_ids.append(sid)
        dist = np.zeros(n_classes, dtype=np.float64)
        n_present = 0
        for rv in json.loads(rv_raw):
            rid = rv.get('rater')
            if rid not in per_rater:
                continue
            st = _vote_stage(rv)
            if st is None:
                continue  # ERROR -> missing for this rater
            cls = no_code if st < 0 else int(st)
            per_rater[rid][sid] = cls
            dist[cls] += 1.0
            n_present += 1
        soft[sid] = (dist / n_present) if n_present else None
    return seg_ids, per_rater, soft


# ---------------------------------------------------------------------------
# per-rater probes + ensembles
# ---------------------------------------------------------------------------
def _fold_probe_proba(embeddings, train_ids, label_of, test_ids, cfg, n_classes):
    """Fit one balanced LogReg probe on the train rows that HAVE a label; proba on test."""
    tr = [s for s in train_ids if s in label_of and s in embeddings]
    te = list(test_ids)
    if not tr:
        return np.full((len(te), n_classes), 1.0 / n_classes)
    scaler, clf = B._fit_probe(B._stack_l2(embeddings, tr),
                               [label_of[s] for s in tr], cfg, n_classes)
    return B._predict_proba_full(scaler, clf, B._stack_l2(embeddings, te), n_classes)


def run_per_rater(df_all, embeddings, folds, cfg, rater: str) -> Dict[str, int]:
    """OOF argmax of a single probe distilled from one rater's labels."""
    n_classes = int(getattr(cfg, 'vaamr_n_classes', 6))
    seg_ids, per_rater, _ = participant_rater_labels(df_all, n_classes)
    seg_ids = [s for s in seg_ids if s in embeddings]
    fold_of, fold_list = B._resolve_folds(seg_ids, df_all, folds)
    label_of = per_rater[rater]
    preds: Dict[str, int] = {}
    for _f, train_ids, test_ids in B._iter_folds(seg_ids, fold_of, fold_list):
        proba = _fold_probe_proba(embeddings, train_ids, label_of, test_ids, cfg, n_classes)
        for s, row in zip(test_ids, proba):
            preds[s] = int(np.argmax(row))
    return preds


def per_rater_oof_proba(df_all, embeddings, folds, cfg
                        ) -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]]]:
    """OOF predict_proba [C] per rater per segment — the 15 probe fits done ONCE.

    Returns (ordered seg_ids, {rater: {seg_id: proba_vec}}).  Every ensemble variant
    is a cheap reduction over this, so we never refit the same probes per-variant.
    """
    n_classes = int(getattr(cfg, 'vaamr_n_classes', 6))
    seg_ids, per_rater, _ = participant_rater_labels(df_all, n_classes)
    seg_ids = [s for s in seg_ids if s in embeddings]
    fold_of, fold_list = B._resolve_folds(seg_ids, df_all, folds)
    proba: Dict[str, Dict[str, np.ndarray]] = {r: {} for r in RATERS}
    for _f, train_ids, test_ids in B._iter_folds(seg_ids, fold_of, fold_list):
        for r in RATERS:
            P = _fold_probe_proba(embeddings, train_ids, per_rater[r], test_ids, cfg, n_classes)
            for s, row in zip(test_ids, P):
                proba[r][s] = row
    return seg_ids, proba


def ensemble_from_proba(seg_ids, proba, n_classes, mode: str,
                        weights: Optional[Dict[str, float]] = None) -> Dict[str, int]:
    """Reduce cached per-rater OOF probas to one label/seg. mode='softavg'|'majority'."""
    w = {r: float((weights or {}).get(r, 1.0)) for r in RATERS}
    wsum = sum(w.values()) or 1.0
    preds: Dict[str, int] = {}
    for s in seg_ids:
        if not all(s in proba[r] for r in RATERS):
            continue
        soft = sum(w[r] * proba[r][s] for r in RATERS) / wsum
        if mode == 'softavg':
            preds[s] = int(np.argmax(soft))
        elif mode == 'majority':
            wcounts = np.zeros(n_classes)
            for r in RATERS:
                wcounts[int(np.argmax(proba[r][s]))] += w[r]
            top = np.flatnonzero(wcounts == wcounts.max())
            preds[s] = int(top[0]) if len(top) == 1 else int(np.argmax(soft))
        else:
            raise ValueError(mode)
    return preds


def per_rater_argmax(seg_ids, proba, rater) -> Dict[str, int]:
    """Single rater's OOF argmax (diagnostic), from the cached probas."""
    return {s: int(np.argmax(proba[rater][s])) for s in seg_ids if s in proba[rater]}


def run_ensemble(df_all, embeddings, folds, cfg, mode: str,
                 weights: Optional[Dict[str, float]] = None) -> Dict[str, int]:
    """Convenience: fit + reduce in one call (used by ad-hoc callers)."""
    n_classes = int(getattr(cfg, 'vaamr_n_classes', 6))
    seg_ids, proba = per_rater_oof_proba(df_all, embeddings, folds, cfg)
    return ensemble_from_proba(seg_ids, proba, n_classes, mode, weights)


# ---------------------------------------------------------------------------
# soft-target MLP (the (b) variant) + hard-target arch control
# ---------------------------------------------------------------------------
def _run_mlp(df_all, embeddings, folds, cfg, soft_loss: bool,
             hidden: int = 128, epochs: int = 200, lr: float = 1e-3,
             wd: float = 1e-4, dropout: float = 0.3, seed: int = 42) -> Dict[str, int]:
    """MLP distilled from the per-segment rater distribution.

    soft_loss=True  -> KL(target || pred) on the empirical rater distribution.
    soft_loss=False -> cross-entropy on the hard argmax of that distribution (arch control).
    Sample-weighted by inverse frequency of the argmax class (mirrors 'balanced').
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fnn
    torch.set_num_threads(min(8, os.cpu_count() or 8))

    n_classes = int(getattr(cfg, 'vaamr_n_classes', 6))
    seg_ids, _per, soft = participant_rater_labels(df_all, n_classes)
    seg_ids = [s for s in seg_ids if soft.get(s) is not None and s in embeddings]
    fold_of, fold_list = B._resolve_folds(seg_ids, df_all, folds)
    X_all = {s: v for s, v in zip(seg_ids, B._stack_l2(embeddings, seg_ids))}
    D = next(iter(X_all.values())).shape[0]

    def _make():
        torch.manual_seed(seed)
        np.random.seed(seed)
        return nn.Sequential(nn.Linear(D, hidden), nn.ReLU(), nn.Dropout(dropout),
                             nn.Linear(hidden, n_classes))

    preds: Dict[str, int] = {}
    for _f, train_ids, test_ids in B._iter_folds(seg_ids, fold_of, fold_list):
        Xtr = torch.tensor(np.stack([X_all[s] for s in train_ids]), dtype=torch.float32)
        T = torch.tensor(np.stack([soft[s] for s in train_ids]), dtype=torch.float32)
        hard = T.argmax(dim=1)
        freq = torch.bincount(hard, minlength=n_classes).float().clamp(min=1.0)
        sw = (1.0 / freq)[hard]
        sw = sw / sw.mean()
        model = _make()
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        model.train()
        for _ep in range(epochs):
            opt.zero_grad()
            logits = model(Xtr)
            logp = Fnn.log_softmax(logits, dim=1)
            if soft_loss:
                loss = (Fnn.kl_div(logp, T, reduction='none').sum(1) * sw).mean()
            else:
                loss = (Fnn.nll_loss(logp, hard, reduction='none') * sw).mean()
            loss.backward()
            opt.step()
        model.eval()
        Xte = torch.tensor(np.stack([X_all[s] for s in test_ids]), dtype=torch.float32)
        with torch.no_grad():
            p = model(Xte).argmax(1).numpy()
        for s, c in zip(test_ids, p):
            preds[s] = int(c)
    return preds


def run_mlp_soft(df_all, embeddings, folds, cfg, **kw):
    return _run_mlp(df_all, embeddings, folds, cfg, soft_loss=True, **kw)


def run_mlp_hard(df_all, embeddings, folds, cfg, **kw):
    return _run_mlp(df_all, embeddings, folds, cfg, soft_loss=False, **kw)
