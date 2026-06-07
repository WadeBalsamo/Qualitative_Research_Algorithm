"""
experiments/classification_scaler/ordinal_twostage.py
-----------------------------------------------------
Label-STRUCTURE arm for the cheap-VAAMR-classifier hunt.

Baseline to beat: A1n = Qwen class-weighted *flat* 6-class linear probe
(human kappa=0.365 / LLM-axis grouped kappa(205)=0.283). The flat probe ignores two
facts about the label space:
  (1) the 5 VAAMR stages are an ORDINAL arc  Vigilance<Avoidance<AttnReg<Metacog<Reappraisal
  (2) "No code" is categorically different from the inter-stage distinctions.

Two levers, both vs the flat probe on the SAME participant-grouped folds + Qwen feats:
  (a) ORDINAL stager  : mord LogisticAT / LogisticIT, Frank-Hall cumulative
                        "stage>=k" binaries, or Ridge regression of the stage
                        coordinate + threshold.
  (b) TWO-STAGE       : a binary No-code-vs-VAAMR gate, then a 5-class stager
                        applied only to gate-predicted-VAAMR segments.

Factorial: {multinomial, mord_at, mord_it, ordbin, ridge}
           x {5-class (no gate) , 6-class (No-code gate)}.
The 5-class family isolates the pure ORDINAL effect (LLM axis over the labeled 205);
the 6-class family adds the No-code gate and isolates the TWO-STAGE / decoupling effect
(the human axis, where No-code=-1 is a real 6th category).

All features L2-normalized; every native-weightable estimator is class-weighted;
mord (no weight support) is balanced by resampling. Scoring + folds + embeddings are
REUSED from experiments.gnn_reliability.{harness,baselines} (identical to A1n), so the
kappa is directly comparable to the ledger / 06b_irr_report.txt. write_ledger=False.
"""

import sys
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import normalize

from experiments.gnn_reliability import harness as H
from experiments.gnn_reliability import baselines as B
from gnn_layer.config import GnnLayerConfig

ABS = '/home/wisgood/qra/Qualitative_Research_Algorithm/data/Meta'
SEED = 42

try:
    import mord
    _HAVE_MORD = True
except Exception:
    _HAVE_MORD = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _balanced_resample(X, y, seed):
    """Oversample each class with replacement to the max class count (mord has no
    class_weight; this is the resampling equivalent)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()
    idx_out = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        take = rng.choice(idx_c, size=target, replace=True)
        idx_out.append(take)
    idx_out = np.concatenate(idx_out)
    rng.shuffle(idx_out)
    return X[idx_out], y[idx_out]


def _balanced_sample_weight(y):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    freq = {c: n for c, n in zip(classes, counts)}
    n, k = len(y), len(classes)
    return np.array([n / (k * freq[v]) for v in y], dtype=float)


def fit_predict_stager(method, Xtr, ytr, Xte, seed):
    """Return integer stage predictions (0..4) for Xte. ``method`` selects the stager.

    multinomial : class-weighted softmax logreg (the flat 5-class baseline stager)
    mord_at     : mord LogisticAT (all-threshold ordinal) on a balanced resample
    mord_it     : mord LogisticIT (immediate-threshold ordinal) on a balanced resample
    ordbin      : Frank-Hall cumulative  K-1  class-weighted "stage>=k" binaries
    ridge       : class-weighted Ridge regression of the stage coordinate + round/clip
    """
    Xtr = np.asarray(Xtr); ytr = np.asarray(ytr); Xte = np.asarray(Xte)
    uniq = np.unique(ytr)
    if len(uniq) < 2:                       # degenerate train fold
        return np.full(len(Xte), int(uniq[0]), dtype=int)

    if method == 'multinomial':
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced')
        clf.fit(Xtr, ytr)
        return clf.predict(Xte).astype(int)

    if method in ('mord_at', 'mord_it'):
        if not _HAVE_MORD:
            raise RuntimeError("mord not installed")
        Xb, yb = _balanced_resample(Xtr, ytr, seed)
        est = mord.LogisticAT(alpha=1.0) if method == 'mord_at' else mord.LogisticIT(alpha=1.0)
        est.fit(Xb, yb)
        return np.clip(est.predict(Xte), 0, 4).astype(int)

    if method == 'ordbin':
        # Frank-Hall: P(y>=k) for k=1..4 via class-weighted binaries; reconstruct P(y=k).
        ks = [1, 2, 3, 4]
        Pge = np.zeros((len(Xte), 5))
        Pge[:, 0] = 1.0                      # P(y>=0)=1
        for k in ks:
            yk = (ytr >= k).astype(int)
            if len(np.unique(yk)) < 2:
                Pge[:, k] = float(yk[0])     # constant
                continue
            clf = LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced')
            clf.fit(Xtr, yk)
            j = list(clf.classes_).index(1)
            Pge[:, k] = clf.predict_proba(Xte)[:, j]
        # P(y=k) = P(>=k) - P(>=k+1) ; last stage uses P(>=4)
        Peq = np.zeros((len(Xte), 5))
        for k in range(4):
            Peq[:, k] = Pge[:, k] - Pge[:, k + 1]
        Peq[:, 4] = Pge[:, 4]
        Peq = np.clip(Peq, 0.0, None)
        return Peq.argmax(axis=1).astype(int)

    if method == 'ridge':
        sw = _balanced_sample_weight(ytr)
        reg = Ridge(alpha=1.0)
        reg.fit(Xtr, ytr.astype(float), sample_weight=sw)
        yhat = reg.predict(Xte)
        return np.clip(np.round(yhat), 0, 4).astype(int)

    raise ValueError(f"unknown stager method {method!r}")


# ---------------------------------------------------------------------------
# OOF producers (both honor the A1n contract: {segment_id: pred_int})
# ---------------------------------------------------------------------------
def oof_five_class(df, emb, folds, method):
    """5-class family: no gate. Stager trained on labeled rows, OOF over the 205."""
    cfg5 = GnnLayerConfig(vaamr_n_classes=5, vaamr_class_balance=True)
    seg_ids, labels, _ = B._prepare_labeled(df, emb, cfg5)
    label_of = dict(zip(seg_ids, labels))
    fold_of, fold_list = B._resolve_folds(seg_ids, df, folds)
    preds = {}
    for f, train_ids, test_ids in B._iter_folds(seg_ids, fold_of, fold_list):
        Xtr = B._stack_l2(emb, train_ids); ytr = [label_of[s] for s in train_ids]
        Xte = B._stack_l2(emb, test_ids)
        yhat = fit_predict_stager(method, Xtr, ytr, Xte, SEED + 1 + f)
        for s, p in zip(test_ids, yhat):
            preds[s] = int(p)
    return preds


def oof_two_stage(df, emb, folds, method, gate_thr=0.5):
    """6-class family: binary No-code gate -> 5-class stager on gate-predicted-VAAMR.

    seg_ids span labeled (0..4) + No-code (class 5) participant rows (same as A1n's
    6-class _prepare_labeled). The gate is trained on ALL train rows (VAAMR=1 vs
    No-code=0); the stager only on the train VAAMR rows. A test row gated below
    ``gate_thr`` is emitted as 5 (No-code); else it gets the stager's stage.
    """
    cfg6 = GnnLayerConfig(vaamr_n_classes=6, vaamr_class_balance=True)
    seg_ids, labels, _ = B._prepare_labeled(df, emb, cfg6)
    label_of = dict(zip(seg_ids, labels))
    fold_of, fold_list = B._resolve_folds(seg_ids, df, folds)
    preds = {}
    for f, train_ids, test_ids in B._iter_folds(seg_ids, fold_of, fold_list):
        # --- gate: VAAMR(1) vs No-code(0), class-weighted ---
        Xtr = B._stack_l2(emb, train_ids)
        y_gate = np.array([1 if label_of[s] < 5 else 0 for s in train_ids])
        Xte = B._stack_l2(emb, test_ids)
        if len(np.unique(y_gate)) < 2:
            p_vaamr = np.full(len(test_ids), float(y_gate[0]))
        else:
            g = LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced')
            g.fit(Xtr, y_gate)
            jv = list(g.classes_).index(1)
            p_vaamr = g.predict_proba(Xte)[:, jv]
        # --- stager: trained on train VAAMR rows only ---
        v_train = [s for s in train_ids if label_of[s] < 5]
        Xv = B._stack_l2(emb, v_train); yv = [label_of[s] for s in v_train]
        stage_pred = fit_predict_stager(method, Xv, yv, Xte, SEED + 1 + f)
        for i, s in enumerate(test_ids):
            preds[s] = int(stage_pred[i]) if p_vaamr[i] >= gate_thr else 5
    return preds


# ---------------------------------------------------------------------------
# adjacency diagnostics (ordinal claim: fewer FAR-off stage errors)
# ---------------------------------------------------------------------------
def adjacency_stats(oof, df):
    """Over the labeled 205: exact/adjacent/far rates + MAE for STAGE-vs-STAGE
    predictions (pred 0..4), plus the count of gate-misfires (true VAAMR -> pred
    No-code=5)."""
    part = df[df['speaker'] == 'participant']
    final_of = {str(r['segment_id']): int(r['final_label'])
                for _, r in part.iterrows() if not _isnan(r.get('final_label'))}
    exact = adj = far = gate_miss = 0
    abserr = []
    for sid, t in final_of.items():
        if sid not in oof:
            continue
        p = oof[sid]
        if p == 5:                          # No-code predicted on a truly-labeled seg
            gate_miss += 1
            continue
        d = abs(p - t)
        abserr.append(d)
        if d == 0:
            exact += 1
        elif d == 1:
            adj += 1
        else:
            far += 1
    n = exact + adj + far
    return {
        'n_stage': n, 'gate_miss': gate_miss,
        'exact': exact / n if n else 0.0,
        'adjacent': adj / n if n else 0.0,
        'far': far / n if n else 0.0,
        'mae': float(np.mean(abserr)) if abserr else float('nan'),
    }


def _isnan(v):
    try:
        return v is None or np.isnan(float(v))
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '  -  '
    return f"{v:.3f}"


def main():
    df = H.load_corpus(ABS)
    folds = H.build_folds(df, seed=SEED, verbose=False)
    emb = H.get_embeddings(df, 'qwen', ABS)

    methods = ['multinomial', 'mord_at', 'mord_it', 'ordbin', 'ridge']
    rows = []

    # ---- reference anchors ----
    specs = []
    # 5-class family (pure ordinal isolation; n_classes=5 scoring)
    for m in methods:
        specs.append((f"f5_{m}", 5, lambda df, emb, folds, m=m: oof_five_class(df, emb, folds, m)))
    # 6-class family (No-code gate; two-stage; n_classes=6 scoring)
    specs.append(("f6_joint(A1n)", 6,
                  lambda df, emb, folds: B.run_linear_probe(
                      df, emb, folds, GnnLayerConfig(vaamr_n_classes=6, vaamr_class_balance=True))))
    for m in methods:
        specs.append((f"ts_{m}", 6, lambda df, emb, folds, m=m: oof_two_stage(df, emb, folds, m)))

    print(f"{'variant':<18} {'nC':>2} | {'LLM k205':>8} {'[lo':>6} {'hi]':>6} "
          f"{'k76':>6} | {'HUM k':>6} {'[lo':>6} {'hi]':>6} hN | "
          f"{'exact':>5} {'adj':>5} {'far':>5} {'MAE':>5} gMiss")
    print('-' * 120)

    import time
    results = {}
    for name, nC, fn in specs:
        _t0 = time.time()
        oof = fn(df, emb, folds)
        res = H.score_arm(name, oof, df, ABS, n_classes=nC,
                          meta={'embedding': 'qwen', 'method': name, 'seed': SEED,
                                'branch': 'scaler/ordinal-twostage'},
                          write_ledger=False)
        adjs = adjacency_stats(oof, df)
        print(f"# {name} done in {time.time()-_t0:.1f}s", flush=True)
        results[name] = (res, adjs)
        llm, hum = res['llm_axis'], res['human_axis']
        print(f"{name:<18} {nC:>2} | {_fmt(llm['cohen_kappa_205']):>8} "
              f"{_fmt(llm['ci95'][0]):>6} {_fmt(llm['ci95'][1]):>6} {_fmt(llm['cohen_kappa_76']):>6} | "
              f"{_fmt(hum['cohen_kappa']):>6} {_fmt(hum['ci95'][0]):>6} {_fmt(hum['ci95'][1]):>6} "
              f"{hum['n']:>2} | {_fmt(adjs['exact']):>5} {_fmt(adjs['adjacent']):>5} "
              f"{_fmt(adjs['far']):>5} {_fmt(adjs['mae']):>5} {adjs['gate_miss']:>4}")

    # ---- per-class recall table (LLM axis) ----
    print('\nPer-class recall (LLM axis, stages 0..4):')
    print(f"{'variant':<18} {'Vig':>5} {'Avo':>5} {'Attn':>5} {'Meta':>5} {'Reap':>5}")
    for name, _nC, _fn in specs:
        res, _ = results[name]
        rc = {r['class_id']: r['recall'] for r in res['llm_axis']['per_class']}
        print(f"{name:<18} " + " ".join(_fmt(rc.get(i)).rjust(5) for i in range(5)))

    return results


if __name__ == '__main__':
    main()
