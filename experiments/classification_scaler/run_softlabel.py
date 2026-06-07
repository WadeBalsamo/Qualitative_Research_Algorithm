"""experiments/gnn_reliability/run_softlabel.py
-------------------------------------------------
SOFT-LABEL DISTILLATION arm for the VAAMR LLM-free scaler.

Baseline to beat = A1n (Qwen class-weighted 6-class linear probe):
    human kappa = 0.365 [0.228, 0.513] , LLM-axis kappa(205) = 0.283 [0.203, 0.346]

Lever: train on the LLM's SOFT multi-run ballot mixture (build_soft_targets) instead
of the argmax final_label. Three variants on IDENTICAL participant-grouped folds:

  (a) MLP-KL   : small MLP, KL / soft-CE to the soft mixture target.
  (b) MLP-CE   : same MLP, hard cross-entropy on the argmax (ablation: soft vs arch).
  (c) ConfProbe: LogisticRegression (== A1n) with per-sample weight = LLM max-confidence.

Fold bookkeeping is REUSED from baselines.py (_prepare_labeled / _resolve_folds /
_iter_folds / _stack_l2) — the same participant-grouped, No-code-by-participant logic the
linear probe and harness use (no leakage). Embeddings are the cached 4096-d Qwen vectors
(L2-normalized). Scoring is H.score_arm (both reference axes + clustered-bootstrap CI).
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get('SL_THREADS', '16')))
# GPU on this box is held by an external embedding server (100% util); default to CPU.
# Override with SL_DEVICE=cuda when the GPU is free.
DEVICE = os.environ.get('SL_DEVICE', 'cpu')
if DEVICE == 'cuda' and not torch.cuda.is_available():
    DEVICE = 'cpu'

from experiments.gnn_reliability import harness as H
from experiments.gnn_reliability import baselines as B
from gnn_layer.config import GnnLayerConfig
from gnn_layer.soft_labels import build_soft_targets

ABS = 'data/Meta'


# ---------------------------------------------------------------------------
# soft-target matrix + class weights (shared)
# ---------------------------------------------------------------------------
def _soft_matrix(soft, seg_ids, n_classes):
    """[n, n_classes] row-normalized soft target matrix aligned to seg_ids."""
    T = np.zeros((len(seg_ids), n_classes), dtype=np.float64)
    for i, s in enumerate(seg_ids):
        v = np.asarray(soft.get(s), dtype=np.float64) if soft.get(s) is not None else None
        if v is None or v.sum() <= 0:
            T[i] = 1.0 / n_classes
        else:
            T[i] = v / v.sum()
    return T


def _balanced_w(y, n_classes):
    """sklearn-'balanced' class weights -> per-example weight (mean 1) by class."""
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    present = counts > 0
    n_present = int(present.sum())
    w_class = np.zeros(n_classes, dtype=np.float64)
    w_class[present] = len(y) / (n_present * counts[present])
    w = w_class[y]
    return w / w.mean()  # normalize to mean 1 (stable lr)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------
class _MLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden=256, n_hidden=1,
                 dropout=0.4, in_dropout=0.1):
        super().__init__()
        self.in_drop = nn.Dropout(in_dropout)
        layers = []
        d = d_in
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        return self.head(self.body(self.in_drop(x)))


def _train_one(Xtr, Ttr, wtr, n_classes, hp, seed):
    """Train one MLP/linear net (soft-CE/KL if Ttr is a distribution; CE handled by caller)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = _MLP(Xtr.shape[1], n_classes, hidden=hp['hidden'], n_hidden=hp['n_hidden'],
               dropout=hp['dropout'], in_dropout=hp['in_dropout']).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=hp['lr'], weight_decay=hp['wd'])
    X = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    T = torch.tensor(Ttr, dtype=torch.float32, device=DEVICE)   # [n, C] target distribution
    w = torch.tensor(wtr, dtype=torch.float32, device=DEVICE)   # [n] per-example weight
    net.train()
    for _ in range(hp['epochs']):
        opt.zero_grad()
        logp = F.log_softmax(net(X), dim=1)
        # soft cross-entropy == KL(target||pred) up to a const; weighted, mean over batch
        ce = -(T * logp).sum(dim=1)
        loss = (w * ce).mean()
        loss.backward()
        opt.step()
    net.eval()
    return net


def _sharpen(T, tau):
    """Temperature-sharpen (tau<1) / soften (tau>1) a row-normalized target matrix."""
    if tau == 1.0:
        return T
    Ts = np.power(np.clip(T, 1e-9, None), 1.0 / tau)
    return Ts / Ts.sum(axis=1, keepdims=True)


def run_distill(df, emb, folds, n_classes, mode, hp, seeds=(0, 1, 2), balance=True,
                tau=1.0, smooth=0.0, pca=None):
    """Grouped-CV out-of-fold predictions for the MLP/linear distillation arms.

    mode='soft' -> KL/soft-CE to build_soft_targets mixture.
    mode='hard' -> CE to the argmax one-hot (ablation).
    tau         -> temperature on the soft target (<1 sharpens toward argmax, >1 softens).
    smooth      -> interpolate target = (1-smooth)*soft + smooth*onehot(argmax) (label-mix).
    pca         -> if set (int), fit PCA(pca) on each fold's TRAIN rows (leak-free) and reduce
                   the 4096-d Qwen features before the MLP (CPU speed; also denoises at n~200).
    Seeds are ensembled (mean softmax) to cut MLP init variance. Reuses the baselines
    fold helpers verbatim, so folds == A1n folds (participant-grouped, No-code-by-part).
    """
    seg_ids, labels, nc = B._prepare_labeled(df, emb, GnnLayerConfig(vaamr_n_classes=n_classes))
    label_of = dict(zip(seg_ids, labels))
    fold_of, fold_list = B._resolve_folds(seg_ids, df, folds)
    soft = build_soft_targets(df, 'weak', n_stages=n_classes)
    T_all = {s: row for s, row in zip(seg_ids, _soft_matrix(soft, seg_ids, n_classes))}

    preds = {}
    for _f, train_ids, test_ids in B._iter_folds(seg_ids, fold_of, fold_list):
        Xtr = B._stack_l2(emb, train_ids).astype(np.float32)
        Xte = B._stack_l2(emb, test_ids).astype(np.float32)
        if pca:
            from sklearn.decomposition import PCA
            n_comp = min(int(pca), Xtr.shape[1], Xtr.shape[0] - 1)
            p = PCA(n_components=n_comp, random_state=0).fit(Xtr)
            Xtr = p.transform(Xtr).astype(np.float32)
            Xte = p.transform(Xte).astype(np.float32)
        ytr = np.array([label_of[s] for s in train_ids], dtype=int)
        onehot = np.eye(n_classes, dtype=np.float64)[ytr]
        if mode == 'soft':
            Ttr = np.stack([T_all[s] for s in train_ids], axis=0)
            Ttr = _sharpen(Ttr, tau)
            if smooth > 0:
                Ttr = (1.0 - smooth) * Ttr + smooth * onehot
        else:  # hard one-hot of argmax label
            Ttr = onehot
        wtr = _balanced_w(ytr, n_classes) if balance else np.ones(len(ytr))
        probs = np.zeros((len(test_ids), n_classes), dtype=np.float64)
        Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
        for sd in seeds:
            net = _train_one(Xtr, Ttr, wtr, n_classes, hp, seed=sd)
            with torch.no_grad():
                probs += F.softmax(net(Xte_t), dim=1).cpu().numpy()
        probs /= len(seeds)
        for s, row in zip(test_ids, probs):
            preds[s] = int(np.argmax(row))
    return preds


# ---------------------------------------------------------------------------
# (c) confidence-weighted linear probe
# ---------------------------------------------------------------------------
def run_conf_probe(df, emb, folds, n_classes, balance=True, conf_pow=1.0):
    """A1n's LogisticRegression with per-sample weight = LLM max-confidence (^conf_pow).

    Confidence = max of the row-normalized soft ballot mixture (ballot agreement;
    No-code rows are one-hot -> conf 1). Composes with class_weight='balanced'.
    """
    from sklearn.linear_model import LogisticRegression
    seg_ids, labels, nc = B._prepare_labeled(df, emb, GnnLayerConfig(vaamr_n_classes=n_classes))
    label_of = dict(zip(seg_ids, labels))
    fold_of, fold_list = B._resolve_folds(seg_ids, df, folds)
    soft = build_soft_targets(df, 'weak', n_stages=n_classes)
    conf = {s: float(np.asarray(soft[s]).max() / max(np.asarray(soft[s]).sum(), 1e-9))
            for s in seg_ids if soft.get(s) is not None}

    preds = {}
    for _f, train_ids, test_ids in B._iter_folds(seg_ids, fold_of, fold_list):
        Xtr = B._stack_l2(emb, train_ids)
        Xte = B._stack_l2(emb, test_ids)
        ytr = np.array([label_of[s] for s in train_ids], dtype=int)
        sw = np.array([conf.get(s, 1.0) ** conf_pow for s in train_ids], dtype=np.float64)
        if len(np.unique(ytr)) < 2:
            for s in test_ids:
                preds[s] = int(ytr[0])
            continue
        cw = 'balanced' if balance else None
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight=cw)
        clf.fit(Xtr, ytr, sample_weight=sw)
        full = np.zeros((len(test_ids), n_classes))
        proba = clf.predict_proba(Xte)
        for j, c in enumerate(clf.classes_):
            full[:, int(c)] = proba[:, j]
        for s, row in zip(test_ids, full):
            preds[s] = int(np.argmax(row))
    return preds


# ---------------------------------------------------------------------------
# scoring helper
# ---------------------------------------------------------------------------
def _score(name, preds, df, n_classes, method, extra=None):
    res = H.score_arm(name, preds, df, ABS, n_classes=n_classes,
                      meta={'embedding': 'qwen', 'method': method, 'imbalance': 'balanced',
                            'seed': 42, 'branch': 'scaler/soft-label',
                            'notes': extra or ''},
                      write_ledger=False)
    llm, hum = res['llm_axis'], res['human_axis']
    rec = {r['class_id']: r['recall'] for r in llm['per_class'] if r['class_id'] < 5}
    rec_str = "/".join(f"{rec.get(i, 0):.2f}" for i in range(5))
    row = {
        'name': name, 'nc': n_classes,
        'llm_k': llm['cohen_kappa_205'], 'llm_lo': llm['ci95'][0], 'llm_hi': llm['ci95'][1],
        'hum_k': hum['cohen_kappa'], 'hum_lo': hum['ci95'][0], 'hum_hi': hum['ci95'][1],
        'hum_n': hum['n'], 'recall': rec_str,
    }
    print(f"  {name:<26} nc={n_classes} | LLM kappa(205)={row['llm_k']:.3f} "
          f"[{row['llm_lo']:.3f},{row['llm_hi']:.3f}] | HUM kappa={row['hum_k']:.3f} "
          f"[{row['hum_lo']:.3f},{row['hum_hi']:.3f}] n={row['hum_n']} | rec V/Av/AR/Mc/Re={rec_str}",
          flush=True)
    return row


if __name__ == '__main__':
    print("loading corpus / folds / qwen embeddings ...")
    df = H.load_corpus(ABS)
    folds = H.build_folds(df, seed=42, verbose=False)
    emb = H.get_embeddings(df, 'qwen', ABS)
    print(f"corpus={len(df)} folds={len(folds)} emb_dim={len(next(iter(emb.values())))}\n")

    # default MLP hyperparameters (tuned lightly in the grid below)
    HP = dict(hidden=256, n_hidden=1, dropout=0.4, in_dropout=0.1,
              lr=1e-3, wd=1e-3, epochs=150)
    rows = []
    # A1n baseline (validated EXACTLY in the self-check: HUM kappa=0.365, LLM kappa=0.283).
    print("A1n baseline (from self-check): HUM kappa=0.365 [0.228,0.513], LLM kappa(205)=0.283 "
          "[0.203,0.346]\n", flush=True)

    print("=== (a) MLP-KL on SOFT mixture ===", flush=True)
    for nc in (6, 5):
        rows.append(_score('MLP-KL_soft', run_distill(df, emb, folds, nc, 'soft', HP),
                           df, nc, 'MLP-KL', 'soft mixture'))

    print("\n=== (b) MLP-CE on ARGMAX (ablation: soft signal vs MLP arch) ===", flush=True)
    for nc in (6, 5):
        rows.append(_score('MLP-CE_hard', run_distill(df, emb, folds, nc, 'hard', HP),
                           df, nc, 'MLP-CE', 'hard argmax'))

    print("\n=== (c) confidence-weighted linear probe ===", flush=True)
    for nc in (6, 5):
        rows.append(_score('ConfProbe', run_conf_probe(df, emb, folds, nc, balance=True),
                           df, nc, 'ConfProbe', 'sample_w=ballot max-conf'))

    print("\nDONE.")
