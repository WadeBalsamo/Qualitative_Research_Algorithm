"""
experiments/gnn_reliability/capacity_scaler.py
----------------------------------------------
SCALER experiment (branch scaler/nonlinear): does model capacity BEYOND a linear
probe reproduce the multi-run LLM VAAMR consensus better than A1n at n~205, or
does it overfit?

Baseline A1n = Qwen class-weighted 6-class LINEAR probe: human kappa=0.365 /
LLM-axis grouped kappa(205)=0.283. Bar: classifier<->LLM grouped kappa >= ~0.45
OR classifier<->human kappa >= ~0.50 (CI-aware).

LEVER = capacity. On the L2-normalized (or StandardScaled) Qwen TARGET features,
class-weighted, 6-class, IDENTICAL participant-grouped folds (harness build_folds),
sweep: (a) MLP [torch, class-weighted; sklearn unweighted], (b) HistGradientBoosting,
(c) SVM-RBF, (d) calibrated linear probe (Platt/isotonic). Hyperparameters tuned on
the LLM axis ONLY (grouped CV); the human axis is read once for the finalists.

Reuses harness CV/scoring + baselines fold bookkeeping (no reinvented CV).
NOT committed. Run:  python experiments/gnn_reliability/capacity_scaler.py [family ...]
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)

import numpy as np  # noqa: E402

from experiments.gnn_reliability import harness as H  # noqa: E402
from experiments.gnn_reliability import baselines as B  # noqa: E402
from gnn_layer.config import GnnLayerConfig  # noqa: E402
from analysis import irr_stats  # noqa: E402

ABS = '/home/wisgood/qra/Qualitative_Research_Algorithm/data/Meta'
CFG6 = GnnLayerConfig(vaamr_n_classes=6)   # 6-class target set (incl. No-code=5)
SEED = 42


# ---------------------------------------------------------------------------
# feature prep (per-fold, leakage-free) + proba expansion
# ---------------------------------------------------------------------------
def _stack_raw(emb, ids):
    return np.stack([np.asarray(emb[s], dtype=np.float64) for s in ids], axis=0)


def _prep_fit(Xtr, Xte, prep):
    """Fit feature prep on TRAIN rows only; return (Xtr', Xte')."""
    from sklearn.preprocessing import normalize, StandardScaler
    if prep == 'l2':
        return normalize(Xtr, 'l2', axis=1), normalize(Xte, 'l2', axis=1)
    if prep == 'standard':
        sc = StandardScaler().fit(Xtr)
        return sc.transform(Xtr), sc.transform(Xte)
    if prep == 'l2+standard':
        Xtr = normalize(Xtr, 'l2', axis=1); Xte = normalize(Xte, 'l2', axis=1)
        sc = StandardScaler().fit(Xtr)
        return sc.transform(Xtr), sc.transform(Xte)
    raise ValueError(prep)


def _proba_full(clf, X, n_classes):
    proba = clf.predict_proba(X)
    full = np.zeros((len(X), n_classes), dtype=np.float64)
    for j, c in enumerate(clf.classes_):
        ci = int(c)
        if 0 <= ci < n_classes:
            full[:, ci] = proba[:, j]
    return full


def run_clf(df, emb, folds, make_clf, prep, sample_weight_balanced=False, config=CFG6):
    """OOF predictions for every labeled participant segment, grouped CV.

    Reuses baselines._prepare_labeled/_resolve_folds/_iter_folds (the harness fold
    bookkeeping, incl. the No-code-by-participant handling). ``make_clf`` is a
    zero-arg factory; class weighting is either inside the estimator
    (class_weight='balanced') or via sample_weight here.
    """
    seg_ids, labels, n_classes = B._prepare_labeled(df, emb, config)
    label_of = dict(zip(seg_ids, labels))
    fold_of, fold_list = B._resolve_folds(seg_ids, df, folds)
    preds = {}
    for _f, tr, te in B._iter_folds(seg_ids, fold_of, fold_list):
        Xtr, Xte = _prep_fit(_stack_raw(emb, tr), _stack_raw(emb, te), prep)
        ytr = np.array([label_of[s] for s in tr], dtype=int)
        if len(np.unique(ytr)) < 2:
            for s in te:
                preds[s] = int(ytr[0])
            continue
        clf = make_clf()
        if sample_weight_balanced:
            from sklearn.utils.class_weight import compute_sample_weight
            clf.fit(Xtr, ytr, sample_weight=compute_sample_weight('balanced', ytr))
        else:
            clf.fit(Xtr, ytr)
        for s, row in zip(te, _proba_full(clf, Xte, n_classes)):
            preds[s] = int(np.argmax(row))
    return preds, n_classes


# ---------------------------------------------------------------------------
# LLM-axis-only objective for tuning (matches score_arm's kappa(205))
# ---------------------------------------------------------------------------
def llm_kappa205(oof, df):
    lab = H._labeled_participants(df)
    final_of = {str(r['segment_id']): int(r['final_label']) for _, r in lab.iterrows()}
    a, b = [], []
    for sid, ref in final_of.items():
        if sid in oof:
            a.append(int(oof[sid])); b.append(int(ref))
    return irr_stats.cohen_kappa(a, b), len(a)


# ---------------------------------------------------------------------------
# torch class-weighted MLP (small, dropout + weight_decay regularized)
# ---------------------------------------------------------------------------
class TorchMLP:
    def __init__(self, n_features, n_classes, hidden=(128,), dropout=0.5,
                 weight_decay=1e-3, lr=1e-3, epochs=200, seed=42, class_weight=True):
        self.n_features = n_features; self.n_classes = n_classes
        self.hidden = hidden; self.dropout = dropout; self.weight_decay = weight_decay
        self.lr = lr; self.epochs = epochs; self.seed = seed; self.class_weight = class_weight
        self.classes_ = np.arange(n_classes)

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        layers = []; d = self.n_features
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(self.dropout)]; d = h
        layers += [nn.Linear(d, self.n_classes)]
        self.net = nn.Sequential(*layers).to(dev); self.dev = dev
        Xt = torch.tensor(X, dtype=torch.float32, device=dev)
        yt = torch.tensor(y, dtype=torch.long, device=dev)
        wt = None
        if self.class_weight:
            from sklearn.utils.class_weight import compute_class_weight
            present = np.unique(y)
            w = np.ones(self.n_classes, dtype=np.float32)
            for c, val in zip(present, compute_class_weight('balanced', classes=present, y=y)):
                w[int(c)] = val
            wt = torch.tensor(w, device=dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lossf = nn.CrossEntropyLoss(weight=wt)
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad(); loss = lossf(self.net(Xt), yt); loss.backward(); opt.step()
        return self

    def predict_proba(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32, device=self.dev)
            p = torch.softmax(self.net(Xt), dim=1).cpu().numpy()
        return p


# ---------------------------------------------------------------------------
# grids (each tuned on LLM kappa(205))
# ---------------------------------------------------------------------------
def grid_linear():
    from sklearn.linear_model import LogisticRegression
    for prep in ('l2', 'standard'):
        for C in (0.5, 1.0, 2.0):
            yield (dict(model='linear', prep=prep, C=C),
                   prep,
                   (lambda C=C: LogisticRegression(max_iter=3000, C=C, class_weight='balanced')),
                   False)


def grid_calibrated():
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    for prep in ('l2', 'standard'):
        for method in ('sigmoid', 'isotonic'):
            base = lambda: LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced')
            yield (dict(model='calibrated', prep=prep, method=method),
                   prep,
                   (lambda m=method, b=base: CalibratedClassifierCV(b(), method=m, cv=3)),
                   False)


def grid_svc():
    from sklearn.svm import SVC
    for prep in ('l2', 'standard'):
        for C in (1.0, 10.0, 100.0):
            for gamma in ('scale', 0.01, 0.1):
                yield (dict(model='svc-rbf', prep=prep, C=C, gamma=gamma),
                       prep,
                       (lambda C=C, g=gamma: SVC(C=C, gamma=g, kernel='rbf',
                                                 class_weight='balanced',
                                                 probability=True, random_state=SEED)),
                       False)


def grid_hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier
    for prep in ('l2', 'standard'):
        for lr in (0.05, 0.1):
            for md in (3, None):
                for l2 in (0.0, 1.0):
                    yield (dict(model='hgb', prep=prep, lr=lr, max_depth=md, l2=l2),
                           prep,
                           (lambda lr=lr, md=md, l2=l2: HistGradientBoostingClassifier(
                               class_weight='balanced', learning_rate=lr, max_depth=md,
                               max_iter=300, min_samples_leaf=15, l2_regularization=l2,
                               early_stopping=False, random_state=SEED)),
                           False)


def grid_mlp(n_features):
    for prep in ('l2', 'standard'):
        for hidden in ((64,), (128,), (128, 64)):
            for dropout in (0.3, 0.5):
                for wd in (1e-3, 1e-2):
                    yield (dict(model='mlp-torch', prep=prep, hidden=hidden,
                                dropout=dropout, wd=wd),
                           prep,
                           (lambda h=hidden, dr=dropout, wd=wd: TorchMLP(
                               n_features, 6, hidden=h, dropout=dr, weight_decay=wd,
                               lr=1e-3, epochs=200, seed=SEED, class_weight=True)),
                           False)


def grid_mlp_sk():
    """sklearn MLP, UNWEIGHTED (capacity check without class weights)."""
    from sklearn.neural_network import MLPClassifier
    for prep in ('l2', 'standard'):
        for hidden in ((64,), (128, 64)):
            for alpha in (1e-3, 1e-1):
                yield (dict(model='mlp-sk', prep=prep, hidden=hidden, alpha=alpha),
                       prep,
                       (lambda h=hidden, a=alpha: MLPClassifier(
                           hidden_layer_sizes=h, alpha=a, max_iter=500,
                           early_stopping=True, random_state=SEED)),
                       False)


# ---------------------------------------------------------------------------
# sweep + finalist scoring
# ---------------------------------------------------------------------------
def sweep(df, emb, folds, gen, label):
    print(f"\n=== sweeping {label} ===", flush=True)
    rows = []
    for spec, prep, make_clf, swb in gen:
        oof, _ = run_clf(df, emb, folds, make_clf, prep, sample_weight_balanced=swb)
        k, n = llm_kappa205(oof, df)
        rows.append((k, spec, oof))
        print(f"  LLMk={k:+.4f}  {spec}", flush=True)
    rows.sort(key=lambda r: (r[0] if r[0] is not None else -1), reverse=True)
    best_k, best_spec, best_oof = rows[0]
    print(f"  -> BEST {label}: LLMk={best_k:+.4f}  {best_spec}", flush=True)
    return best_spec, best_oof, best_k


def finalize(df, oof, tag, spec):
    res = H.score_arm(tag, oof, df, ABS, n_classes=6,
                      meta={'embedding': 'qwen', 'method': spec.get('model'),
                            'seed': SEED, 'branch': 'scaler/nonlinear'},
                      write_ledger=False)
    llm, hum = res['llm_axis'], res['human_axis']
    rec = {r['class_id']: r['recall'] for r in llm['per_class'] if r['class_id'] < 5}
    return {
        'tag': tag, 'spec': spec,
        'llm_k': llm['cohen_kappa_205'], 'llm_ci': llm['ci95'], 'llm_n': llm['n'],
        'hum_k': hum['cohen_kappa'], 'hum_ci': hum['ci95'], 'hum_n': hum['n'],
        'recall': rec,
    }


def main(families=None):
    df = H.load_corpus(ABS)
    folds = H.build_folds(df, seed=SEED, verbose=False)
    emb = H.get_embeddings(df, 'qwen', ABS)
    D = int(len(next(iter(emb.values()))))
    print(f"corpus loaded: emb n={len(emb)} dim={D}; folds over labeled-205", flush=True)

    families = families or ['linear', 'calibrated', 'svc', 'hgb', 'mlp', 'mlp_sk']
    gens = {
        'linear': (grid_linear(), 'Linear (class-weighted, ref/scaling)'),
        'calibrated': (grid_calibrated(), '(d) Calibrated linear probe'),
        'svc': (grid_svc(), '(c) SVM-RBF'),
        'hgb': (grid_hgb(), '(b) HistGradientBoosting'),
        'mlp': (grid_mlp(D), '(a) MLP torch class-weighted'),
        'mlp_sk': (grid_mlp_sk(), '(a) MLP sklearn unweighted'),
    }
    finals = []
    for fam in families:
        gen, lbl = gens[fam]
        try:
            spec, oof, _ = sweep(df, emb, folds, gen, lbl)
            finals.append(finalize(df, oof, f'S4_{fam}', spec))
        except Exception as e:  # noqa: BLE001 — keep other families alive
            import traceback
            print(f"  !! family {fam} FAILED: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()

    print("\n" + "=" * 100)
    print("FINALIST SCORING (both axes, participant-clustered 95% CI; human axis read once)")
    print("=" * 100)
    hdr = f"{'model':<26} {'LLM k(205) [CI]':<26} {'human k [CI]':<26} {'recall V/Av/AR/Mc/Re'}"
    print(hdr); print("-" * 100)
    for fr in finals:
        llm_ci = f"[{_f(fr['llm_ci'][0])},{_f(fr['llm_ci'][1])}]"
        hum_ci = f"[{_f(fr['hum_ci'][0])},{_f(fr['hum_ci'][1])}]"
        rc = fr['recall']
        recall_s = "/".join(_f(rc.get(i)) for i in range(5))
        print(f"{fr['tag']:<26} {_f(fr['llm_k'])+' '+llm_ci:<26} "
              f"{_f(fr['hum_k'])+' '+hum_ci:<26} {recall_s}")
        print(f"   spec={fr['spec']}")
    print("-" * 100)
    print("BASELINE A1n: LLM k(205)=0.2831 [0.2034,0.3455] | human k=0.3652 [0.2276,0.5133]")
    print("BAR: LLM k>=~0.45 OR human k>=~0.50 (CI-aware)")
    return finals


def _f(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ' . '
    return f"{v:.3f}"


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    main(families=args or None)
