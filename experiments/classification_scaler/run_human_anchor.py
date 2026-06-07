"""HUMAN ANCHOR experiment: can the scarce 66 human consensus codes pull the
Qwen linear probe toward human-level VAAMR without overfitting?

Variants vs A1n (Qwen, 6-class, class-weighted linear probe):
  (a) Mix training  : LLM(339) + human-override up-weighted (sweep weight).
  (b) Recalibration : base LLM probe + correction fit LEAVE-ONE-PARTICIPANT-OUT
                      over the human-coded participants (no leakage).
  (c) Human-only    : probe trained only on the 66 human codes, LOPO (the floor).

All folds participant-grouped (reuses harness build_folds + baselines._resolve_folds).
Human axis scored exactly like analysis.irr_analysis via H.score_arm.

Run from repo root:
    python experiments/classification_scaler/run_human_anchor.py

Design reference: experiments/docs/graph_experiments.md, experiments/docs/design_decisions.md
"""
import sys, os, warnings
warnings.filterwarnings('ignore')

# --- bootstrap src/ + repo root onto sys.path (run as a script or imported) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))   # repo root
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from scipy.optimize import minimize
from experiments.gnn_reliability import harness as H, baselines as B
from gnn_layer.config import GnnLayerConfig

ABS = 'data/Meta'
SEED = 42
EPS = 1e-6


# ============================================================================
# helpers (module-level so callers can import without side effects)
# ============================================================================
def _f(v):
    return 'nan' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.3f}"


def show(tag, res):
    llm, hum = res['llm_axis'], res['human_axis']
    pc = ", ".join(f"{r['class_name'][:4]}={r['recall']:.2f}" for r in llm['per_class'] if r['class_id'] < 5)
    print(f"\n### {tag}")
    print(f"  LLM  κ(205)={llm['cohen_kappa_205']:.3f} [{_f(llm['ci95'][0])},{_f(llm['ci95'][1])}] n={llm['n']}  κ(76)={_f(llm['cohen_kappa_76'])}")
    print(f"  HUM  κ      ={hum['cohen_kappa']:.3f} [{_f(hum['ci95'][0])},{_f(hum['ci95'][1])}] n={hum['n']} ({hum['n_clusters']}p)")
    print(f"  recall: {pc}")
    return dict(tag=tag, llm=llm['cohen_kappa_205'], llm_lo=llm['ci95'][0], llm_hi=llm['ci95'][1],
               hum=hum['cohen_kappa'], hum_lo=hum['ci95'][0], hum_hi=hum['ci95'][1],
               pc={r['class_id']: r['recall'] for r in llm['per_class']})


def _Xrows(emb, ids):
    return normalize(np.stack([np.asarray(emb[s], dtype=np.float64) for s in ids]), norm='l2', axis=1)


def _fit_probe_w(X, y, w):
    """class-weighted LR with optional per-sample weights; constant fallback for <2 classes."""
    if len(np.unique(y)) < 2:
        c = int(np.unique(y)[0])
        class K:
            classes_ = np.array([c])
            def predict_proba(self, Z): return np.ones((len(Z), 1))
        return K()
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced')
    clf.fit(X, y, sample_weight=w)
    return clf


def _proba_full(clf, X, n=6):
    P = clf.predict_proba(X)
    full = np.zeros((len(X), n))
    for j, c in enumerate(clf.classes_):
        if 0 <= int(c) < n:
            full[:, int(c)] = P[:, j]
    return full


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_bias(LP, Y, lam):
    """per-class additive bias b on log-probs (scale fixed=1): min CE + lam||b||^2."""
    n, C = LP.shape
    oh = np.eye(C)[Y]
    def fg(b):
        S = _softmax(LP + b)
        loss = -(oh * np.log(S + EPS)).sum() / n + lam * (b @ b)
        grad = (S - oh).sum(0) / n + 2 * lam * b
        return loss, grad
    r = minimize(fg, np.zeros(C), jac=True, method='L-BFGS-B')
    return r.x


def fit_scalebias(LP, Y, lam):
    """scalar temperature s>0 + per-class bias b (Guo vector-scaling, shared s)."""
    n, C = LP.shape
    oh = np.eye(C)[Y]
    def fg(th):
        s = th[0]; b = th[1:]
        Z = s * LP + b
        S = _softmax(Z)
        loss = -(oh * np.log(S + EPS)).sum() / n + lam * (b @ b)
        gb = (S - oh).sum(0) / n + 2 * lam * b
        gs = ((S - oh) * LP).sum() / n
        return loss, np.concatenate([[gs], gb])
    r = minimize(fg, np.concatenate([[1.0], np.zeros(C)]), jac=True, method='L-BFGS-B')
    return r.x


def fit_stack(LP, Y, C_reg):
    clf = LogisticRegression(max_iter=5000, C=C_reg, class_weight='balanced', multi_class='multinomial')
    clf.fit(LP, Y)
    return clf


def main():
    # -------------------------------------------------------------------------
    # load once
    # -------------------------------------------------------------------------
    df = H.load_corpus(ABS)
    folds = H.build_folds(df, seed=SEED, verbose=False)
    emb = H.get_embeddings(df, 'qwen', ABS)
    cfg6 = GnnLayerConfig()
    cfg6.vaamr_n_classes = 6
    cfg6.vaamr_class_balance = True
    cfg6.seed = SEED

    # human map: segment_id -> human class (No-code -1 -> 5)
    hp = df[(df['speaker'] == 'participant') & (df['in_human_coded_subset'] == True)].copy()
    human_cls = {str(r['segment_id']): (5 if int(r['human_label']) == -1 else int(r['human_label']))
                 for _, r in hp.iterrows()}
    part_of = {str(r['segment_id']): str(r['participant_id']) for _, r in df.iterrows()}
    human_parts = sorted({part_of[s] for s in human_cls})
    print(f"[setup] human-coded segs={len(human_cls)} across {len(human_parts)} participants; "
          f"emb_dim={len(next(iter(emb.values())))}")

    ROWS = []

    # -------------------------------------------------------------------------
    # SELF-CHECK: reproduce A1n via baselines.run_linear_probe
    # -------------------------------------------------------------------------
    oof = B.run_linear_probe(df, emb, folds, cfg6)
    res_a1n = H.score_arm('A1n_repro', oof, df, ABS, n_classes=6,
                          meta={'embedding': 'qwen', 'method': 'LinearProbe', 'seed': SEED}, write_ledger=False)
    ROWS.append(show('A1n (self-check baseline)', res_a1n))
    A1N_LLM = res_a1n['llm_axis']['cohen_kappa_205']
    A1N_LLM_LO = res_a1n['llm_axis']['ci95'][0]

    # all participant seg_ids + LLM labels (No-code -> 5), via baselines contract
    seg_ids, llm_labels, _ = B._prepare_labeled(df, emb, cfg6)
    llm_lab = dict(zip(seg_ids, llm_labels))
    fold_of, fold_list = B._resolve_folds(seg_ids, df, folds)

    # =========================================================================
    # VARIANT (a) — MIX TRAINING (human override + up-weight sweep)
    # =========================================================================
    def run_mix(weight):
        preds = {}
        for f, tr, te in B._iter_folds(seg_ids, fold_of, fold_list):
            ylab = [human_cls[s] if s in human_cls else llm_lab[s] for s in tr]
            w    = [weight     if s in human_cls else 1.0        for s in tr]
            clf = _fit_probe_w(_Xrows(emb, tr), np.array(ylab), np.array(w, dtype=float))
            Pr = _proba_full(clf, _Xrows(emb, te))
            for s, row in zip(te, Pr):
                preds[s] = int(np.argmax(row))
        return preds

    print("\n========== VARIANT (a) MIX TRAINING (human override, up-weight sweep) ==========")
    mix_rows = []
    for w in [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]:
        r = H.score_arm(f'mix_w{w:g}', run_mix(w), df, ABS, n_classes=6,
                        meta={'method': 'mix'}, write_ledger=False)
        mix_rows.append(show(f'(a) mix  weight={w:g}', r))
        mix_rows[-1]['w'] = w

    # =========================================================================
    # base OOF probabilities (for variant b) — grouped CV on LLM labels
    # =========================================================================
    base_proba = {}
    for f, tr, te in B._iter_folds(seg_ids, fold_of, fold_list):
        clf = _fit_probe_w(_Xrows(emb, tr), np.array([llm_lab[s] for s in tr]), np.ones(len(tr)))
        Pr = _proba_full(clf, _Xrows(emb, te))
        for s, row in zip(te, Pr):
            base_proba[s] = row
    base_logp = {s: np.log(p + EPS) for s, p in base_proba.items()}

    # =========================================================================
    # VARIANT (b) — RECALIBRATION, LEAVE-ONE-PARTICIPANT-OUT
    # =========================================================================
    def run_recal(kind, hp_):
        """LOPO over human participants; non-human participants use ALL human codes."""
        h_ids = list(human_cls)
        H_LP = np.stack([base_logp[s] for s in h_ids])
        H_Y = np.array([human_cls[s] for s in h_ids])
        h_part = np.array([part_of[s] for s in h_ids])
        preds = {}

        # cache a "global" fit (all human) for non-human participants
        def fit(mask):
            LP, Y = H_LP[mask], H_Y[mask]
            if kind == 'bias':      return ('p', fit_bias(LP, Y, hp_))
            if kind == 'scalebias': return ('s', fit_scalebias(LP, Y, hp_))
            if kind == 'stack':     return ('c', fit_stack(LP, Y, hp_))

        def apply(model, LP):
            t, m = model
            if t == 'p': return np.argmax(_softmax(LP + m), axis=1)
            if t == 's': return np.argmax(_softmax(m[0] * LP + m[1:]), axis=1)
            if t == 'c':
                full = np.zeros((len(LP), 6))
                for j, c in enumerate(m.classes_):
                    full[:, int(c)] = m.predict_proba(LP)[:, j]
                return np.argmax(full, axis=1)

        global_model = fit(np.ones(len(h_ids), dtype=bool))
        per_part = {p: fit(h_part != p) for p in human_parts}   # leave-p-out fit
        for s in seg_ids:
            p = part_of[s]
            model = per_part[p] if p in per_part else global_model
            preds[s] = int(apply(model, base_logp[s][None, :])[0])
        return preds

    print("\n========== VARIANT (b) RECALIBRATION (leave-one-participant-out) ==========")
    recal_rows = []
    for kind, sweep in [('bias', [0.0, 0.05, 0.2, 1.0]), ('scalebias', [0.0, 0.05, 0.2, 1.0]), ('stack', [0.1, 0.5, 2.0])]:
        for hp_ in sweep:
            r = H.score_arm(f'recal_{kind}_{hp_:g}', run_recal(kind, hp_), df, ABS, n_classes=6,
                            meta={'method': f'recal_{kind}'}, write_ledger=False)
            row = show(f'(b) recal {kind}  reg={hp_:g}', r)
            row['kind'] = kind
            row['reg'] = hp_
            recal_rows.append(row)

    # =========================================================================
    # VARIANT (c) — HUMAN-ONLY PROBE, LEAVE-ONE-PARTICIPANT-OUT
    # =========================================================================
    def run_human_only():
        h_ids = list(human_cls)
        h_part = np.array([part_of[s] for s in h_ids])
        HY = np.array([human_cls[s] for s in h_ids])
        preds = {}
        # fit-all for non-human participants (LLM-axis 205 coverage)
        clf_all = _fit_probe_w(_Xrows(emb, h_ids), HY, np.ones(len(h_ids)))
        per_part = {}
        for p in human_parts:
            mask = h_part != p
            per_part[p] = _fit_probe_w(
                _Xrows(emb, [h_ids[i] for i in range(len(h_ids)) if mask[i]]),
                HY[mask],
                np.ones(int(mask.sum()))
            )
        for s in seg_ids:
            p = part_of[s]
            clf = per_part[p] if p in per_part else clf_all
            preds[s] = int(np.argmax(_proba_full(clf, _Xrows(emb, [s]))[0]))
        return preds

    print("\n========== VARIANT (c) HUMAN-ONLY PROBE (leave-one-participant-out floor) ==========")
    res_c = H.score_arm('human_only', run_human_only(), df, ABS, n_classes=6,
                        meta={'method': 'human_only'}, write_ledger=False)
    ROWS.append(show('(c) human-only probe (LOPO)', res_c))

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    A1N_HUM = res_a1n['human_axis']['cohen_kappa']
    print("\n\n================== SUMMARY (vs A1n: LLMκ=%.3f[lo %.3f], HUMκ=%.3f) ==================" % (
        A1N_LLM, A1N_LLM_LO, A1N_HUM))
    hdr = f"{'variant':28s} {'LLMκ205':>8s} {'[lo,hi]':>16s} {'HUMκ':>7s} {'[lo,hi]':>16s} {'beatA1n':>7s} {'bar':>5s}"
    print(hdr)
    print('-' * len(hdr))

    def line(tag, llm, llm_lo, llm_hi, hum, hum_lo, hum_hi):
        beat = 'Y' if hum > A1N_HUM + 1e-9 else '.'
        bar  = 'HUM' if hum >= 0.50 else ('LLM' if (llm_lo is not None and llm_lo >= 0.45) else '.')
        print(f"{tag:28s} {llm:8.3f} [{_f(llm_lo)},{_f(llm_hi)}]".ljust(54) +
              f" {hum:7.3f} [{_f(hum_lo)},{_f(hum_hi)}] {beat:>7s} {bar:>5s}")

    line('A1n', A1N_LLM, res_a1n['llm_axis']['ci95'][0], res_a1n['llm_axis']['ci95'][1],
         A1N_HUM, res_a1n['human_axis']['ci95'][0], res_a1n['human_axis']['ci95'][1])
    for r in mix_rows:
        line(f"(a)mix w={r['w']:g}", r['llm'], r['llm_lo'], r['llm_hi'],
             r['hum'], r['hum_lo'], r['hum_hi'])
    for r in recal_rows:
        line(f"(b){r['kind']} reg={r['reg']:g}", r['llm'], r['llm_lo'], r['llm_hi'],
             r['hum'], r['hum_lo'], r['hum_hi'])
    line('(c)human-only',
         res_c['llm_axis']['cohen_kappa_205'], res_c['llm_axis']['ci95'][0], res_c['llm_axis']['ci95'][1],
         res_c['human_axis']['cohen_kappa'], res_c['human_axis']['ci95'][0], res_c['human_axis']['ci95'][1])
    print("\nGUARDRAIL note: LLM-axis tuning rule = keep LLMκ(205) >= A1n LLM lower-CI (%.3f)." % A1N_LLM_LO)


if __name__ == '__main__':
    main()
