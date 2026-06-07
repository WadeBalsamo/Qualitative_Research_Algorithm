"""Tune the per-rater ensemble on the LLM axis: sweep LogReg C (regularization).
Point kappa only for the sweep; full CI scoring for the winner. (scaler/ensemble)"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '8'
import sys, dataclasses
sys.path.insert(0, 'src'); sys.path.insert(0, '.')
import numpy as np
from sklearn.linear_model import LogisticRegression
from experiments.gnn_reliability import harness as H, baselines as B
from experiments.gnn_reliability import rater_distill as RD
from gnn_layer.config import GnnLayerConfig
from analysis import irr_stats

ABS = '/home/wisgood/qra/Qualitative_Research_Algorithm/data/Meta'
CFG = dataclasses.replace(GnnLayerConfig(), vaamr_n_classes=6, vaamr_class_balance=True)
df = H.load_corpus(ABS)
folds = H.build_folds(df, seed=42, verbose=False)
emb = H.get_embeddings(df, 'qwen', ABS)

lab = H._labeled_participants(df)
final_of = {str(r['segment_id']): int(r['final_label']) for _, r in lab.iterrows()}
NC = 6


def proba_for_C(C):
    """Per-rater OOF proba dict with a custom LogReg C (balanced)."""
    seg_ids, per_rater, _ = RD.participant_rater_labels(df, NC)
    seg_ids = [s for s in seg_ids if s in emb]
    fold_of, fold_list = B._resolve_folds(seg_ids, df, folds)
    out = {r: {} for r in RD.RATERS}
    for _f, tr, te in B._iter_folds(seg_ids, fold_of, fold_list):
        Xte = B._stack_l2(emb, te)
        for r in RD.RATERS:
            rows = [s for s in tr if s in per_rater[r]]
            if len(set(per_rater[r][s] for s in rows)) < 2:
                continue
            clf = LogisticRegression(max_iter=3000, C=C, class_weight='balanced')
            clf.fit(B._stack_l2(emb, rows), [per_rater[r][s] for s in rows])
            full = np.zeros((len(te), NC))
            for j, c in enumerate(clf.classes_):
                if 0 <= int(c) < NC:
                    full[:, int(c)] = clf.predict_proba(Xte)[:, j]
            for s, row in zip(te, full):
                out[r][s] = row
    return seg_ids, out


def llm_kappa(oof):
    p = [oof[s] for s in final_of if s in oof]
    r = [final_of[s] for s in final_of if s in oof]
    return irr_stats.cohen_kappa(p, r)


print('C-sweep (ens_softavg), LLM-axis point kappa over 205:')
best = (None, -1)
for C in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
    seg_ids, proba = proba_for_C(C)
    oof = RD.ensemble_from_proba(seg_ids, proba, NC, 'softavg')
    k = llm_kappa(oof)
    print(f'  C={C:<5g}  ens_softavg LLM kappa={k:+.4f}', flush=True)
    if k > best[1]:
        best = (C, k)
print(f'-> best C={best[0]} (LLM kappa={best[1]:+.4f})')

# full CI score for the winning C
seg_ids, proba = proba_for_C(best[0])
oof = RD.ensemble_from_proba(seg_ids, proba, NC, 'softavg')
res = H.score_arm(f'ens_softavg_C{best[0]}', oof, df, ABS, n_classes=6,
                  meta={'embedding': 'qwen', 'method': 'ens_soft_tuned', 'seed': 42,
                        'branch': 'scaler/ensemble'}, write_ledger=False)
H._print_result(res)
