"""Extend the C grid (check for overfit/plateau) + paired Delta-kappa of the tuned
ens_softavg winner vs A1n on both axes. (scaler/ensemble)"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '8'
import sys, dataclasses
_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo src/
for _p in (os.path.dirname(_SRC), _SRC):  # repo root then src/ -> src/ ends up first on sys.path
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
from sklearn.linear_model import LogisticRegression
from experiments.gnn_reliability import harness as H, baselines as B
from experiments.classification_scaler import rater_distill as RD
from gnn_layer.config import GnnLayerConfig
from process import irr_import
from analysis import irr_stats, stats as _stats
from analysis.irr_analysis import _consensus_rows

ABS = '/home/wisgood/qra/Qualitative_Research_Algorithm/data/Meta'
CFG = dataclasses.replace(GnnLayerConfig(), vaamr_n_classes=6, vaamr_class_balance=True)
df = H.load_corpus(ABS)
folds = H.build_folds(df, seed=42, verbose=False)
emb = H.get_embeddings(df, 'qwen', ABS)
part_of = {str(r['segment_id']): str(r['participant_id']) for _, r in df.iterrows()}
lab = H._labeled_participants(df)
final_of = {str(r['segment_id']): int(r['final_label']) for _, r in lab.iterrows()}
NC = 6


def proba_for_C(C):
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
            clf = LogisticRegression(max_iter=4000, C=C, class_weight='balanced')
            clf.fit(B._stack_l2(emb, rows), [per_rater[r][s] for s in rows])
            P = clf.predict_proba(Xte)
            full = np.zeros((len(te), NC))
            for j, c in enumerate(clf.classes_):
                if 0 <= int(c) < NC:
                    full[:, int(c)] = P[:, j]
            for s, row in zip(te, full):
                out[r][s] = row
    return seg_ids, out


def llm_kappa(oof):
    return irr_stats.cohen_kappa([oof[s] for s in final_of if s in oof],
                                 [final_of[s] for s in final_of if s in oof])


print('extended C-sweep (ens_softavg LLM-axis point kappa):', flush=True)
cache = {}
for C in [4, 6, 8, 12, 16]:
    seg_ids, proba = proba_for_C(C)
    cache[C] = (seg_ids, proba)
    oof = RD.ensemble_from_proba(seg_ids, proba, NC, 'softavg')
    print(f'  C={C:<4g}  LLM kappa={llm_kappa(oof):+.4f}', flush=True)

# A1n OOF for the paired test
oof_a1n = B.run_linear_probe(df, emb, folds, CFG)


def paired_delta(items, n_boot=3000, seed=42):
    ref = np.array([t[0] for t in items]); pa = np.array([t[1] for t in items])
    pb = np.array([t[2] for t in items]); cl = [t[3] for t in items]
    packed = ((ref + 1) * 49 + (pa + 1) * 7 + (pb + 1)).astype(float)

    def stat(arr):
        c = arr.astype(int); rr = (c // 49) - 1; pp = ((c % 49) // 7) - 1; qq = (c % 7) - 1
        ka = irr_stats.cohen_kappa(pp.tolist(), rr.tolist())
        kb = irr_stats.cohen_kappa(qq.tolist(), rr.tolist())
        return float('nan') if (ka is None or kb is None) else float(kb - ka)

    res = _stats.cluster_bootstrap_ci(packed, cl, statistic=stat, n_boot=n_boot, seed=seed)
    ka = irr_stats.cohen_kappa(pa.tolist(), ref.tolist())
    kb = irr_stats.cohen_kappa(pb.tolist(), ref.tolist())
    res['point'] = float(kb - ka)
    return res


def map6(p):
    return -1 if p == 5 else p


def llm_items(oB):
    return [(int(r['final_label']), int(oof_a1n[s]), int(oB[s]), part_of.get(s))
            for _, r in lab.iterrows() if (s := str(r['segment_id'])) in oof_a1n and s in oB]


def human_items(oB):
    codes = irr_import.read_human_codes(ABS); master = set(df['segment_id'].astype(str))
    out = []
    for c in _consensus_rows(codes):
        s = c.get('segment_id')
        if not s or str(s) not in master or c.get('primary') is None:
            continue
        s = str(s)
        if s in oof_a1n and s in oB:
            out.append((int(c['primary']), map6(int(oof_a1n[s])), map6(int(oB[s])), part_of.get(s)))
    return out


# paired delta for C=4 (the LLM-axis pick) vs A1n
seg_ids, proba = cache[4]
oof4 = RD.ensemble_from_proba(seg_ids, proba, NC, 'softavg')
print('\n=== PAIRED Delta-kappa: ens_softavg_C4 vs A1n (cluster-bootstrap, n=3000) ===')
for axis, items in [('LLM ', llm_items(oof4)), ('HUM ', human_items(oof4))]:
    d = paired_delta(items)
    excl0 = (d['lo'] is not None and (d['lo'] > 0 or d['hi'] < 0))
    print(f'  {axis} n={len(items):3d}  Delta={d["point"]:+.4f}  '
          f'CI[{d["lo"]:+.4f},{d["hi"]:+.4f}]  {"RELIABLE" if excl0 else "overlaps 0"}', flush=True)
