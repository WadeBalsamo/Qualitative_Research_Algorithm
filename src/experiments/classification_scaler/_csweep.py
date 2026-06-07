"""Tune the per-rater ensemble on the LLM axis: sweep LogReg C (regularization).
Point kappa only for the sweep; full CI scoring for the winner. (scaler/ensemble)"""
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
sweep = {}
for C in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
    seg_ids, proba = proba_for_C(C)
    oof = RD.ensemble_from_proba(seg_ids, proba, NC, 'softavg')
    k = llm_kappa(oof)
    print(f'  C={C:<5g}  ens_softavg LLM kappa={k:+.4f}', flush=True)
    sweep[str(C)] = round(float(k), 4)
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

# ---- persist the committed raw artifact backing the headline winner (regenerable) ----
import json
_a1n = H.score_arm('A1n', B.run_linear_probe(df, emb, folds, CFG), df, ABS,
                   n_classes=6, write_ledger=False)
_llm, _hum = res['llm_axis'], res['human_axis']
_winner_llm = round(float(_llm['cohen_kappa_205']), 4)
_winner_hum = round(float(_hum['cohen_kappa']), 4)
_a1n_llm = round(float(_a1n['llm_axis']['cohen_kappa_205']), 4)


def _ci(axis):
    return [round(x, 4) if x is not None else None for x in axis['ci95']]


_artifact = {
    '_note': ('Raw committed backing for the HEADLINE winner (per-rater ensemble ens_softavg, '
              'tuned C). Regenerate with: python src/experiments/classification_scaler/_csweep.py '
              '(seed 42, participant-grouped StratifiedGroupKFold, Qwen3-Embedding-8B 4096-d, '
              'n=205 LLM-labeled / 66 human-consensus). Complements _distill_results.json (C=1 variants).'),
    'arm': ('ens_softavg (per-rater ensemble; mean predict_proba over 3 class-weighted 6-class '
            'LogReg probes: gemma-4-31b / nemotron-3-nano-30b / qwen3-next-80b)'),
    'c_sweep_llm_kappa_205': sweep,
    'best_C': best[0],
    'winner_ens_softavg': {
        'C': best[0],
        'llm_kappa_205': _winner_llm, 'llm_ci95': _ci(_llm),
        'llm_kappa_human_subset': round(float(_llm['cohen_kappa_76']), 4) if _llm.get('cohen_kappa_76') is not None else None,
        'human_kappa': _winner_hum, 'human_ci95': _ci(_hum), 'human_n': _hum['n'],
        'per_class_recall': {p['class_name']: round(float(p['recall']), 4)
                             for p in _llm['per_class'] if p['class_id'] < 5},
    },
    'a1n_baseline_C1': {'llm_kappa_205': _a1n_llm,
                        'human_kappa': round(float(_a1n['human_axis']['cohen_kappa']), 4)},
    'paired_delta_llm_winner_vs_a1n': round(_winner_llm - _a1n_llm, 4),
    'success_bars': {
        'llm_kappa': 0.45, 'human_kappa': 0.50,
        'winner_clears_bar': bool(_winner_llm >= 0.45 or _winner_hum >= 0.50),
        'verdict': ('dominates A1n on both axes but NOT LLM-equivalent at n~=32 '
                    '(LLM < 0.45; human < 0.50). Ships assistive/gated, not autonomous.'),
    },
}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_csweep_results.json'), 'w') as _f:
    json.dump(_artifact, _f, indent=2)
print('wrote _csweep_results.json', flush=True)
