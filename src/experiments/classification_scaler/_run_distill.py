"""Runner for the per-rater distillation battery (scaler/ensemble). NOT committed."""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '8'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys, time, dataclasses, json
_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo src/
for _p in (os.path.dirname(_SRC), _SRC):  # repo root then src/ -> src/ ends up first on sys.path
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from experiments.gnn_reliability import harness as H, baselines as B
from experiments.classification_scaler import rater_distill as RD
from gnn_layer.config import GnnLayerConfig
from process import irr_import
from analysis import irr_stats
from analysis.irr_analysis import _consensus_rows

ABS = '/home/wisgood/qra/Qualitative_Research_Algorithm/data/Meta'
CFG = dataclasses.replace(GnnLayerConfig(), vaamr_n_classes=6, vaamr_class_balance=True)
A1N_LLM, A1N_HUM = 0.2831, 0.3652

t0 = time.time()
print('loading corpus + embeddings ...', flush=True)
df = H.load_corpus(ABS)
folds = H.build_folds(df, seed=42, verbose=False)
emb = H.get_embeddings(df, 'qwen', ABS)
print(f'  ready ({time.time()-t0:.1f}s)', flush=True)

# ---- per-rater human-subset kappa (for the (c) weighting variant) ----
codes = irr_import.read_human_codes(ABS)
consensus = _consensus_rows(codes)
human_primary = {str(c['segment_id']): int(c['primary']) for c in consensus
                 if c.get('segment_id') and c.get('primary') is not None}
seg_all, per_rater, soft = RD.participant_rater_labels(df, n_classes=6)
rater_human_kappa = {}
for r in RD.RATERS:
    h, m = [], []
    for sid, hp in human_primary.items():
        if sid in per_rater[r]:
            lab = per_rater[r][sid]
            h.append(hp); m.append(-1 if lab == 5 else lab)
    rater_human_kappa[r] = (irr_stats.cohen_kappa(h, m), len(h))
print('\nper-rater kappa vs human subset (66 consensus items):', flush=True)
for r in RD.RATERS:
    k, n = rater_human_kappa[r]
    print(f'  {r:30s} kappa={k:+.4f}  n={n}', flush=True)
hw = {r: max(rater_human_kappa[r][0], 0.05) for r in RD.RATERS}
print('  -> human-kappa softavg weights:',
      {r.split("/")[-1]: round(v, 3) for r, v in hw.items()}, flush=True)

# ---- fit the 15 per-rater probes ONCE ----
print('\nfitting 3 per-rater probes x 5 folds ...', flush=True)
t = time.time()
seg_ids, proba = RD.per_rater_oof_proba(df, emb, folds, CFG)
print(f'  done ({time.time()-t:.1f}s)', flush=True)

results = []  # (name, res)

def add(name, oof, method):
    t = time.time()
    res = H.score_arm(name, oof, df, ABS, n_classes=6,
                      meta={'embedding': 'qwen', 'method': method, 'seed': 42,
                            'branch': 'scaler/ensemble'}, write_ledger=False)
    results.append((name, res))
    llm = res['llm_axis']; hum = res['human_axis']
    print(f'  [{name:22s}] LLM={llm["cohen_kappa_205"]:+.3f} '
          f'[{(llm["ci95"][0] or float("nan")):+.3f},{(llm["ci95"][1] or float("nan")):+.3f}]  '
          f'HUM={hum["cohen_kappa"]:+.3f} '
          f'[{(hum["ci95"][0] or float("nan")):+.3f},{(hum["ci95"][1] or float("nan")):+.3f}]  '
          f'({time.time()-t:.1f}s)', flush=True)
    return res

print('\nscoring variants ...', flush=True)
add('A1n', B.run_linear_probe(df, emb, folds, CFG), 'probe6w')
for r in RD.RATERS:
    short = r.split('/')[-1]
    add(f'per-rater:{short}', RD.per_rater_argmax(seg_ids, proba, r), 'probe_rater')
add('ens_majority', RD.ensemble_from_proba(seg_ids, proba, 6, 'majority'), 'ens_maj')
add('ens_softavg', RD.ensemble_from_proba(seg_ids, proba, 6, 'softavg'), 'ens_soft')
add('ens_softavg_w', RD.ensemble_from_proba(seg_ids, proba, 6, 'softavg', weights=hw), 'ens_soft_w')
print('  training MLPs (4096->128, full-batch) ...', flush=True)
add('mlp_hard', RD.run_mlp_hard(df, emb, folds, CFG), 'mlp_hard')
add('mlp_soft_kl', RD.run_mlp_soft(df, emb, folds, CFG), 'mlp_soft')

# ---- summary table ----
def kfmt(v):
    return '  .   ' if v is None else f'{v:+.3f}'

print('\n' + '=' * 120, flush=True)
print(f'A1n baseline: LLM kappa={A1N_LLM:+.3f}  HUM kappa={A1N_HUM:+.3f}  | bar: LLM>=0.45 OR HUM>=0.50 (CI-aware)')
print('-' * 120)
print(f'{"variant":22s} | {"LLM kappa [95% CI]":26s} | {"HUM kappa [95% CI]":26s} | {"beats A1n":10s} | bar?')
print('-' * 120)
for name, res in results:
    llm = res['llm_axis']; hum = res['human_axis']
    lk, lci = llm['cohen_kappa_205'], llm['ci95']
    hk, hci = hum['cohen_kappa'], hum['ci95']
    beats = '+'.join([x for x in (('LLM' if lk > A1N_LLM + 1e-9 else ''),
                                  ('HUM' if hk > A1N_HUM + 1e-9 else '')) if x]) or 'no'
    bar = ','.join([x for x in (('LLM>=.45' if lk >= 0.45 else ''),
                                ('HUM>=.50' if hk >= 0.50 else '')) if x]) or '-'
    print(f'{name:22s} | {lk:+.3f} [{kfmt(lci[0])},{kfmt(lci[1])}]      | '
          f'{hk:+.3f} [{kfmt(hci[0])},{kfmt(hci[1])}]      | {beats:10s} | {bar}')
print('=' * 120)

# ---- per-class recall (LLM axis) ----
names5 = ['Vig', 'Avoid', 'AttReg', 'Metacog', 'Reapp']
print('\nper-class recall (LLM axis vs final_label, 205):')
print(f'{"variant":22s} | ' + ' '.join(f'{n:>8s}' for n in names5) + f' | {"n205":>4s} {"nHum":>4s}')
print('-' * 92)
for name, res in results:
    pc = {p['class_id']: p['recall'] for p in res['llm_axis']['per_class']}
    row = ' '.join(f'{(pc.get(i) if pc.get(i) is not None else float("nan")):8.3f}' for i in range(5))
    print(f'{name:22s} | {row} | {res["llm_axis"]["n"]:4d} {res["human_axis"]["n"]:4d}')

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_distill_results.json'), 'w') as f:
    json.dump({name: {'llm': res['llm_axis']['cohen_kappa_205'], 'llm_ci': res['llm_axis']['ci95'],
                      'llm_76': res['llm_axis']['cohen_kappa_76'],
                      'hum': res['human_axis']['cohen_kappa'], 'hum_ci': res['human_axis']['ci95'],
                      'hum_n': res['human_axis']['n'], 'llm_n': res['llm_axis']['n'],
                      'recall': {p['class_name']: p['recall'] for p in res['llm_axis']['per_class']}}
               for name, res in results}, f, indent=2)
print(f'\n[done in {time.time()-t0:.1f}s]', flush=True)
