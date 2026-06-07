"""experiments/gnn_reliability/run_softlabel_grid.py
------------------------------------------------------
Comprehensive soft-label distillation table + levers, on the full 4096-d Qwen
features (CPU; GPU is held by an external server). Each result is appended to
_softlabel_results.jsonl AND printed immediately, ordered so the decisive rows
(6-class a/b/c) land first. Run unbuffered.

Baseline A1n (validated in self-check): HUM kappa=0.365 [0.228,0.513],
LLM kappa(205)=0.283 [0.203,0.346].
Bar: classifier<->LLM grouped kappa >= ~0.45 OR classifier<->human kappa >= ~0.50.
"""
import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from experiments.gnn_reliability import harness as H
from experiments.gnn_reliability import baselines as B
from experiments.gnn_reliability import run_softlabel as S
from gnn_layer.config import GnnLayerConfig

ABS = S.ABS
OUT = os.path.join(_HERE, '_softlabel_results.jsonl')
SEEDS = (0,)
EP = 80


def log(rec):
    with open(OUT, 'a') as f:
        f.write(json.dumps(rec) + "\n")
    if 'name' in rec:
        b_llm = "Y" if (rec['llm_k'] >= 0.45) else "."
        b_hum = "Y" if (rec['hum_k'] >= 0.50) else "."
        beats = "BEATS-A1n" if (rec['llm_k'] > 0.283 or rec['hum_k'] > 0.365) else ""
        print(f"  {rec['name']:<22} nc={rec['nc']} | LLM={rec['llm_k']:.3f} "
              f"[{rec['llm_lo']:.3f},{rec['llm_hi']:.3f}] | HUM={rec['hum_k']:.3f} "
              f"[{rec['hum_lo']:.3f},{rec['hum_hi']:.3f}] n{rec['hum_n']} | "
              f"bar[L{b_llm}/H{b_hum}] {beats} | rec={rec['recall']} | {rec['secs']}s",
              flush=True)
    else:
        print(json.dumps(rec), flush=True)


def ev(name, preds, df, nc, method, note, t0):
    res = H.score_arm(name, preds, df, ABS, n_classes=nc,
                      meta={'embedding': 'qwen', 'method': method, 'imbalance': 'balanced',
                            'seed': 42, 'branch': 'scaler/soft-label', 'notes': note},
                      write_ledger=False)
    llm, hum = res['llm_axis'], res['human_axis']
    rec = "/".join(f"{[r for r in llm['per_class'] if r['class_id']==i][0]['recall']:.2f}"
                   for i in range(5))
    r = {'name': name, 'nc': nc, 'method': method, 'note': note,
         'llm_k': round(llm['cohen_kappa_205'], 4), 'llm_lo': round(llm['ci95'][0], 4),
         'llm_hi': round(llm['ci95'][1], 4), 'hum_k': round(hum['cohen_kappa'], 4),
         'hum_lo': round(hum['ci95'][0], 4), 'hum_hi': round(hum['ci95'][1], 4),
         'hum_n': hum['n'], 'recall': rec, 'secs': round(time.time() - t0, 1)}
    log(r)
    return r


def mlp(df, emb, folds, nc, mode, **kw):
    hp = dict(hidden=kw.get('hidden', 256), n_hidden=kw.get('n_hidden', 1),
              dropout=kw.get('dropout', 0.4), in_dropout=kw.get('in_dropout', 0.1),
              lr=kw.get('lr', 1e-3), wd=kw.get('wd', 1e-3), epochs=kw.get('epochs', EP))
    return S.run_distill(df, emb, folds, nc, mode, hp, seeds=SEEDS,
                         tau=kw.get('tau', 1.0), smooth=kw.get('smooth', 0.0),
                         pca=kw.get('pca', None))


if __name__ == '__main__':
    open(OUT, 'w').close()
    print(f"loading ... (epochs={EP}, seeds={SEEDS}, device={S.DEVICE})", flush=True)
    df = H.load_corpus(ABS)
    folds = H.build_folds(df, seed=42, verbose=False)
    emb = H.get_embeddings(df, 'qwen', ABS)
    print("A1n baseline: HUM=0.365 [0.228,0.513]  LLM=0.283 [0.203,0.346]\n"
          "=== core table (full 4096-d) ===", flush=True)

    # --- decisive 6-class trio first ---
    t = time.time(); ev('a_MLP-KL_soft', mlp(df, emb, folds, 6, 'soft'), df, 6, 'MLP-KL', 'soft', t)
    t = time.time(); ev('b_MLP-CE_hard', mlp(df, emb, folds, 6, 'hard'), df, 6, 'MLP-CE', 'hard', t)
    t = time.time(); ev('c_ConfProbe', S.run_conf_probe(df, emb, folds, 6, balance=True), df, 6, 'ConfProbe', 'conf-wt', t)
    # control: linear-KL soft (A1n capacity, soft loss) — isolates soft signal w/o MLP capacity
    t = time.time(); ev('d_LinKL_soft', mlp(df, emb, folds, 6, 'soft', n_hidden=0, hidden=0, dropout=0.0, in_dropout=0.0, epochs=200), df, 6, 'Lin-KL', 'soft-linear', t)

    # --- 5-class trio ---
    t = time.time(); ev('a_MLP-KL_soft', mlp(df, emb, folds, 5, 'soft'), df, 5, 'MLP-KL', 'soft', t)
    t = time.time(); ev('b_MLP-CE_hard', mlp(df, emb, folds, 5, 'hard'), df, 5, 'MLP-CE', 'hard', t)
    t = time.time(); ev('c_ConfProbe', S.run_conf_probe(df, emb, folds, 5, balance=True), df, 5, 'ConfProbe', 'conf-wt', t)

    # --- levers on the soft MLP (6-class), trying to clear the bar ---
    print("=== levers (6-class MLP-KL soft) ===", flush=True)
    for kw in [dict(tau=0.5), dict(tau=2.0), dict(smooth=0.3), dict(hidden=512),
               dict(hidden=128, dropout=0.3), dict(wd=3e-3), dict(n_hidden=2)]:
        tag = "+".join(f"{k}{v}" for k, v in kw.items())
        t = time.time(); ev(f'lever_{tag}', mlp(df, emb, folds, 6, 'soft', **kw), df, 6, 'MLP-KL', tag, t)

    print("GRID DONE.", flush=True)
