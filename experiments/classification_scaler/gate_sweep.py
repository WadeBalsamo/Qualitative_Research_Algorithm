"""Gate-threshold sweep for the best two-stage variant (No-code gate + multinomial
stager). The gate threshold trades the LLM axis (kappa over the truly-labeled 205;
hurt by gating a real-VAAMR seg to No-code) against the human axis (where No-code=-1
is a real category the gate can recover). Tuned/observed on BOTH axes; reused harness
scoring so kappa is comparable to A1n."""
import sys, os
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from experiments.gnn_reliability import harness as H
from experiments.classification_scaler.ordinal_twostage import oof_two_stage, adjacency_stats, _fmt, ABS, SEED


def main():
    df = H.load_corpus(ABS)
    folds = H.build_folds(df, seed=SEED, verbose=False)
    emb = H.get_embeddings(df, 'qwen', ABS)
    print(f"{'gate_thr':>8} {'method':<11} | {'LLM k205':>8} {'[lo':>6} {'hi]':>6} | "
          f"{'HUM k':>6} {'[lo':>6} {'hi]':>6} hN | gMiss")
    print('-' * 86)
    for method in ('multinomial', 'ordbin'):
        for thr in (0.35, 0.45, 0.50, 0.55, 0.65):
            oof = oof_two_stage(df, emb, folds, method, gate_thr=thr)
            res = H.score_arm(f'ts_{method}_t{thr}', oof, df, ABS, n_classes=6,
                              meta={'embedding': 'qwen', 'seed': SEED}, write_ledger=False)
            adjs = adjacency_stats(oof, df)
            llm, hum = res['llm_axis'], res['human_axis']
            print(f"{thr:>8.2f} {method:<11} | {_fmt(llm['cohen_kappa_205']):>8} "
                  f"{_fmt(llm['ci95'][0]):>6} {_fmt(llm['ci95'][1]):>6} | "
                  f"{_fmt(hum['cohen_kappa']):>6} {_fmt(hum['ci95'][0]):>6} "
                  f"{_fmt(hum['ci95'][1]):>6} {hum['n']:>2} | {adjs['gate_miss']:>4}", flush=True)


if __name__ == '__main__':
    main()
