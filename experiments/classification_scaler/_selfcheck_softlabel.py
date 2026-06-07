"""Self-check: reproduce A1n via the harness, and inspect the soft targets."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from experiments.gnn_reliability import harness as H
from experiments.gnn_reliability import baselines as B
from gnn_layer.config import GnnLayerConfig
from gnn_layer.soft_labels import build_soft_targets

ABS = 'data/Meta'


def main():
    print("== load corpus ==")
    df = H.load_corpus(ABS)
    print("rows:", len(df), "| participant rows:",
          int((df['speaker'] == 'participant').sum()))
    folds = H.build_folds(df, seed=42)
    print("folds over", len(folds), "labeled participant segments")

    print("\n== embeddings (qwen, cache) ==")
    emb = H.get_embeddings(df, 'qwen', ABS)
    dim = len(next(iter(emb.values())))
    print("n embeddings:", len(emb), "| dim:", dim)

    print("\n== SELF-CHECK: A1n = LinearProbe 6-class balanced ==")
    cfg = GnnLayerConfig(vaamr_n_classes=6, vaamr_class_balance=True, seed=42)
    oof = B.run_linear_probe(df, emb, folds, cfg)
    print("n oof preds:", len(oof))
    res = H.score_arm('SELFCHECK_A1n', oof, df, ABS, n_classes=6,
                      meta={'embedding': 'qwen', 'method': 'LinearProbe',
                            'imbalance': 'balanced', 'seed': 42,
                            'branch': 'scaler/soft-label'},
                      write_ledger=False)
    H._print_result(res)

    print("\n== INSPECT soft targets (weak, 6-class) ==")
    soft6 = build_soft_targets(df, 'weak', n_stages=6)
    print("n soft6 targets:", len(soft6))
    lab = df[(df['speaker'] == 'participant') & df['final_label'].notna()].copy()
    lab_sids = set(lab['segment_id'].astype(str))
    maxes, ents, peaky = [], [], 0
    for sid in lab_sids:
        v = np.asarray(soft6.get(sid))
        if v is None or v.sum() == 0:
            continue
        p = v / v.sum()
        maxes.append(p.max())
        nz = p[p > 0]
        ents.append(float(-(nz*np.log(nz)).sum()))
        if p.max() >= 0.999:
            peaky += 1
    maxes = np.array(maxes); ents = np.array(ents)
    print(f"labeled rows with soft target: {len(maxes)}")
    print(f"  max-prob (ballot agreement): mean={maxes.mean():.3f} "
          f"median={np.median(maxes):.3f} min={maxes.min():.3f}")
    print(f"  one-hot (peaky) rows: {peaky}/{len(maxes)} = {peaky/len(maxes):.1%}")
    print(f"  entropy: mean={ents.mean():.3f} max={ents.max():.3f}")
    print("  sample soft vectors (labeled, non-one-hot):")
    shown = 0
    for sid in sorted(lab_sids):
        v = np.asarray(soft6.get(sid))
        if v is None or v.sum() == 0:
            continue
        p = v / v.sum()
        if p.max() < 0.95:
            print("   ", sid, np.round(p, 3))
            shown += 1
        if shown >= 6:
            break


if __name__ == '__main__':
    main()
