"""Multi-seed confirmation of the LLM-axis champion (5-class soft MLP) + sharpening push."""
import os, sys, time
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
for _p in (os.path.join(_ROOT, 'src'), _ROOT):  # src/ ends up first on sys.path
    if _p not in sys.path:
        sys.path.insert(0, _p)
from experiments.gnn_reliability import harness as H
from experiments.classification_scaler import run_softlabel as S
ABS = S.ABS


def go(tag, nc, df, emb, folds, tau=1.0):
    hp = dict(hidden=256, n_hidden=1, dropout=0.4, in_dropout=0.1, lr=1e-3, wd=1e-3, epochs=100)
    t = time.time()
    pr = S.run_distill(df, emb, folds, nc, 'soft', hp, seeds=(0, 1, 2), tau=tau)
    r = H.score_arm(tag, pr, df, ABS, n_classes=nc, meta={}, write_ledger=False)
    l, h = r['llm_axis'], r['human_axis']
    rec = "/".join(f"{[x for x in l['per_class'] if x['class_id']==i][0]['recall']:.2f}" for i in range(5))
    print(f"{tag:<26} nc={nc} | LLM {l['cohen_kappa_205']:.3f}[{l['ci95'][0]:.3f},{l['ci95'][1]:.3f}] "
          f"| HUM {h['cohen_kappa']:.3f}[{h['ci95'][0]:.3f},{h['ci95'][1]:.3f}] | rec {rec} | {round(time.time()-t)}s", flush=True)


def main():
    df = H.load_corpus(ABS)
    folds = H.build_folds(df, seed=42, verbose=False)
    emb = H.get_embeddings(df, 'qwen', ABS)
    print(f"device={S.DEVICE}", flush=True)
    go('a5_soft_3seed', 5, df, emb, folds)
    go('a5_soft_tau0.5_3seed', 5, df, emb, folds, tau=0.5)
    go('a6_soft_3seed', 6, df, emb, folds)
    print("CONFIRM5 DONE.", flush=True)


if __name__ == '__main__':
    main()
