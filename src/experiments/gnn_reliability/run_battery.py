"""
experiments/gnn_reliability/run_battery.py
------------------------------------------
Architect orchestration for the pre-registered VAAMR reliability battery
(design_decisions.md §6). Builds the corpus + participant-grouped folds + Qwen
embeddings ONCE, then runs every arm on the SAME folds and scores both axes
(LLM κ over 205 + human κ over 66) via the shared harness scorer + ledger.

Arms (A0 = MiniLM GraphSAGE baseline is run by the harness self-check separately):
  A1   Qwen linear probe (no graph)
  A1w  Qwen linear probe + class-weighted
  A1n  Qwen linear probe + class-weighted + 6-class (No code)
  A2   Qwen Correct-&-Smooth (5-class)
  A2n  Qwen Correct-&-Smooth (6-class)
  A3   Qwen GraphSAGE (no imbalance handling)
  A4   Qwen GraphSAGE + class-balance + focal
  A4n  Qwen GraphSAGE + class-balance + focal + 6-class (No code)

Run:  python experiments/gnn_reliability/run_battery.py            # full battery
      python experiments/gnn_reliability/run_battery.py A3 A4 A4n  # subset
All numbers go to docs/gnn_experiments/ledger.csv (append) + stdout.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

OUT = os.environ.get('QRA_BATTERY_OUTPUT', 'data/Meta')

# Qwen-via-LM-Studio embedding settings (the cache is pre-warmed; this is a hit).
QWEN = dict(embedding_backend='openai',
            embedding_base_url='http://10.0.0.58:1234/v1',
            embedding_model='text-embedding-qwen3-embedding-8b',
            use_query_prefix=True, embedding_batch_size=8)


def _qcfg(**kw):
    from gnn_layer.config import GnnLayerConfig
    return GnnLayerConfig(**{**QWEN, **kw})


def _arms():
    """(name, kind, config, n_classes) for each pre-registered arm. kind ∈ probe|cs|gnn."""
    return [
        ('A1',  'probe', _qcfg(vaamr_n_classes=5), 5),
        ('A1w', 'probe', _qcfg(vaamr_n_classes=5, vaamr_class_balance=True), 5),
        ('A1n', 'probe', _qcfg(vaamr_n_classes=6, vaamr_class_balance=True), 6),
        ('A2',  'cs',    _qcfg(vaamr_n_classes=5, vaamr_class_balance=True), 5),
        ('A2n', 'cs',    _qcfg(vaamr_n_classes=6, vaamr_class_balance=True), 6),
        ('A3',  'gnn',   _qcfg(vaamr_n_classes=5), 5),
        ('A4',  'gnn',   _qcfg(vaamr_n_classes=5, vaamr_class_balance=True,
                               vaamr_focal_gamma=2.0), 5),
        ('A4n', 'gnn',   _qcfg(vaamr_n_classes=6, vaamr_class_balance=True,
                               vaamr_focal_gamma=2.0), 6),
    ]


_METHOD = {'probe': 'LinearProbe', 'cs': 'Correct&Smooth', 'gnn': 'GraphSAGE'}


def main(only=None):
    from experiments.gnn_reliability import harness as H
    from experiments.gnn_reliability import baselines as B

    df = H.load_corpus(OUT)
    folds = H.build_folds(df, verbose=True)
    qemb = H.get_embeddings(df, 'qwen', OUT)
    dim = int(len(next(iter(qemb.values())))) if qemb else None
    print(f"Qwen embeddings: n={len(qemb)} dim={dim}\n")

    for name, kind, cfg, ncls in _arms():
        if only and name not in only:
            continue
        print(f"\n>>> running arm {name} ({kind}, n_classes={ncls}) ...")
        if kind == 'probe':
            oof = B.run_linear_probe(df, qemb, folds, cfg)
        elif kind == 'cs':
            oof = B.run_correct_smooth(df, qemb, folds, cfg)
        else:
            oof = H.run_gnn_arm(df, qemb, folds, cfg)
        imb = ('balanced' if getattr(cfg, 'vaamr_class_balance', False) else 'none')
        if float(getattr(cfg, 'vaamr_focal_gamma', 0.0) or 0.0) > 0:
            imb += '+focal'
        meta = dict(embedding='qwen', embed_dim=dim, method=_METHOD[kind],
                    imbalance=imb, seed=int(cfg.seed), branch=H._git_branch())
        res = H.score_arm(name, oof, df, OUT, ncls, meta=meta, write_ledger=True)
        H._print_result(res)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    main(only=set(args) if args else None)
