"""Paired Delta-kappa bootstrap: variant vs A1n on identical items, clustered by
participant (scaler/ensemble). Answers: is the gain reliable, or CI-overlap noise?"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '8'
import sys, dataclasses, json
_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo src/
for _p in (os.path.dirname(_SRC), _SRC):  # repo root then src/ -> src/ ends up first on sys.path
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
from experiments.gnn_reliability import harness as H, baselines as B
from experiments.classification_scaler import rater_distill as RD
from gnn_layer.config import GnnLayerConfig
from process import irr_import
from analysis import irr_stats, stats as _stats
from analysis.irr_analysis import _consensus_rows

ABS = 'data/Meta'
CFG = dataclasses.replace(GnnLayerConfig(), vaamr_n_classes=6, vaamr_class_balance=True)


def _map6(p):  # 6-class No-code(5) -> ABSTAIN(-1) for the human axis
    return -1 if p == 5 else p


# --- axis item builders: list of (ref, predA, predB, cluster) over common items ---
def llm_items(df, part_of, oA, oB):
    lab = H._labeled_participants(df)
    out = []
    for _, r in lab.iterrows():
        s = str(r['segment_id'])
        if s in oA and s in oB:
            out.append((int(r['final_label']), int(oA[s]), int(oB[s]), part_of.get(s)))
    return out


def human_items(df, part_of, oA, oB):
    codes = irr_import.read_human_codes(ABS)
    master = set(df['segment_id'].astype(str))
    out = []
    for c in _consensus_rows(codes):
        s = c.get('segment_id')
        if not s or str(s) not in master or c.get('primary') is None:
            continue
        s = str(s)
        if s in oA and s in oB:
            out.append((int(c['primary']), _map6(int(oA[s])), _map6(int(oB[s])), part_of.get(s)))
    return out


def paired_delta(items, n_boot=3000, seed=42):
    """Delta = kappa(predB,ref) - kappa(predA,ref). Cluster-bootstrap CI by participant.
    Pack (ref+1, pa+1, pb+1) each in [0,6] -> r*49+pa*7+pb (<343) as one float."""
    ref = np.array([t[0] for t in items]); pa = np.array([t[1] for t in items])
    pb = np.array([t[2] for t in items]); cl = [t[3] for t in items]
    packed = ((ref + 1) * 49 + (pa + 1) * 7 + (pb + 1)).astype(float)

    def stat(arr):
        c = arr.astype(int)
        rr = (c // 49) - 1
        pp = ((c % 49) // 7) - 1
        qq = (c % 7) - 1
        ka = irr_stats.cohen_kappa(pp.tolist(), rr.tolist())
        kb = irr_stats.cohen_kappa(qq.tolist(), rr.tolist())
        if ka is None or kb is None:
            return float('nan')
        return float(kb - ka)

    res = _stats.cluster_bootstrap_ci(packed, cl, statistic=stat, n_boot=n_boot, seed=seed)
    # exact point delta
    ka = irr_stats.cohen_kappa(pa.tolist(), ref.tolist())
    kb = irr_stats.cohen_kappa(pb.tolist(), ref.tolist())
    res['point'] = float(kb - ka)
    res['frac_pos'] = None
    return res


def main():
    df = H.load_corpus(ABS)
    folds = H.build_folds(df, seed=42, verbose=False)
    emb = H.get_embeddings(df, 'qwen', ABS)
    part_of = {str(r['segment_id']): str(r['participant_id']) for _, r in df.iterrows()}

    # --- OOF predictions for the three arms of interest ---
    print('computing OOF preds (A1n, ens_softavg, mlp_soft_kl) ...', flush=True)
    oof_a1n = B.run_linear_probe(df, emb, folds, CFG)
    seg_ids, proba = RD.per_rater_oof_proba(df, emb, folds, CFG)
    oof_soft = RD.ensemble_from_proba(seg_ids, proba, 6, 'softavg')
    oof_mlp = RD.run_mlp_soft(df, emb, folds, CFG)

    print('\n=== PAIRED Delta-kappa vs A1n (cluster-bootstrap by participant, n=3000) ===')
    print('(Delta>0 => variant beats A1n; CI excluding 0 => reliable)\n')
    for name, oB in [('ens_softavg', oof_soft), ('mlp_soft_kl', oof_mlp)]:
        for axis, builder in [('LLM ', llm_items), ('HUM ', human_items)]:
            items = builder(df, part_of, oof_a1n, oB)
            d = paired_delta(items)
            excl0 = (d['lo'] is not None and d['hi'] is not None and (d['lo'] > 0 or d['hi'] < 0))
            print(f'{name:14s} {axis} n={len(items):3d}  Delta={d["point"]:+.4f}  '
                  f'CI[{d["lo"]:+.4f},{d["hi"]:+.4f}]  {"RELIABLE" if excl0 else "overlaps 0"}')
    print()


if __name__ == '__main__':
    main()
