"""Wave 2b — principled S5⊕S6: POOLED No-code gate (consensus) × per-rater 5-class ensemble stager.

Run from repo root:
    python experiments/classification_scaler/run_wave2b_hybrid.py

Design reference: experiments/docs/graph_experiments.md, experiments/docs/design_decisions.md
"""
import sys, os, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'src'); sys.path.insert(0, '.')
import numpy as np
from gnn_layer.config import GnnLayerConfig
from experiments.gnn_reliability import harness as H, baselines as B
from sklearn.linear_model import LogisticRegression

ABS = 'data/Meta'
RATERS = ['google/gemma-4-31b', 'nvidia/nemotron-3-nano-30b', 'qwen/qwen3-next-80b']


def _vote_stage(rv):
    v = rv.get('vote')
    if v == 'ABSTAIN': return -1
    if v == 'CODED': return rv.get('stage')
    if v is None and rv.get('stage') is not None: return rv.get('stage')
    return None


def participant_rater_labels(df_all, n=6):
    seg = []; per = {r: {} for r in RATERS}
    for _, r in df_all.iterrows():
        if str(r.get('speaker', '') or '') != 'participant': continue
        sid = str(r.get('segment_id')); rv = r.get('rater_votes')
        if not isinstance(rv, str) or not rv.strip() or rv.strip().lower() == 'nan': continue
        seg.append(sid)
        for b in json.loads(rv):
            rid = b.get('rater')
            if rid not in per: continue
            st = _vote_stage(b)
            if st is None: continue
            per[rid][sid] = (n - 1) if st < 0 else int(st)
    return seg, per


def score(df, name, preds):
    res = H.score_arm(name, preds, df, ABS, 6, meta={'embedding': 'qwen', 'method': name, 'seed': 42}, write_ledger=False)
    l, h = res['llm_axis'], res['human_axis']
    print(f"  {name:34s}: LLM κ={l['cohen_kappa_205']:.3f} {[round(x, 3) for x in l['ci95']]}  HUM κ={h['cohen_kappa']:.3f} {[round(x, 3) for x in h['ci95']]} (n={h['n']})", flush=True)


def main():
    df = H.load_corpus(ABS); folds = H.build_folds(df, seed=42, verbose=False)
    emb = {str(k): np.asarray(v, dtype=np.float32) for k, v in H.get_embeddings(df, 'qwen', ABS).items()}
    seg_ids, per_rater = participant_rater_labels(df, 6); seg_ids = [s for s in seg_ids if s in emb]
    fold_of, fold_list = B._resolve_folds(seg_ids, df, folds)
    cs_ids, lab6, _ = B._prepare_labeled(df, emb, GnnLayerConfig(vaamr_n_classes=6))
    CONS = dict(zip([str(x) for x in cs_ids], lab6))

    print("=== WAVE 2b: pooled No-code gate × per-rater 5-class ensemble (S5⊕S6) ===", flush=True)
    for C in (1.0, 4.0):
        preds = {}
        for _f, tr, te in B._iter_folds(seg_ids, fold_of, fold_list):
            trg = [s for s in tr if s in CONS]
            yb = [1 if CONS[s] == 5 else 0 for s in trg]
            gate = LogisticRegression(max_iter=4000, class_weight='balanced', C=C).fit(B._stack_l2(emb, trg), yb)
            Xte = B._stack_l2(emb, list(te))
            pno = gate.predict_proba(Xte)[:, list(gate.classes_).index(1)] if 1 in gate.classes_ else np.zeros(len(te))
            mats = []
            for r in RATERS:
                tr5 = [s for s in tr if per_rater[r].get(s, 9) < 5]
                if len(set(per_rater[r][s] for s in tr5)) >= 2:
                    stg = LogisticRegression(max_iter=4000, class_weight='balanced', C=C).fit(B._stack_l2(emb, tr5), [per_rater[r][s] for s in tr5])
                    ps = stg.predict_proba(Xte); cls = list(stg.classes_); M = np.zeros((len(te), 5))
                    for j, c in enumerate(cls):
                        if c < 5: M[:, c] = ps[:, j]
                    mats.append(M)
            ens = np.mean(mats, axis=0) if mats else np.full((len(te), 5), 1/5)
            ens = ens / ens.sum(1, keepdims=True).clip(min=1e-9)
            for i, s in enumerate(te):
                v = np.zeros(6); v[:5] = ens[i] * (1 - pno[i]); v[5] = pno[i]; preds[s] = int(np.argmax(v))
        score(df, f'W2b_pooledgate_x_perrater_C{C}', preds)
    print("DONE_WAVE2B", flush=True)


if __name__ == '__main__':
    main()
