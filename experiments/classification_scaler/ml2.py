import sys, json, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'src'); sys.path.insert(0, '.')
import numpy as np, pandas as pd
from gnn_layer.config import GnnLayerConfig
from experiments.gnn_reliability import harness as H, baselines as B
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


def main():
    ABS = 'data/Meta'
    df = H.load_corpus(ABS)
    folds = H.build_folds(df, seed=42, verbose=False)
    emb = H.get_embeddings(df, 'qwen', ABS)
    print(f"corpus {len(df)} segs, emb dim {len(next(iter(emb.values())))}", flush=True)
    results = []

    def score(name, preds, ncls, method):
        try:
            res = H.score_arm(name, preds, df, ABS, ncls,
                              meta={'embedding': 'qwen', 'method': method, 'seed': 42,
                                    'branch': 'scaler'},
                              write_ledger=False)
            llm, hum = res['llm_axis'], res['human_axis']
            row = dict(
                arm=name, method=method,
                llm_k=round(llm['cohen_kappa_205'], 3),
                llm_ci=[round(x, 3) for x in llm['ci95']],
                hum_k=round(hum['cohen_kappa'], 3) if hum['cohen_kappa'] is not None else None,
                hum_ci=[round(x, 3) if x is not None else None for x in hum['ci95']],
                hum_n=hum['n'],
                recall={r['class_name'][:6]: round(r['recall'], 2)
                        for r in llm['per_class']
                        if r['class_id'] < 5 and r['recall'] is not None},
            )
            results.append(row)
            print(f"  {name:22} LLM κ={row['llm_k']} {row['llm_ci']}  "
                  f"HUM κ={row['hum_k']} {row['hum_ci']} (n={row['hum_n']})", flush=True)
        except Exception as e:
            print(f"  {name} SCORE-ERR {e}", flush=True)

    def run_sklearn(ncls, make_clf):
        cfg = GnnLayerConfig(vaamr_n_classes=ncls, vaamr_class_balance=True)
        seg_ids, labels, _ = B._prepare_labeled(df, emb, cfg)
        label_of = dict(zip(seg_ids, labels))
        fold_of, fl = B._resolve_folds(seg_ids, df, folds)
        preds = {}
        for f, tr, te in B._iter_folds(seg_ids, fold_of, fl):
            ytr = [label_of[s] for s in tr]
            if len(set(ytr)) < 2:
                for s in te:
                    preds[s] = ytr[0]
                continue
            clf = make_clf()
            clf.fit(B._stack_l2(emb, tr), ytr)
            p = clf.predict(B._stack_l2(emb, te))
            for s, pr in zip(te, p):
                preds[s] = int(pr)
        return preds

    print("=== self-check (reproduce A1n) ===", flush=True)
    score('A1n_selfcheck',
          B.run_linear_probe(df, emb, folds,
                             GnnLayerConfig(vaamr_n_classes=6, vaamr_class_balance=True)),
          6, 'logreg-bal-6')

    print("=== S4 model capacity ===", flush=True)
    for name, mk in [
        ('S4_mlp256',
         lambda: MLPClassifier(hidden_layer_sizes=(256,), alpha=1.0, max_iter=300,
                               early_stopping=True, random_state=42)),
        ('S4_gbm',
         lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                l2_regularization=1.0,
                                                class_weight='balanced', random_state=42)),
        ('S4_calib_iso',
         lambda: CalibratedClassifierCV(
             LogisticRegression(max_iter=3000, class_weight='balanced'),
             method='isotonic', cv=3)),
    ]:
        try:
            score(name, run_sklearn(6, mk), 6, name)
        except Exception as e:
            print(f"  {name} RUN-ERR {e}", flush=True)

    print("=== S5 two-stage (No-code detector + 5-class stager) ===", flush=True)
    try:
        seg_ids, lab6, _ = B._prepare_labeled(
            df, emb, GnnLayerConfig(vaamr_n_classes=6))
        L = dict(zip(seg_ids, lab6))
        fo, fl = B._resolve_folds(seg_ids, df, folds)
        preds = {}
        for f, tr, te in B._iter_folds(seg_ids, fo, fl):
            yb = [1 if L[s] == 5 else 0 for s in tr]
            bclf = LogisticRegression(max_iter=3000, class_weight='balanced').fit(
                B._stack_l2(emb, tr), yb)
            tr5 = [s for s in tr if L[s] < 5]
            sclf = LogisticRegression(max_iter=3000, class_weight='balanced').fit(
                B._stack_l2(emb, tr5), [L[s] for s in tr5])
            for s in te:
                x = B._stack_l2(emb, [s])
                preds[s] = 5 if bclf.predict(x)[0] == 1 else int(sclf.predict(x)[0])
        score('S5_twostage', preds, 6, 'nocode-det+5stager')
    except Exception as e:
        print("  S5 ERR", e, flush=True)

    print("=== S7 human-mix (LLM + human up-weighted, grouped) ===", flush=True)
    try:
        hl = {str(r.segment_id): (5 if int(r.human_label) == -1 else int(r.human_label))
              for r in df.itertuples()
              if bool(getattr(r, 'in_human_coded_subset', False))
              and not pd.isna(r.human_label)}
        seg_ids, lab6, _ = B._prepare_labeled(
            df, emb, GnnLayerConfig(vaamr_n_classes=6))
        L = dict(zip(seg_ids, lab6))
        fo, fl = B._resolve_folds(seg_ids, df, folds)
        for W in [3.0, 10.0]:
            preds = {}
            for f, tr, te in B._iter_folds(seg_ids, fo, fl):
                y = [hl.get(s, L[s]) for s in tr]
                sw = [W if s in hl else 1.0 for s in tr]
                clf = LogisticRegression(max_iter=3000, class_weight='balanced').fit(
                    B._stack_l2(emb, tr), y, sample_weight=sw)
                for s in te:
                    preds[s] = int(clf.predict(B._stack_l2(emb, [s]))[0])
            score(f'S7_humanmix_w{int(W)}', preds, 6, f'human-upweight-{W}')
    except Exception as e:
        print("  S7 ERR", e, flush=True)

    _out = 'experiments/classification_scaler/scaler_model_results.json'
    json.dump(results, open(_out, 'w'), indent=2)
    print(f"\nDONE — results in {_out}", flush=True)
    return results


if __name__ == '__main__':
    main()
