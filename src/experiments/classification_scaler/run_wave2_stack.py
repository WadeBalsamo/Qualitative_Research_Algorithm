"""Wave 2 — S5⊕S6 stack: per-rater TWO-STAGE (No-code gate→5-class) probes, soft-averaged."""
import sys, os, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'src'); sys.path.insert(0,'.')
import numpy as np
from gnn_layer.config import GnnLayerConfig
from experiments.gnn_reliability import harness as H, baselines as B
from sklearn.linear_model import LogisticRegression
ABS='data/Meta'
RATERS=['google/gemma-4-31b','nvidia/nemotron-3-nano-30b','qwen/qwen3-next-80b']
def _vote_stage(rv):
    v=rv.get('vote')
    if v=='ABSTAIN': return -1
    if v=='CODED': return rv.get('stage')
    if v is None and rv.get('stage') is not None: return rv.get('stage')
    return None
def participant_rater_labels(df_all,n=6):
    seg=[]; per={r:{} for r in RATERS}
    for _,r in df_all.iterrows():
        if str(r.get('speaker','') or '')!='participant': continue
        sid=str(r.get('segment_id')); rv=r.get('rater_votes')
        if not isinstance(rv,str) or not rv.strip() or rv.strip().lower()=='nan': continue
        seg.append(sid)
        for b in json.loads(rv):
            rid=b.get('rater')
            if rid not in per: continue
            st=_vote_stage(b)
            if st is None: continue
            per[rid][sid]=(n-1) if st<0 else int(st)
    return seg,per
df=H.load_corpus(ABS); folds=H.build_folds(df,seed=42,verbose=False)
emb={str(k):np.asarray(v,dtype=np.float32) for k,v in H.get_embeddings(df,'qwen',ABS).items()}
seg_ids,per_rater=participant_rater_labels(df,6); seg_ids=[s for s in seg_ids if s in emb]
fold_of,fold_list=B._resolve_folds(seg_ids,df,folds)
def twostage_proba(train_ids,label_of,test_ids,C):
    tr=[s for s in train_ids if s in label_of and s in emb]; te=list(test_ids)
    if not tr or 1>len(set(label_of[s] for s in tr)): return {s:np.full(6,1/6) for s in te}
    yb=[1 if label_of[s]==5 else 0 for s in tr]
    gate=LogisticRegression(max_iter=4000,class_weight='balanced',C=C).fit(B._stack_l2(emb,tr),yb)
    Xte=B._stack_l2(emb,te)
    pno=gate.predict_proba(Xte)[:,list(gate.classes_).index(1)] if 1 in gate.classes_ else np.zeros(len(te))
    tr5=[s for s in tr if label_of[s]<5]; out={}
    if len(set(label_of[s] for s in tr5))>=2:
        stg=LogisticRegression(max_iter=4000,class_weight='balanced',C=C).fit(B._stack_l2(emb,tr5),[label_of[s] for s in tr5])
        ps=stg.predict_proba(Xte); cls=list(stg.classes_)
        for i,s in enumerate(te):
            v=np.zeros(6)
            for j,c in enumerate(cls): v[c]=ps[i,j]
            v=v*(1-pno[i]); v[5]=pno[i]; out[s]=v
    else:
        only=list(set(label_of[s] for s in tr5))
        for i,s in enumerate(te):
            v=np.zeros(6); v[5]=pno[i]
            if only: v[only[0]]=1-pno[i]
            out[s]=v
    return out
def score(name,preds):
    res=H.score_arm(name,preds,df,ABS,6,meta={'embedding':'qwen','method':name,'seed':42},write_ledger=False)
    l,h=res['llm_axis'],res['human_axis']
    print(f"  {name:30s}: LLM κ={l['cohen_kappa_205']:.3f} {[round(x,3) for x in l['ci95']]}  HUM κ={h['cohen_kappa']:.3f} {[round(x,3) for x in h['ci95']]} (n={h['n']})",flush=True)
print("=== WAVE 2: per-rater TWO-STAGE soft-avg (S5⊕S6) vs A1n .283/.365 · S6 .361/.450 · S5 .281/.447 ===",flush=True)
for C in (1.0,4.0):
    proba={r:{} for r in RATERS}
    for _f,tr,te in B._iter_folds(seg_ids,fold_of,fold_list):
        for r in RATERS: proba[r].update(twostage_proba(tr,per_rater[r],te,C))
    preds={}
    for s in seg_ids:
        if all(s in proba[r] for r in RATERS):
            preds[s]=int(np.argmax(sum(proba[r][s] for r in RATERS)/3))
    score(f'W2_perrater_twostage_C{C}',preds)
print("DONE_WAVE2",flush=True)
