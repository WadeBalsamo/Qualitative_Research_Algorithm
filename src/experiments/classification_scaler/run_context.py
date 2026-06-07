"""S1 context lever — ready to fire when 10.0.0.58 is back. Resumable small-batch embed."""
import sys, os, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'src'); sys.path.insert(0,'.')
import numpy as np, pandas as pd
from gnn_layer.config import GnnLayerConfig
from gnn_layer import embeddings_remote as er
from experiments.gnn_reliability import harness as H, baselines as B
from sklearn.linear_model import LogisticRegression
ABS='data/Meta'
df=H.load_corpus(ABS); folds=H.build_folds(df,seed=42,verbose=False)
raw=pd.read_csv(os.path.join(ABS,'02_meta/training_data/master_segments.csv'))
raw['segment_id']=raw['segment_id'].astype(str); raw['start_time_ms']=raw['start_time_ms'].fillna(0).astype(int)
bysess={s:g.sort_values('start_time_ms') for s,g in raw.groupby('session_id')}
def ctx_of(r,k=6):
    g=bysess[r.session_id]; pri=g[g.start_time_ms<r.start_time_ms].tail(k)
    return "\n".join(f"[{x.speaker}]: {str(x.text)[:1200]}" for x in pri.itertuples())
part=raw[raw.speaker=='participant']
texts={r.segment_id: (ctx_of(r)[:3500]+"\n>>> [TARGET]: "+str(r.text)[:4000]) for r in part.itertuples()}
CACHE='/tmp/ctx_combined.npz'
cfg=GnnLayerConfig(embedding_backend='openai',embedding_base_url='http://10.0.0.58:1234/v1',
                   embedding_model='text-embedding-qwen3-embedding-8b',use_query_prefix=True,embedding_batch_size=4)
cvec={}
if os.path.isfile(CACHE):
    d=np.load(CACHE,allow_pickle=True); cvec={str(s):v for s,v in zip(d['segment_ids'],d['embeddings'])}
ids=list(texts); todo=[s for s in ids if s not in cvec]
print(f"context embed: {len(todo)}/{len(ids)} to do (resumable)",flush=True)
import time as _t
for s in range(0,len(todo),8):
    ch=todo[s:s+8]
    v=np.asarray(er.embed_texts_remote([texts[i][:5000] for i in ch],cfg,is_query=True,timeout=150,max_retries=5),dtype=np.float32)
    for k,i in enumerate(ch): cvec[i]=v[k]
    np.savez_compressed(CACHE,segment_ids=np.array(list(cvec)),embeddings=np.stack([cvec[i] for i in cvec]))
    print(f"  embedded {min(s+8,len(todo))}/{len(todo)}",flush=True)
    _t.sleep(0.4)
print(f"context features ready (dim {len(next(iter(cvec.values())))})",flush=True)
emb=dict(cvec)
def score(name,preds,ncls):
    res=H.score_arm(name,preds,df,ABS,ncls,meta={'embedding':'qwen-ctx','method':name,'seed':42,'branch':'scaler-ctx'},write_ledger=False)
    l,h=res['llm_axis'],res['human_axis']
    print(f"  {name}: LLM κ={l['cohen_kappa_205']:.3f} {[round(x,3) for x in l['ci95']]}  HUM κ={h['cohen_kappa']:.3f} {[round(x,3) for x in h['ci95']]} (n={h['n']})",flush=True)
print("=== S1 context results (vs A1n 0.283/0.365 ; S5 two-stage human 0.447) ===",flush=True)
score('S1_ctx_probe', B.run_linear_probe(df,emb,folds,GnnLayerConfig(vaamr_n_classes=6,vaamr_class_balance=True)), 6)
# two-stage No-code gate on CONTEXT features (stack the two winning levers)
seg_ids,lab6,_=B._prepare_labeled(df,emb,GnnLayerConfig(vaamr_n_classes=6)); L=dict(zip(seg_ids,lab6)); fo,fl=B._resolve_folds(seg_ids,df,folds)
preds={}
for f,tr,te in B._iter_folds(seg_ids,fo,fl):
    bclf=LogisticRegression(max_iter=3000,class_weight='balanced').fit(B._stack_l2(emb,tr),[1 if L[s]==5 else 0 for s in tr])
    tr5=[s for s in tr if L[s]<5]; sclf=LogisticRegression(max_iter=3000,class_weight='balanced').fit(B._stack_l2(emb,tr5),[L[s] for s in tr5])
    for s in te:
        x=B._stack_l2(emb,[s]); preds[s]=5 if bclf.predict(x)[0]==1 else int(sclf.predict(x)[0])
score('S1_ctx_twostage', preds, 6)
print("DONE — context experiment complete",flush=True)
