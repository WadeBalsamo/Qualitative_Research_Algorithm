"""QWEN3 concat context test — cached A1n target ⊕ freshly-embedded context. Endpoint-gentle."""
import sys, os, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'src'); sys.path.insert(0,'.')
import numpy as np, pandas as pd, time as _t
from gnn_layer.config import GnnLayerConfig
from gnn_layer import embeddings_remote as er
from experiments.gnn_reliability import harness as H, baselines as B
from sklearn.linear_model import LogisticRegression
ABS='data/Meta'
df=H.load_corpus(ABS); folds=H.build_folds(df,seed=42,verbose=False)
tgt={str(k):np.asarray(v,dtype=np.float32) for k,v in H.get_embeddings(df,'qwen',ABS).items()}
raw=pd.read_csv(os.path.join(ABS,'02_meta/training_data/master_segments.csv'))
raw['segment_id']=raw['segment_id'].astype(str); raw['start_time_ms']=raw['start_time_ms'].fillna(0).astype(int)
bysess={s:g.sort_values('start_time_ms') for s,g in raw.groupby('session_id')}
def ctx_of(r,k=6):
    g=bysess[r.session_id]; pri=g[g.start_time_ms<r.start_time_ms].tail(k)
    return "\n".join(f"[{x.speaker}]: {str(x.text)[:1000]}" for x in pri.itertuples())
part=raw[raw.speaker=='participant']
ctx_txt={r.segment_id:(ctx_of(r) or "[no prior context]")[:4500] for r in part.itertuples() if r.segment_id in tgt}
ids=[s for s in ctx_txt if s in tgt]
CACHE='/tmp/ctx_only_qwen.npz'; cvec={}
if os.path.isfile(CACHE):
    d=np.load(CACHE,allow_pickle=True); cvec={str(s):v for s,v in zip(d['segment_ids'],d['embeddings'])}
todo=[s for s in ids if s not in cvec]
cfg=GnnLayerConfig(embedding_backend='openai',embedding_base_url='http://10.0.0.58:1234/v1',
                   embedding_model='text-embedding-qwen3-embedding-8b',use_query_prefix=True,embedding_batch_size=4)
print(f"context-only embed (qwen): {len(todo)}/{len(ids)} to do (resumable)",flush=True)
for s in range(0,len(todo),8):
    ch=todo[s:s+8]
    v=np.asarray(er.embed_texts_remote([ctx_txt[i][:5000] for i in ch],cfg,is_query=True,timeout=150,max_retries=5),dtype=np.float32)
    for k,i in enumerate(ch): cvec[i]=v[k]
    np.savez_compressed(CACHE,segment_ids=np.array(list(cvec)),embeddings=np.stack([cvec[i] for i in cvec]))
    print(f"  {min(s+8,len(todo))}/{len(todo)}",flush=True); _t.sleep(0.4)
print(f"context embedded (dim {len(next(iter(cvec.values())))}); scoring",flush=True)
iso={s:tgt[s] for s in ids}
cat={s:np.concatenate([tgt[s],cvec[s]]) for s in ids}
def score(name,emb_or_preds,ncls,is_preds=False):
    preds = emb_or_preds if is_preds else B.run_linear_probe(df,emb_or_preds,folds,GnnLayerConfig(vaamr_n_classes=ncls,vaamr_class_balance=True))
    res=H.score_arm(name,preds,df,ABS,ncls,meta={'embedding':'qwen-concat','method':name,'seed':42},write_ledger=False)
    l,h=res['llm_axis'],res['human_axis']
    print(f"  {name:22s}: LLM κ={l['cohen_kappa_205']:.3f} {[round(x,3) for x in l['ci95']]}  HUM κ={h['cohen_kappa']:.3f} {[round(x,3) for x in h['ci95']]}",flush=True)
print("=== QWEN3 context concat (target⊕context, 8192-d) vs A1n 0.283/0.365 ===",flush=True)
score('Q_isolated_chk', iso, 6)
score('Q_concat_ctx', cat, 6)
seg_ids,lab6,_=B._prepare_labeled(df,cat,GnnLayerConfig(vaamr_n_classes=6)); L=dict(zip(seg_ids,lab6)); fo,fl=B._resolve_folds(seg_ids,df,folds)
preds={}
for f,tr,te in B._iter_folds(seg_ids,fo,fl):
    bclf=LogisticRegression(max_iter=3000,class_weight='balanced').fit(B._stack_l2(cat,tr),[1 if L[s]==5 else 0 for s in tr])
    tr5=[s for s in tr if L[s]<5]; sclf=LogisticRegression(max_iter=3000,class_weight='balanced').fit(B._stack_l2(cat,tr5),[L[s] for s in tr5])
    for s in te:
        x=B._stack_l2(cat,[s]); preds[s]=5 if bclf.predict(x)[0]==1 else int(sclf.predict(x)[0])
score('Q_concat_twostage', preds, 6, is_preds=True)
print("DONE_QWEN_CONCAT",flush=True)
