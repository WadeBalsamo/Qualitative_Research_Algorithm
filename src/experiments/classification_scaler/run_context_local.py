"""Local MiniLM context test — endpoint-free. Isolated vs target+context (concat & combined)."""
import sys, os, warnings; warnings.filterwarnings('ignore')
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,'src'); sys.path.insert(0,'.')
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from gnn_layer.config import GnnLayerConfig
from experiments.gnn_reliability import harness as H, baselines as B
from sklearn.linear_model import LogisticRegression
ABS='data/Meta'
df=H.load_corpus(ABS); folds=H.build_folds(df,seed=42,verbose=False)
raw=pd.read_csv(os.path.join(ABS,'02_meta/training_data/master_segments.csv'))
raw['segment_id']=raw['segment_id'].astype(str); raw['start_time_ms']=raw['start_time_ms'].fillna(0).astype(int)
bysess={s:g.sort_values('start_time_ms') for s,g in raw.groupby('session_id')}
def ctx_of(r,k=6):
    g=bysess[r.session_id]; pri=g[g.start_time_ms<r.start_time_ms].tail(k)
    return "\n".join(f"[{x.speaker}]: {str(x.text)[:1000]}" for x in pri.itertuples())
part=raw[raw.speaker=='participant']
tgt_txt={r.segment_id:str(r.text)[:2000] for r in part.itertuples()}
ctx_txt={r.segment_id:(ctx_of(r) or "[no prior context]")[:4000] for r in part.itertuples()}
cmb_txt={s:(ctx_txt[s]+"\n>>> TARGET: "+tgt_txt[s])[:5000] for s in tgt_txt}
ids=list(tgt_txt)
print(f"embedding {len(ids)} segs ×3 (target / context / combined) with all-MiniLM-L6-v2 ...",flush=True)
m=SentenceTransformer('all-MiniLM-L6-v2')
iso=m.encode([tgt_txt[i] for i in ids],normalize_embeddings=True,batch_size=64,show_progress_bar=False)
ctx=m.encode([ctx_txt[i] for i in ids],normalize_embeddings=True,batch_size=64,show_progress_bar=False)
cmb=m.encode([cmb_txt[i] for i in ids],normalize_embeddings=True,batch_size=64,show_progress_bar=False)
iso_emb={i:iso[k] for k,i in enumerate(ids)}
cat_emb={i:np.concatenate([iso[k],ctx[k]]) for k,i in enumerate(ids)}
cmb_emb={i:cmb[k] for k,i in enumerate(ids)}
def score(name,emb_or_preds,ncls,is_preds=False):
    preds = emb_or_preds if is_preds else B.run_linear_probe(df,emb_or_preds,folds,GnnLayerConfig(vaamr_n_classes=ncls,vaamr_class_balance=True))
    res=H.score_arm(name,preds,df,ABS,ncls,meta={'embedding':'minilm','method':name,'seed':42},write_ledger=False)
    l,h=res['llm_axis'],res['human_axis']
    print(f"  {name:20s}: LLM κ={l['cohen_kappa_205']:.3f} {[round(x,3) for x in l['ci95']]}  HUM κ={h['cohen_kappa']:.3f} {[round(x,3) for x in h['ci95']]} (n={h['n']})",flush=True)
print("=== LOCAL MiniLM context test (relative: same embedder, isolated vs +context) ===",flush=True)
score('M_isolated', iso_emb, 6)
score('M_concat_ctx', cat_emb, 6)
score('M_combined', cmb_emb, 6)
seg_ids,lab6,_=B._prepare_labeled(df,cat_emb,GnnLayerConfig(vaamr_n_classes=6)); L=dict(zip(seg_ids,lab6)); fo,fl=B._resolve_folds(seg_ids,df,folds)
preds={}
for f,tr,te in B._iter_folds(seg_ids,fo,fl):
    bclf=LogisticRegression(max_iter=3000,class_weight='balanced').fit(B._stack_l2(cat_emb,tr),[1 if L[s]==5 else 0 for s in tr])
    tr5=[s for s in tr if L[s]<5]; sclf=LogisticRegression(max_iter=3000,class_weight='balanced').fit(B._stack_l2(cat_emb,tr5),[L[s] for s in tr5])
    for s in te:
        x=B._stack_l2(cat_emb,[s]); preds[s]=5 if bclf.predict(x)[0]==1 else int(sclf.predict(x)[0])
score('M_concat_twostage', preds, 6, is_preds=True)
print("DONE_LOCAL_CONTEXT",flush=True)
