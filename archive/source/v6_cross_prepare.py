"""64k matched-document factorial; only missing encoder/corpus cells are encoded."""
from pathlib import Path
import sys,json,time,hashlib,gc,os
R=Path(__file__).resolve().parent;sys.path[:0]=[str(R/'evidence-v4-faiss-sm120-build/faiss/python'),str(R/'evidence-v4-faiss-lib'),str(R/'strong-lib'),str(R/'lib')]
import numpy as np,torch,pyarrow.parquet as pq,faiss
torch.set_num_threads(8);faiss.omp_set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
V=R/'v6';N=64000;C=8192
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
for corpus,original in [('nq','nq-full'),('hotpot','hotpotqa-colbertv2')]:
 s=R/'vectors'/original;m=json.loads((s/'meta.json').read_text());off=np.load(s/'off.npy');dim=m['dimension'];raw=np.memmap(s/'raw.f16',mode='r',dtype='float16',shape=(m['tokens'],dim))
 pick=np.sort(np.random.default_rng(2026102002).choice(len(off)-1,N,replace=False));docids=[m['doc_ids'][i]for i in pick];wanted=set(docids)
 if corpus=='nq':
  qq=np.concatenate([np.load(s/'queries-final.npy'),np.load(s/'queries-audit.npy')]);qids=json.loads((s/'meta-final.json').read_text())['query_ids']+json.loads((s/'meta-audit.json').read_text())['query_ids']
 else:qq=np.load(s/'queries.npy')[32:];qids=m['query_ids'][32:]
 qp=sorted(range(len(qids)),key=lambda i:hashlib.sha256(('v6-diagnostic-'+qids[i]).encode()).hexdigest())[:64];qi=[qids[i]for i in qp]
 manifest=V/f'cross-{corpus}-selection.json';manifest.write_text(json.dumps({'doc_ids':docids,'source_row_indices':pick.tolist(),'query_ids':qi,'seed':2026102002,'documents':N,'scope':'Uniformdocument subset diagnostic; no corpusrelevance guarantees; matchedidentities acrossencoders.'}))
 if corpus=='nq':
  texts={}
  for batch in pq.ParquetFile(R/'data/nq-corpus.parquet').iter_batches(batch_size=8192):
   for d in batch.to_pylist():
    if str(d['_id'])in wanted:texts[str(d['_id'])]=d.get('title','')+' '+d['text']
  queries={str(d['_id']):d['text']for batch in pq.ParquetFile(R/'data/nq-queries.parquet').iter_batches()for d in batch.to_pylist()}
 else:
  texts={}
  for line in (R/'data/hotpotqa/corpus.jsonl').open():
   d=json.loads(line)
   if str(d['_id'])in wanted:texts[str(d['_id'])]=(d.get('title','')+' '+d['text']).strip()
  queries={str(d['_id']):d['text']for d in map(json.loads,(R/'data/hotpotqa/queries.jsonl').read_text().splitlines())}
 assert len(texts)==N and all(q in queries for q in qi)
 for model in ['small','colbert']:
  dest=V/f'cross-{corpus}-{model}';dest.mkdir(exist_ok=True);cached=(corpus=='nq'and model=='small')or(corpus=='hotpot'and model=='colbert');st=time.time();lens=[]
  if not(dest/'READY.json').exists():
   if model=='small':from maxsim_encode import Encoder
   else:from maxsim_colbertv2_encode import Encoder
   enc=Encoder()
   # New query embedding compared to cached64 queries for reused diagonal.
   qnew=np.stack([v.numpy()for v in enc.encode([queries[k]for k in qi],True)]).astype('float16')
   errors={}
   if cached:
    errors['query_max_abs']=float(np.max(np.abs(qnew.astype('float32')-qq[qp].astype('float32'))));assert errors['query_max_abs']<.002
    for j,v in enumerate(enc.encode([texts[k]for k in docids[:4]])):
     old=raw[off[pick[j]]:off[pick[j]+1]];assert old.shape==v.shape
     errors[f'doc{j}_max_abs']=float(np.max(np.abs(old.astype('float32')-v.numpy())));assert errors[f'doc{j}_max_abs']<.002
    with (dest/'raw.f16').open('wb')as f:
     for i in pick:
      x=raw[off[i]:off[i+1]];x.tofile(f);lens.append(len(x))
    qsave=qq[qp]
   else:
    with (dest/'raw.f16').open('wb')as f:
     for start in range(0,N,1024):
      vv=enc.encode([texts[k]for k in docids[start:start+1024]]);np.concatenate([x.numpy()for x in vv]).astype('float16').tofile(f);lens.extend(len(x)for x in vv)
      print('ENCODE',corpus,model,start+len(vv),flush=True)
    qsave=qnew
   np.save(dest/'queries.npy',qsave);np.save(dest/'off.npy',np.r_[0,np.cumsum(lens)].astype('int64'))
   metadata={'documents':N,'tokens':sum(lens),'dimension':qsave.shape[-1],'doc_ids':docids,'query_ids':qi,'model':model,'corpus':corpus,'reused_full_vectors':cached,'encoding_seconds':time.time()-st,'cache_verification':errors,'encoding_protocol':{k:enc.meta.get(k)for k in ['query_maxlen','doc_maxlen','attend_to_mask_tokens','dim','mask_punctuation']},'scope':'Matched corpus subset; main effect is encoder plus its protocol, not architecture-only causal effect.'}
   (dest/'meta.json').write_text(json.dumps(metadata));(dest/'READY.json').write_text(json.dumps({'complete':True,'raw_sha256':sha(dest/'raw.f16'),'queries_sha256':sha(dest/'queries.npy'),'seconds':time.time()-st}));del enc;gc.collect();torch.cuda.empty_cache()
  meta=json.loads((dest/'meta.json').read_text());ix=dest/'index';ix.mkdir(exist_ok=True)
  if(ix/'READY.json').exists():continue
  t=time.time();D=meta['dimension'];nt=meta['tokens'];x=np.memmap(dest/'raw.f16',mode='r',dtype='float16',shape=(nt,D));oo=np.load(dest/'off-raw.npy'if(dest/'off-raw.npy').exists()else dest/'off.npy');sample=x[np.sort(np.random.default_rng(2026102003).choice(nt,min(262144,nt),False))].astype('float32')
  km=faiss.Kmeans(D,C,niter=10,gpu=True,verbose=False,seed=20261020);km.train(sample);cent=torch.nn.functional.normalize(torch.from_numpy(km.centroids),dim=-1).cuda();np.save(ix/'centroids.npy',cent.cpu().half().numpy())
  cent=cent.half();codes=np.empty(nt,dtype='int32')
  for start in range(0,nt,8192):codes[start:start+8192]=(cent@torch.from_numpy(x[start:start+8192].copy()).cuda().T).argmax(0).cpu().numpy()
  # Deduplicate centroid IDs per document, matching existing coarse-summary semantics.
  cc=[np.unique(codes[oo[i]:oo[i+1]])for i in range(N)];lens=[len(z)for z in cc];codes=np.concatenate(cc);np.save(ix/'merged_codes.npy',codes);np.save(dest/'off-raw.npy',oo);np.save(dest/'off.npy',np.r_[0,np.cumsum(lens)].astype('int64'))
  pairs=np.concatenate([z.astype('int64')*N+i for i,z in enumerate(cc)]);pairs.sort();np.save(ix/'ivf.npy',(pairs%N).astype('int32'));np.save(ix/'ivf_lengths.npy',np.bincount(pairs//N,minlength=C).astype('int64'))
  (ix/'READY.json').write_text(json.dumps({'complete':True,'seconds':time.time()-t,'centroids':C,'sample_tokens':len(sample),'kmeans_iterations':10,'training_seed':20261020,'assignment':'storedFP16centroids @storedFP16tokens, torchTF32disabled','code_tokens':len(codes),'scope':'All4factorialcells samebuilder. Differsfromfullcodebooktraining; do notcompareabsolute fullvs subsetlatency.'}));print('CROSS INDEX READY',corpus,model,flush=True)
  del cent,km,sample,codes,cc,pairs;gc.collect();torch.cuda.empty_cache()
 del raw,texts,queries;gc.collect()
print('CROSS_PREPARATION_COMPLETE',flush=True)
