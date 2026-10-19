"""Frozen full-test streams, same downstream pipeline and public FastPLAID."""
from pathlib import Path
import sys,json,time,gc,resource,hashlib,os
R=Path(__file__).resolve().parent
assert (R/'evidence-v5-algorithm-freeze.json').exists()
kind=sys.argv[sys.argv.index('--kind')+1];j=sys.argv.index('--kind');del sys.argv[j:j+2]
assert kind in ['conditional','normal32','power32','additive','lik','fastplaid']
if kind=='fastplaid':
 base=R/'strong_bench.py';src=base.read_text().split('for selected_config in')[0]
else:
 base=R/'tiered_bench.py';src=base.read_text().split('results=[]')[0]
 src=src.replace('rawpin=torch.empty((nt,D),dtype=torch.float16,pin_memory=True);np.copyto(rawpin.numpy(),raw)','rawpin=copy_pinned_source(raw)')
 src=src.replace('import numpy as np,torch,triton','import numpy as np,torch,triton\nfrom pinned_source import copy_pinned_source')
src=src.replace('[32:160]','[32:]').replace('2**ideal-1','np.maximum(ideal,0)').replace('2**g-1','np.maximum(g,0)')
loadstart=time.time();exec(compile(src,str(base),'exec'));loadseconds=time.time()-loadstart
lastinfo={}
if kind!='fastplaid':
 from frontier_certified_guard import GuardedDenseRankedFrontier
 from frontier_densefinish_engine import DenseNormalFrontier
 from frontier_v5_bounds import MatchedBoundFrontier
 from frontier_v5_power32 import Power32Frontier
 kw=dict(programs=8192,block=256,compact_final=False,unsorted_seed=True)
 if kind=='conditional':engine=GuardedDenseRankedFrontier(cent,lookup_codes,lookup_off,io,ivf,maxbudget,512,2,8,groups=16,score_power=2.,**kw)
 elif kind=='power32':engine=Power32Frontier(cent,lookup_codes,lookup_off,io,ivf,maxbudget,512,2,8,**kw)
 elif kind=='normal32':engine=DenseNormalFrontier(cent,lookup_codes,lookup_off,io,ivf,maxbudget,512,2,8,**kw)
 elif kind=='additive':engine=MatchedBoundFrontier(cent,lookup_codes,lookup_off,io,ivf,maxbudget,512,2,8,groups=16,score_power=2.,bound='additive',**kw)
 else:
  from late_interaction_kernels.plaid import plaid_approx_score
  from evidence_v5_pad import pad_codes
  lens=(lookup_off[1:]-lookup_off[:-1]).long();L=int(lens.max());padded=torch.empty((n,L),device='cuda',dtype=torch.int32)
  pad_codes[(n,)](lookup_codes,lookup_off,padded,N=n,L=L,B=128)
  def engine(q,probe,budget):
   sc=plaid_approx_score((cent@q.T).contiguous(),padded,lens);v,j=sc.topk(budget);return j,torch.isfinite(v)
 candidates=engine;config=a.configs;_,budget=map(int,config.split('_'))
 def call(x):
  global lastinfo
  ids,scores,info=run(x,16,budget,'uvawinner');lastinfo=info;return ids,scores
else:call=run;config=a.config
out=R/'evidence-v5-stream'/a.dataset;out.mkdir(parents=True,exist_ok=True);dest=out/f'{kind}-{config}-cap{a.cap_gib:g}.json'
if dest.exists() and json.loads(dest.read_text()).get('complete'):sys.exit(0)
torch.cuda.synchronize();loadseconds=time.time()-loadstart;warmstart=time.time()
for x in dev[:8]:call(x)
torch.cuda.synchronize();warmseconds=time.time()-warmstart
gc.disable();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();resident=torch.cuda.memory_allocated();rows={};rng=np.random.default_rng(2026101923);passseconds=[]
for rep in range(a.reps):
 passstart=time.perf_counter()
 for qi in rng.permutation(len(qq)):
  torch.cuda.synchronize();st=time.perf_counter();ids,scores=call(qq[qi]);torch.cuda.synchronize();ms=(time.perf_counter()-st)*1e3
  assert len(ids)==len(set(ids))==100
  row=rows.setdefault(int(qi),{'query_id':qids[qi],'ids':ids,'scores':scores,'times_ms':[],'ndcg10':ndcg(ids,qids[qi],10),'ndcg100':ndcg(ids,qids[qi],100),**lastinfo})
  assert row['ids']==ids,('nondeterministic rank IDs',kind,qi,rep)
  row['times_ms'].append(ms)
 passseconds.append(time.perf_counter()-passstart)
 for row in rows.values():row['ms']=float(np.median(row['times_ms']))
 d={'started_unix':loadstart,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(loadstart)),'last_written_unix':time.time(),'dataset':a.dataset,'kind':kind,'config':config,'cap_gib':a.cap_gib,'gpu':torch.cuda.get_device_name(),'complete':rep+1==a.reps,'completed':len(rows),'completed_passes':rep+1,'queries':len(qq),'reps':a.reps,'scope':'Frozen independent full-test; whole-stream shuffled3passes, external FiQA8warmups, no samequerywarm. CPU input embedding to CPU100ids/scores; query encoding/index build excluded and reported separately. All same-pipeline controls include identical local6 refinement. GIGP timing handled separately.','load_seconds':loadseconds,'external_warmup_and_jit_seconds':warmseconds,'cpu_affinity':sorted(os.sched_getaffinity(0)),'resident_allocated_bytes':resident,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'host_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'pass_seconds_including_python_bookkeeping':passseconds,'records':list(rows.values())}
 dest.write_text(json.dumps(d));print('STREAM PASS',kind,config,rep+1,len(rows),float(np.mean([r['ms']for r in rows.values()])),flush=True)
print('V5 STREAM COMPLETE',flush=True)
