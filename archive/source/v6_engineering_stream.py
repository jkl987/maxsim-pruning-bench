"""Paired full existing refinement pipeline: ordinary4-bin vs8-bin independent state."""
from pathlib import Path
import sys,time,json,hashlib,gc,resource
R=Path(__file__).resolve().parent
ds_argument=sys.argv[sys.argv.index('--dataset')+1]
if '--dedup' in sys.argv and not(R/'tiered-summary'/ds_argument/'codes.npy').exists():sys.argv.remove('--dedup')
src=(R/'tiered_bench.py').read_text().split('results=[]')[0]
src=src.replace('[32:160]','[32:]').replace('2**ideal-1','np.maximum(ideal,0)').replace('2**g-1','np.maximum(g,0)')
src=src.replace('rawpin=torch.empty((nt,D),dtype=torch.float16,pin_memory=True);np.copyto(rawpin.numpy(),raw)','rawpin=copy_pinned_source(raw)').replace('import numpy as np,torch,triton','import numpy as np,torch,triton\nfrom pinned_source import copy_pinned_source')
exec(compile(src,str(R/'tiered_bench.py'),'exec'))
from frontier_v5_power32 import Power32Frontier
engines={f'ind{L}':Power32Frontier(cent,lookup_codes,lookup_off,io,ivf,maxbudget,512,2,L,programs=8192,block=256,compact_final=False,unsorted_seed=True)for L in [4,8]}
order=sorted(range(len(qids)),key=lambda i:hashlib.sha256(('v6-diagnostic-'+qids[i]).encode()).hexdigest())[:64];budget=4096;rows={};rng=np.random.default_rng(2026102005)
def call(name,x):
 global candidates
 candidates=engines[name];return run(x,16,budget,'uvawinner')
for name in engines:
 for x in dev[:8]:call(name,x)
torch.cuda.synchronize();gc.disable();torch.cuda.reset_peak_memory_stats()
for rep in range(5):
 for qi in rng.permutation(order):
  rr=rows.setdefault(qids[qi],{'query_id':qids[qi],'methods':{}})
  for name in rng.permutation(list(engines)):
   torch.cuda.synchronize();t=time.perf_counter();ids,scores,info=call(name,qq[qi]);torch.cuda.synchronize();ms=(time.perf_counter()-t)*1000
   z=rr['methods'].setdefault(name,{'ids':ids,'scores':scores,'times_ms':[],'ndcg10':ndcg(ids,qids[qi],10),'repeat_rank_changes':0,**info});z['times_ms'].append(ms);z['repeat_rank_changes']+=int(ids!=z['ids'])
 print('FULL PIPELINE',a.dataset,rep+1,flush=True)
out={'complete':True,'dataset':a.dataset,'queries':64,'repetitions':5,'budget':budget,'scope':'Observedfixedsample, pairedwholepasses; CPUqueryembeddingtoCPU100idsandscores, includingunchangedlocal6/UVArawrefinement. Oneprocesssession, no newpublicsystemcomparison; standaloneengineeringevidence, notalgorithmnovelty.','dedup_used':a.dedup,'coarse_code_count':len(lookup_codes),'ranking_changes':[r['query_id']for r in rows.values()if r['methods']['ind4']['ids']!=r['methods']['ind8']['ids']],'gpu_peak_allocated_bytes_with_both_engines':torch.cuda.max_memory_allocated(),'host_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'records':list(rows.values())}
(R/'v6'/f'{a.dataset}-engineering-full.json').write_text(json.dumps(out));print('ENGINEERING COMPLETE',a.dataset,len(out['ranking_changes']),flush=True)
