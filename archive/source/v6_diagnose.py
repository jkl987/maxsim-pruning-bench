from pathlib import Path
import sys,json,time,hashlib,argparse,gc,os
R=Path(__file__).resolve().parent;sys.path[:0]=[str(R/'strong-lib'),str(R/'lib')]
import numpy as np,torch
from v6_engine import DiagnosticFrontier,exact_table
from gpu_candidate_kernels import centroid_lookup
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--queries',type=int,default=64);p.add_argument('--reps',type=int,default=5);p.add_argument('--prefix',type=int,default=512);p.add_argument('--pilot',action='store_true');a=p.parse_args()
torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
ds=a.dataset;sub=ds.startswith('cross-');s=R/'v6'/ds if sub else R/'vectors'/ds;m=json.loads((s/'meta.json').read_text())
if sub:qq=np.load(s/'queries.npy');qids=m['query_ids'];ix=s/'index';budget=128
elif ds=='nq-full':
 qq=np.concatenate([np.load(s/'queries-final.npy'),np.load(s/'queries-audit.npy')]);qids=json.loads((s/'meta-final.json').read_text())['query_ids']+json.loads((s/'meta-audit.json').read_text())['query_ids'];ix=R/'plaid-index'/ds;budget=4096
else:qq=np.load(s/'queries.npy')[32:];qids=m['query_ids'][32:];ix=R/'plaid-index'/ds;budget=4096
order=sorted(range(len(qids)),key=lambda i:hashlib.sha256(('v6-diagnostic-'+qids[i]).encode()).hexdigest())[:a.queries]
dest=R/'v6'/f'{ds}-P{a.prefix}{"-pilot"if a.pilot else ""}.json'
if dest.exists() and json.loads(dest.read_text()).get('complete') and not json.loads(dest.read_text()).get('failures'):print('REUSE',dest);sys.exit()
cent=torch.from_numpy(np.load(ix/'centroids.npy')).cuda();ivf=torch.from_numpy(np.load(ix/'ivf.npy').astype('int32')).cuda();io=torch.tensor(np.r_[0,np.cumsum(np.load(ix/'ivf_lengths.npy'))],device='cuda',dtype=torch.int64)
sx=R/'tiered-summary'/ds;offpath=s/'off.npy'if sub else sx/'off.npy' if(sx/'off.npy').exists()else s/'off.npy';codepath=ix/'merged_codes.npy'if sub else sx/'codes.npy'if(sx/'codes.npy').exists()else ix/'merged_codes.npy'
off=torch.from_numpy(np.load(offpath).astype('int32')).cuda();codes=torch.from_numpy(np.load(codepath).astype('int32')).cuda();n=len(off)-1
spec=[('ind2','independent',32,2),('ind4','independent',32,4),('ind8','independent',32,8),('cond4','conditional',16,4),('cond8','conditional',16,8),('cond16','conditional',16,16),('delta8','additive',16,8)]
engines={name:DiagnosticFrontier(cent,codes,off,io,ivf,budget,a.prefix,2,L,groups=G,score_power=2.,kind=kind,programs=8192,block=256,compact_final=False,unsorted_seed=True)for name,kind,G,L in spec}
ref=torch.empty(n,device='cuda');selected=torch.empty(n,dtype=torch.bool,device='cuda');ids=torch.arange(n,device='cuda',dtype=torch.int32);count=torch.tensor([n],device='cuda',dtype=torch.int32)
times={i:{k:[]for k in engines}for i in order};rng=np.random.default_rng(2026102001)
def run(e,x,**kw):
 q=torch.from_numpy(x).cuda();j,_=e(q,16,budget,**kw);return j.cpu().numpy()
# JIT and all engine code paths warmed before formal timing; diagnostic data are observed.
for e in engines.values():run(e,qq[order[0]])
torch.cuda.synchronize();gc.disable()
for rep in range(a.reps):
 for qi in rng.permutation(order):
  for name in rng.permutation(list(engines)):
   torch.cuda.synchronize();t=time.perf_counter();run(engines[name],qq[qi]);torch.cuda.synchronize();times[qi][name].append((time.perf_counter()-t)*1000)
 print('TIMED',ds,rep+1,flush=True)
rows=[];failures=[]
def inspect(e,returned,truthcut):
 selected.zero_();selected[torch.from_numpy(returned).cuda()]=True
 strictmissing=int(((ref>truthcut)&~selected).sum());viol=int((e.upper<ref).sum());ties=int((ref==truthcut).sum())
 return {'upper_violations':viol,'strict_missing':strictmissing,'true_cut':float(truthcut),'seed_cut':float(e.cut),'true_cut_ties':ties,'slack_mean':float((e.upper-ref).mean()),'slack_p50_p90':torch.quantile(e.upper-ref,torch.tensor([.5,.9],device='cuda')).tolist(),'false_survivors_truecut':int(((e.upper>=truthcut)&(ref<truthcut)).sum()),'false_survivors_seedcut':int(((e.upper>=e.cut)&(ref<e.cut)).sum()),'scored_documents':int(e.survivors.sum()),'scored_tokens':int(((off[1:]-off[:-1]).long()*e.survivors).sum()),'state_bytes_per_doc':e.dbits.numel()*8/n,'owned_workspace_bytes':e.owned_bytes(),'logical_postings':int(((io[1:]-io[:-1])*(e.cf!=0)).sum())}
for ii,qi in enumerate(order):
 q=torch.from_numpy(qq[qi]).cuda();lut=cent@q.T;assert q.shape[0]==32 and torch.isfinite(lut).all() and float(lut.abs().max())<=2
 centroid_lookup[(8192,)](lut,codes,off,ids,count,ref,B=256,num_warps=4,num_stages=1);truthcut=ref.topk(budget).values[-1]
 row={'query_id':qids[qi],'times_ms':times[qi],'methods':{}}
 for name,e in engines.items():
  torch.cuda.synchronize();resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();ret=run(e,qq[qi]);torch.cuda.synchronize();extra=max(0,torch.cuda.max_memory_allocated()-resident)
  z=inspect(e,ret,truthcut);z['transient_peak_extra_bytes']=extra;z['workspace_plus_transient_bytes']=z['owned_workspace_bytes']+extra
  if ii<8:
   run(e,qq[qi],instrument=True);torch.cuda.synchronize();z['instrumented_gpu_ms']=e.phase_ms()
  row['methods'][name]=z
  if z['upper_violations']or z['strict_missing']:failures.append({'query_id':qids[qi],'method':name,**z})
 # Complete finite-codebook envelope for exactly the same16 anchors/groups/bin states.
 e=engines['cond8'];run(e,qq[qi]);approx=e.table.clone();anc=e.anchors.clone();groups=e.assign.clone();thresholds=e.thresholds.clone();dbits=e.dbits.clone()
 torch.cuda.synchronize();st=time.perf_counter();ideal=exact_table(lut,anc,groups,thresholds,e.levels);torch.cuda.synchronize();buildms=(time.perf_counter()-st)*1000
 assert bool(torch.isfinite(ideal).all()),'Empty feasible sets need explicit impossible-state handling'
 # Independent CPU float64 check of every used table entry for first2 diagnostic queries.
 err=None
 if ii<2:
  ll=lut.cpu().numpy().astype('float64');aa=anc.cpu().numpy();gg=groups.cpu().numpy();tt=thresholds.cpu().numpy();expected=np.zeros_like(ideal.cpu().numpy(),dtype='float64')
  for g,ar in enumerate(aa):
   for b in range(e.levels+1):
    cap=ll[:,ar].max()if b==0 else tt[b-1,g];mask=ll[:,ar]<=cap if b==0 else ll[:,ar]<cap
    for term in np.where(gg==g)[0]:expected[g,b]+=cap if term==ar else ll[mask,term].max()
  err=float(np.max(np.abs(expected-ideal.cpu().numpy())));assert err<3e-5
 tt=[]
 for _ in range(3):
  torch.cuda.synchronize();st=time.perf_counter();ret=run(e,qq[qi],override=ideal);torch.cuda.synchronize();tt.append((time.perf_counter()-st)*1000)
 assert torch.equal(anc,e.anchors)and torch.equal(groups,e.assign)and torch.equal(dbits,e.dbits)
 z=inspect(e,ret,truthcut);z.update(offline_envelope_build_ms=buildms,cached_oracle_times_ms=tt,estimated_build_plus_cached_ms=buildms+float(np.median(tt)),cpu_f64_table_max_error=err,table_approx_minus_ideal_mean=float((approx-ideal).mean()),table_approx_minus_ideal_max=float((approx-ideal).max()))
 row['methods']['ideal8']=z
 if z['upper_violations']or z['strict_missing']:failures.append({'query_id':qids[qi],'method':'ideal8',**z})
 rows.append(row)
 out={'complete':len(rows)==len(order),'dataset':ds,'prefix':a.prefix,'budget':budget,'reps':a.reps,'documents':n,'centroids':len(cent),'tokens':len(codes),'dimension':cent.shape[1],'scope':'Observed-data coarseTopB diagnosis. CPU embedding to CPU B IDs; no token refinement. Whole shuffled paired passes after explicit warmup. Oracle is offline and its build cost excluded only in explicitly cached_oracle times. Instrumented GPU events separate from formal wallclock.','threshold_rules':'Common exact coarseTopB threshold: false survivor upper>=cut and exactscore<cut. Separate seed-derived operational threshold included. Exact ties at cut may choose arbitrary IDs; strict-above-cut must never be omitted. Guard4e-4 for all new controls.','shared_index_bytes':sum(x.numel()*x.element_size()for x in [cent,codes,off,io,ivf]),'gpu':torch.cuda.get_device_name(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'records':rows,'failures':failures}
 dest.write_text(json.dumps(out));print('AUDIT',ds,len(rows),len(order),'failures',len(failures),flush=True)
assert not failures,failures[:1]
