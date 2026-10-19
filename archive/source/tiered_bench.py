"""End-to-end two-tier retrieval; source FP16 vectors stay in host RAM.
Modes: quant-only; fixed full-document rerank; interval-selected full-document
rerank; interval-selected possible-winner-token rerank. Same PLAID candidates.
"""
import os,sys,json,time,gc,argparse,resource
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path[:0]=[str(R/'strong-lib'),str(R/'lib')]
import numpy as np,torch,triton
from gpu_candidate_kernels import mark_ivf,compact_flags,centroid_lookup
from tiered_kernels import quant_bounds,possible_tokens,pack_token_ids,uva_refine,gather_host_tokens
from kernels import exact
p=argparse.ArgumentParser();p.add_argument('--dataset',default='nq-full');p.add_argument('--split',default='dev');p.add_argument('--limit',type=int,default=32);p.add_argument('--reps',type=int,default=3);p.add_argument('--configs',default='2_1024,8_16384,16_65536');p.add_argument('--modes',default='quant,fixed256,document,winner');p.add_argument('--cap-gib',type=float,default=32);p.add_argument('--validate',type=int,default=0);p.add_argument('--centered',action='store_true');p.add_argument('--tag',default='unbound');p.add_argument('--local6',action='store_true');p.add_argument('--dedup',action='store_true');p.add_argument('--disable-gc',action='store_true');a=p.parse_args()
if a.local6 and not a.centered:p.error("--local6 requires --centered: residual reconstruction must include centroid and projected bias")
torch.set_num_threads(8);torch.manual_seed(42)
torch.cuda.set_per_process_memory_fraction(a.cap_gib*2**30/torch.cuda.get_device_properties(0).total_memory)
s=R/'vectors'/a.dataset;m=json.loads((s/'meta.json').read_text());D=m['dimension'];nt=m['tokens'];off=np.load(s/'off.npy');n=len(off)-1;maxl=int(np.diff(off).max());dev=np.load(s/'queries.npy')[:32]
if a.split=='dev':qq=dev;qids=m['query_ids'][:32];rels=m['qrels']
elif a.dataset=='nq-full':
    qq=np.concatenate([np.load(s/'queries-final.npy'),np.load(s/'queries-audit.npy')]);f=json.loads((s/'meta-final.json').read_text());aa=json.loads((s/'meta-audit.json').read_text());qids=f['query_ids']+aa['query_ids'];rels={**f['qrels'],**aa['qrels']}
else:qq=np.load(s/'queries.npy')[32:160];qids=m['query_ids'][32:160];rels=m['qrels']
if a.limit:qq=qq[:a.limit];qids=qids[:a.limit]
raw=np.memmap(s/'raw.f16',mode='r',dtype='float16',shape=(nt,D));ix=R/'plaid-index'/a.dataset;cx=R/('tiered-local6-index' if a.local6 else ('tiered-centered-index' if a.centered else 'tiered-index'))/a.dataset
codes=torch.from_numpy(np.memmap(cx/('codes.u6' if a.local6 else 'codes.i8'),mode='r',dtype='uint8' if a.local6 else 'int8',shape=(nt,D*3//4 if a.local6 else D))).cuda()
scale=torch.from_numpy(np.memmap(cx/'scale.f16',mode='r',dtype='float16',shape=(nt,))).cuda()
err=torch.from_numpy(np.memmap(cx/'error.f16',mode='r',dtype='float16',shape=(nt,))).cuda()
center=torch.from_numpy(np.load(cx/'center.npy')).cuda() if a.centered else torch.zeros(D,device='cuda');axis=torch.from_numpy(np.load(cx/'axis.npy')).cuda() if a.centered else center.float();bias=torch.from_numpy(np.memmap(cx/'bias.f32',mode='r',dtype='float32',shape=(nt,))).cuda() if a.centered else torch.zeros(1,device='cuda')
o=torch.from_numpy(off.astype('int32')).cuda();cent=torch.from_numpy(np.load(ix/'centroids.npy')).cuda();cids=torch.from_numpy(np.load(ix/'merged_codes.npy').astype('int32')).cuda();ivf=torch.from_numpy(np.load(ix/'ivf.npy').astype('int32')).cuda();il=np.load(ix/'ivf_lengths.npy');io=torch.tensor(np.r_[0,np.cumsum(il)],device='cuda',dtype=torch.int64)
if a.local6:
    from tiered_local6_kernels import quant_bounds,possible_tokens,uva_refine
    center=cent.half()
codec_kwargs={'CodeIDs':cids} if a.local6 else {}
lookup_codes=cids;lookup_off=o
if a.dedup:
    sx=R/'tiered-summary'/a.dataset;lookup_codes=torch.from_numpy(np.load(sx/'codes.npy')).cuda();lookup_off=torch.from_numpy(np.load(sx/'off.npy')).cuda()
    if not a.local6:cids=lookup_codes
flags=torch.empty(n,device='cuda',dtype=torch.int32);pids=torch.empty_like(flags);pc=torch.empty(1,device='cuda',dtype=torch.int32);cs=torch.empty(n,device='cuda');dummy=torch.empty(1,device='cuda')
maxbudget=max(int(c.split('_')[1]) for c in a.configs.split(','));coarse=torch.empty(maxbudget,device='cuda');lower=torch.empty_like(coarse);upper=torch.empty_like(coarse);rowlower=torch.empty((maxbudget,32),device='cuda')
tokenids=torch.empty((maxbudget,maxl),device='cuda',dtype=torch.int32);counts=torch.empty(maxbudget,device='cuda',dtype=torch.int32)
# Fixed staging buffers are counted in peak/resident memory. Overflow is explicit.
capacity=2**23;packed=torch.empty(capacity,device='cuda',dtype=torch.int32)
host=torch.empty((capacity,D),dtype=torch.float16,pin_memory=True);hostnp=host.numpy();stage=torch.empty_like(host,device='cuda');refscore=torch.empty(maxbudget,device='cuda')
class HostPointer:
    def __init__(self,x):self.x=x;self.dtype=x.dtype
    def data_ptr(self):return self.x.data_ptr()
rawpin=None
if 'uva' in a.modes:
    tpin=time.time();rawpin=torch.empty((nt,D),dtype=torch.float16,pin_memory=True);np.copyto(rawpin.numpy(),raw);rawptr=HostPointer(rawpin);print('PINNED HOST READY seconds',time.time()-tpin,flush=True)
doctokens=torch.empty(maxbudget,device='cuda',dtype=torch.int32)
out=R/('tiered-local6-results' if a.local6 else ('tiered-centered-results' if a.centered else 'tiered-results'))/a.dataset;out.mkdir(parents=True,exist_ok=True)
def ndcg(ids,qid,k):
    rel=rels[qid];g=np.array([rel.get(m['doc_ids'][int(i)],0) for i in ids[:k]]);ideal=np.array(sorted(rel.values(),reverse=True)[:k]);den=((2**ideal-1)/np.log2(np.arange(len(ideal))+2)).sum();return float(((2**g-1)/np.log2(np.arange(len(g))+2)).sum()/den) if den else 0.
def candidates(q,probe,budget):
    lut=(cent@q.T).contiguous();cells=lut.topk(probe,dim=0).indices.T.contiguous().reshape(-1).int();flags.zero_();mark_ivf[(32*probe,8)](cells,io,ivf,flags,P=probe,B=256);pc.zero_();compact_flags[(triton.cdiv(n,256),)](flags,pids,pc,N=n,B=256);cs.fill_(-float('inf'));centroid_lookup[(512,)](lut,lookup_codes,lookup_off,pids,pc,cs,B=128,num_warps=4,num_stages=1);vv,ii=cs.topk(budget)
    ids=torch.where(torch.isfinite(vv),pids[ii],0).long()
    return ids,torch.isfinite(vv)
def run(qcpu,probe,budget,mode,debug=False):
    q=torch.from_numpy(qcpu).cuda();ids,valid=candidates(q,probe,budget);qp=q.float()@axis;qr=(q.float()-qp[:,None]*axis[None,:]).norm(dim=1)
    quant_bounds[(budget,)](q,codes,scale,err,o,ids,coarse,lower,upper,rowlower,center,bias,qp,qr,D=D,B=64,GUARD=2e-5,CENTERED=a.centered,num_warps=4,num_stages=1,enable_fp_fusion=False,**codec_kwargs)
    coarse[:budget].masked_fill_(~valid,-float('inf'));lower[:budget].masked_fill_(~valid,-float('inf'));upper[:budget].masked_fill_(~valid,-float('inf'))
    if mode=='quant':
        v,j=coarse[:budget].topk(100);return ids[j].cpu().tolist(),v.cpu().tolist(),{'refine_docs':0,'refine_tokens':0,'transfer_bytes':0}
    if mode in ['uvadocument','uvawinner']:
        cut=lower[:budget].topk(100).values[-1]
        uva_refine[(256,)](q,rawptr,codes,scale,err,o,ids,upper,cut,rowlower,center,bias,qp,qr,refscore,counts,doctokens,N=budget,D=D,B=64,GUARD=2e-5,CENTERED=a.centered,FULLDOC=mode=='uvadocument',num_warps=4,num_stages=1,enable_fp_fusion=False,**codec_kwargs)
        v,j=refscore[:budget].topk(100);rid=ids[j].cpu().tolist();scores=v.cpu().tolist()
        ns,num,docn=torch.stack([(counts[:budget]>0).sum(),counts[:budget].sum(),doctokens[:budget].sum()]).cpu().tolist()
        info={'refine_docs':ns,'refine_tokens':num,'transfer_bytes':None,'requested_raw_bytes':num*D*2,'selected_document_tokens':docn,'transport':'direct GPU loads from pinned host; requested bytes are logical, not measured PCIe bus traffic'}
        if debug:info.update(candidate_ids=ids.detach(),q=q,valid=valid)
        return rid,scores,info
    if mode.startswith('fixed'):sel=coarse[:budget].topk(min(int(mode[5:]),budget)).indices
    else:sel=torch.nonzero(upper[:budget]>=lower[:budget].topk(100).values[-1],as_tuple=False).flatten()
    ns=sel.numel()
    possible_tokens[(ns,)](q,codes,scale,err,o,ids,sel,rowlower,tokenids,counts,center,bias,qp,qr,D=D,B=64,MAXL=maxl,GUARD=2e-5,FULLDOC=mode not in ['winner','uvapack'],CENTERED=a.centered,num_warps=4,num_stages=1,enable_fp_fusion=False,**codec_kwargs)
    prefix=torch.cat([torch.zeros(1,device='cuda',dtype=torch.int32),torch.cumsum(counts[:ns],0,dtype=torch.int32)])
    num=int(prefix[-1]);assert num<=capacity,(num,capacity)
    pack_token_ids[(ns,)](tokenids,counts,prefix,packed,N=ns,MAXL=maxl,B=256)
    if mode=='uvapack':gather_host_tokens[(triton.cdiv(num,16),)](rawptr,packed,stage,N=num,D=D,T=16,num_warps=4)
    else:
        cpuids=packed[:num].cpu().numpy()
        np.take(raw,cpuids,axis=0,out=hostnp[:num],mode='clip')
        stage[:num].copy_(host[:num],non_blocking=True)
    exact[(ns,)](q,stage,prefix,dummy,dummy,dummy,refscore,D=D,BD=64,MODE=0,num_warps=4,num_stages=1)
    v,j=refscore[:ns].topk(100);rid=ids[sel[j]].cpu().tolist();scores=v.cpu().tolist()
    info={'refine_docs':ns,'refine_tokens':num,'transfer_bytes':num*D*2+num*4,'selected_document_tokens':int((o[ids[sel]+1]-o[ids[sel]]).sum())}
    if mode=='uvapack':info.update(transfer_bytes=None,requested_raw_bytes=num*D*2,transport='GPU gathers from pinned host; logical requested bytes')
    if debug:info.update(candidate_ids=ids.detach(),selected=sel.detach(),q=q,valid=valid)
    return rid,scores,info
results=[]
if a.disable_gc:gc.disable()
for config in a.configs.split(','):
    probe,budget=map(int,config.split('_'))
    for mode in a.modes.split(','):
        for q in dev[:4]:run(q,probe,budget,mode)
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();resident=torch.cuda.memory_allocated();rows=[]
        for qi in np.random.default_rng(20261019).permutation(len(qq)):
            run(qq[qi],probe,budget,mode);times=[]
            for _ in range(a.reps):
                torch.cuda.synchronize();t=time.perf_counter();ids,scores,info=run(qq[qi],probe,budget,mode);torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
            assert len(ids)==len(set(ids))==100
            rows.append({'query_id':qids[qi],'ids':ids,'scores':scores,'ms':float(np.median(times)),'times_ms':times,'ndcg10':ndcg(ids,qids[qi],10),'ndcg100':ndcg(ids,qids[qi],100),**info})
        data={'cpu_affinity':sorted(os.sched_getaffinity(0)),'tag':a.tag,'dataset':a.dataset,'split':a.split,'engine':'tiered','config':config,'mode':mode,'reps':a.reps,'cap_gib':a.cap_gib,'gpu':torch.cuda.get_device_name(),'completed':len(rows),'total':len(qq),'records':rows,'resident_allocated_bytes':resident,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'host_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'raw_host_bytes':raw.nbytes,'pinned_raw_host_bytes':raw.nbytes if rawpin is not None else 0,'staging_capacity_tokens':capacity,'gc_disabled':not gc.isenabled(),'local6':a.local6,'deduplicated_centroid_summary':a.dedup,'centered':a.centered,'bound_guard':2e-5,'certification_scope':'input PLAID candidates only; empirical floating-point bound validation required','timing':'per-query untimed warmup; CPU query through CPU top100 including host gather, both PCIe directions, staging, synchronization and diagnostic counters; no encoding/build/cold startup'}
        path=out/f'{config}-{mode}-{a.split}-r{a.reps}-cap{a.cap_gib:g}-{a.tag}.json';path.write_text(json.dumps(data));results.append(data)
        print('DONE',config,mode,'ms',np.mean([r['ms'] for r in rows]),'ndcg',np.mean([r['ndcg10'] for r in rows]),'docs',np.mean([r['refine_docs'] for r in rows]),'tokens',np.mean([r['refine_tokens'] for r in rows]),'GiB',data['peak_allocated_bytes']/2**30,flush=True)
if a.validate:
    # Validation allocation is intentionally after and outside measured results.
    torch.cuda.set_per_process_memory_fraction(.94);rawgpu=torch.from_numpy(raw).cuda();gold=torch.empty(maxbudget,device='cuda');checks=[]
    refpath=R/'tiered-reference'/f'{a.dataset}-{a.split}.json';refpath.parent.mkdir(exist_ok=True)
    refs=json.loads(refpath.read_text()) if refpath.exists() else {}
    from late_interaction_kernels import maxsim_varlen
    qo=torch.tensor([0,32],device='cuda',dtype=torch.int32);parts=[];st=0
    while st<n:
        en=min(n,max(st+1,int(np.searchsorted(off,off[st]+(1<<30)//D,side='right')-1)));po=torch.tensor((off[st:en+1]-off[st]).astype('int32'),device='cuda');parts.append((rawgpu[int(off[st]):int(off[en])],po,int(np.diff(off[st:en+1]).max())));st=en
    for qi in range(min(a.validate,len(qq))):
        if qids[qi] in refs:continue
        q=torch.from_numpy(qq[qi]).cuda();truth=torch.cat([maxsim_varlen(q,xx,qo,oo,max_seqlen_q=32,max_seqlen_d=ll).flatten() for xx,oo,ll in parts]);sv,si=truth.topk(100);ii=si.cpu().tolist();refs[qids[qi]]={'ids':ii,'scores':sv.cpu().tolist(),'ndcg10':ndcg(ii,qids[qi],10)}
    refpath.write_text(json.dumps(refs))
    for config in a.configs.split(','):
        probe,budget=map(int,config.split('_'))
        for qi in range(min(a.validate,len(qq))):
            rid,scores,info=run(qq[qi],probe,budget,'uvawinner' if rawpin is not None else 'winner',debug=True);q=info['q'];ids=info['candidate_ids']
            exact[(budget,)](q,rawgpu,o,ids,dummy,dummy,gold,D=D,BD=64,MODE=1,num_warps=4,num_stages=1)
            gold[:budget].masked_fill_(~info['valid'],-float('inf'));v,j=gold[:budget].topk(100);goldids=ids[j].cpu().tolist();v=v.cpu().numpy()
            lowviol=int(((lower[:budget]>gold[:budget]+1e-5)&info['valid']).sum());upviol=int(((upper[:budget]<gold[:budget]-1e-5)&info['valid']).sum())
            checks.append({'config':config,'query_id':qids[qi],'lower_violations':lowviol,'upper_violations':upviol,'top100_set_equal':set(rid)==set(goldids),'ordered_ids_equal':rid==goldids,'max_sorted_score_error':float(np.max(np.abs(np.array(scores)-v)))})
    (out/f'validation-{a.split}-{a.tag}.json').write_text(json.dumps(checks,indent=2));print('VALIDATION cases',len(checks),'failures',[c for c in checks if c['lower_violations'] or c['upper_violations'] or not c['top100_set_equal']],flush=True)
