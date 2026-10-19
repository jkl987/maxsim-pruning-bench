"""Information-only local8pair test. Selection usesquery/centroids, never true scores/times."""
from pathlib import Path
R=Path(__file__).resolve().parent
source=(R/'v6_diagnose.py').read_text();setup=source[:source.index('spec=[')];setup='\n'.join(line for line in setup.splitlines()if not line.startswith('if dest.exists()'));exec(compile(setup,'<diagnostic shared input setup>','exec'))
import triton,triton.language as tl
from frontier_rankedanchor_engine import tail_losses
@triton.jit
def term_decode(Bits,Guide,Stride,Table,Out,N:tl.constexpr,B:tl.constexpr):
 d=tl.program_id(0)*B+tl.arange(0,B);t=tl.arange(0,32);g=tl.load(Guide+t);step=tl.load(Stride+t)
 bits=tl.load(Bits+d[:,None]*4+g[None,:]//8,d[:,None]<N,0);bb=((bits>>((g[None,:]%8)*8))&255).to(tl.uint32);before=(bb&(0-bb))-1
 ix=tl.inline_asm_elementwise('popc.b32 $0, $1;',constraints='=r,r',args=[before],dtype=tl.int32,is_pure=True,pack=1);ix=tl.minimum(ix,8);ix=(ix//step[None,:])*step[None,:]
 v=tl.load(Table+t[None,:]*9+ix);tl.store(Out+d,tl.sum(v,1)+4e-4,d<N)
e=DiagnosticFrontier(cent,codes,off,io,ivf,budget,512,2,8,groups=32,score_power=2.,kind='independent',programs=8192,block=256,compact_final=False,unsorted_seed=True)
ref=torch.empty(n,device='cuda');rows=[]
for qi in order:
 q=torch.from_numpy(qq[qi]).cuda();e(q,16,budget);lut=cent@q.T;top=lut.topk(512,dim=0)
 tail_losses[(32,)](lut,top.indices,top.values,e.loss,P=512,num_warps=8)
 # Fixed deterministic proxy: normalized conditional tail loss, cosine only a tiny tie-break.
 score=e.loss/(top.values[0].float()-top.values[-1].float()).clamp_min(1e-3)[None,:];score+=.001*(1-q.float()@q.float().T)
 sc=score.cpu().numpy();used=set();guide=np.arange(32,dtype='int32');pairs=[]
 for _ in range(8):
  choices=[(float(sc[a0,i]),a0,i)for a0 in range(32)for i in range(32)if a0!=i and a0 not in used and i not in used]
  cost,ar,term=min(choices);pairs.append([ar,term,cost]);used.update([ar,term]);guide[term]=ar
 centroid_lookup[(8192,)](lut,codes,off,e.ids,e.count,ref,B=256,num_warps=4,num_stages=1);cut=ref.topk(budget).values[-1]
 independent=e.table.clone();ideal=independent.clone();approx=independent.clone();cap=torch.cat([top.values[:1].float(),e.thresholds],0)
 for ar,term,_ in pairs:
  values,idx=torch.sort(lut[:,ar].float());vv=lut[idx,term].float();pref=torch.cummax(vv,0).values;pos=torch.searchsorted(values,e.thresholds[:,ar].contiguous(),right=False)-1
  ideal[term,0]=vv.max();ideal[term,1:]=pref[pos.clamp_min(0)]
  assert (pos>=0).all()
  for b in range(9):
   mask=lut[top.indices[:,term],ar].float()<=cap[b,ar]if b==0 else lut[top.indices[:,term],ar].float()<cap[b,ar]
   approx[term,b]=torch.maximum(top.values[-1,term].float(),top.values[:,term].float().masked_fill(~mask,-float('inf')).max())
 assert float((approx-ideal).min())>=-1e-6, float((approx-ideal).min())
 guide_gpu=torch.from_numpy(guide).cuda();identity=torch.arange(32,device='cuda',dtype=torch.int32);one=torch.ones(32,device='cuda',dtype=torch.int32);mixed=torch.full_like(one,2);mixed[(top.values[0]-top.values[-1]).topk(16).indices]=1
 z={}
 for name,gg,step,table in [('ind32_8',identity,one,independent),('ind32_4',identity,one*2,independent),('mixed24',identity,mixed,independent),('local24_ideal',guide_gpu,one,ideal),('local24_topP',guide_gpu,one,approx)]:
  term_decode[(triton.cdiv(n,32),)](e.dbits,gg,step,table,e.upper,N=n,B=32)
  violations=int((e.upper<ref).sum());assert violations==0,(name,violations)
  ret,_=e.finish(lut,budget);selected=torch.zeros(n,dtype=torch.bool,device='cuda');selected[ret]=True;assert not((ref>cut)&~selected).any()
  z[name]={'false_survivors_truecut':int(((e.upper>=cut)&(ref<cut)).sum()),'slack_mean':float((e.upper-ref).mean()),'scored_documents':int(e.survivors.sum()),'scored_tokens':int(((off[1:]-off[:-1]).long()*e.survivors).sum())}
 rows.append({'query_id':qids[qi],'pairs':pairs,'metrics':z,'ideal_vs_topP_max_gap':float((approx-ideal).max())});out={'complete':len(rows)==len(order),'dataset':ds,'queries':len(rows),'scope':'Information-only diagnostic. Pairpolicy usesnormalized tail loss plus.001cosinetiebreak,8disjointpairs. Completecodebook oracle cannotbeusedonlineforfree. No formalperformanceclaimed. 24B layout notimplemented here; decodedfrom32B reference state.','records':rows}
 (R/'v6'/f'{ds}-local-ceiling.json').write_text(json.dumps(out));print('LOCAL CEILING',ds,len(rows),flush=True)
