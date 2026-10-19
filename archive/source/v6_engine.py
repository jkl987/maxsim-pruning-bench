"""Diagnostic implementation; keeps v5 source untouched. Coarse TopB only."""
import torch,triton
from frontier_v5_bounds import MatchedBoundFrontier,additive_parts,simple_table
from frontier_rankedanchor_engine import tail_losses,choose_cover,score_thresholds,conditional_group_table
from frontier_v5_power32 import independent_table
from frontier_anchor_kernels import anchor_encode
from frontier_certified_guard import sum_certified_anchor_bits
from frontier_bits import make_tiles,scatter_tiles
from gpu_candidate_kernels import compact_flags,centroid_lookup
from frontier_kernels import seed_map,compact_survivors,scatter_scores

def exact_table(lut,anchors,assign,thresholds,L):
 """Full-codebook offline envelope, no TopP truncation. Construction is timed separately."""
 G=len(anchors);table=torch.zeros((G,L+1),device=lut.device)
 for g,a in enumerate(anchors.tolist()):
  members=torch.where(assign==g)[0]
  if not len(members):continue
  values,order=torch.sort(lut[:,a]);v=lut[order][:,members].float()
  prefix=torch.cummax(v,dim=0).values
  positions=torch.searchsorted(values,thresholds[:,g].contiguous(),right=False)-1
  caps=torch.cat([values[-1:].float(),thresholds[:,g]])
  bounds=torch.cat([v.max(0).values[None,:],prefix[positions.clamp_min(0)]])
  bounds[1:]=torch.where((positions>=0)[:,None],bounds[1:],torch.full_like(bounds[1:],-float('inf')))
  bounds[:,members==a]=caps[:,None]
  table[g]=bounds.sum(1)
 return table

class DiagnosticFrontier(MatchedBoundFrontier):
 def __init__(self,*args,kind='conditional',**kwargs):
  self.kind=kind
  super().__init__(*args,bound='additive' if kind=='additive' else 'conditional',**kwargs)
  if kind=='independent':
   assert self.groups==32
   self.anchors.copy_(torch.arange(32,device=self.cent.device));self.assign.copy_(self.anchors)
  self.events=[]
 def stamp(self,name,instrument):
  if instrument:
   e=torch.cuda.Event(enable_timing=True);e.record();self.events.append((name,e))
 def __call__(self,q,probe,budget,override=None,instrument=False):
  self.events=[];self.stamp('start',instrument)
  n=self.n;c=self.c;G=self.groups;L=self.levels;W=self.words
  lut=(self.cent@q.T).contiguous();top=lut.topk(self.prefix,dim=0);self.stamp('lut_top',instrument)
  if self.kind!='independent':
   tail_losses[(32,)](lut,top.indices,top.values,self.loss,P=self.prefix,num_warps=8)
   choose_cover[(1,)](self.loss,self.anchors,self.assign,G=G,num_warps=4)
  score_thresholds[(1,)](top.values,self.anchors,self.fractions,self.thresholds,G=G,L=L,P=self.prefix)
  if override is not None:self.table.copy_(override)
  elif self.kind=='independent':independent_table[(L+1,)](top.values,self.thresholds,self.table,L=L)
  elif self.kind=='additive':
   additive_parts[(triton.cdiv(c,512),)](lut,self.anchors,self.assign,self.delta_parts,C=c,B=512,num_warps=8)
   torch.amax(self.delta_parts,dim=0,out=self.delta)
   simple_table[(G,L+1)](top.values,self.thresholds,self.anchors,self.assign,self.delta,self.table,G=G,L=L,MODE=1)
  else:conditional_group_table[(G,L+1)](lut,top.indices,top.values,self.thresholds,self.anchors,self.assign,self.table,G=G,L=L,P=self.prefix,num_warps=8)
  self.stamp('selection_table',instrument)
  anchor_encode[(triton.cdiv(c,32),)](lut,self.thresholds,self.anchors,self.cbits,self.cf,C=c,G=G,L=L,W=W,B=32)
  self.cn.zero_();compact_flags[(triton.cdiv(c,256),)](self.cf,self.ci,self.cn,N=c,B=256);self.dbits.zero_();self.tn.zero_()
  make_tiles[(2048,)](self.ci,self.cn,self.starts,self.tc,self.to,self.tn,T=128,B=128);self.stamp('encode_tiles',instrument)
  scatter_tiles[(2048,)](self.cbits,self.tc,self.to,self.tn,self.starts,self.ivf,self.flags,self.dbits,W=W,B=128);self.stamp('scatter',instrument)
  sum_certified_anchor_bits[(triton.cdiv(n,32),)](self.dbits,self.table,self.upper,N=n,G=G,L=L,W=W,B=32);self.stamp('decode',instrument)
  seedsize=budget*self.factor;seed=self.upper.topk(seedsize,sorted=False).indices;seedids=self.ids[seed].contiguous();self.stamp('seed_select',instrument)
  centroid_lookup[(self.programs,)](lut,self.codes,self.off,seedids,self.seed_counts[budget],self.ss,B=self.block,num_warps=4,num_stages=1);self.stamp('seed_score',instrument)
  self.cut=self.ss[:seedsize].topk(budget,sorted=False).values.min()
  self.already.zero_();self.scores.fill_(-float('inf'))
  seed_map[(triton.cdiv(seedsize,256),)](seed,self.ss,self.already,self.scores,K=seedsize,B=256);self.wn.zero_()
  compact_survivors[(triton.cdiv(n,256),)](self.upper,self.cut,self.already,self.ids,self.count,self.wi,self.wp,self.wn,self.survivors,B=256);self.stamp('threshold_compact',instrument)
  centroid_lookup[(self.programs,)](lut,self.codes,self.off,self.wi,self.wn,self.ws,B=self.block,num_warps=4,num_stages=1);self.stamp('survivor_score',instrument)
  scatter_scores[(triton.cdiv(n,256),)](self.wp,self.wn,self.ws,self.scores,B=256)
  v,j=self.scores.topk(budget);self.stamp('final_select',instrument)
  return j,torch.isfinite(v)
 def phase_ms(self):return {b[0]:a[1].elapsed_time(b[1])for a,b in zip(self.events,self.events[1:])}
 def owned_bytes(self):
  shared={x.untyped_storage().data_ptr()for x in [self.cent,self.codes,self.off,self.starts,self.ivf]};seen={}
  def visit(x):
   if isinstance(x,torch.Tensor) and x.is_cuda:
    s=x.untyped_storage();p=s.data_ptr()
    if p not in shared:seen[p]=s.nbytes()
   elif isinstance(x,dict):
    for v in x.values():visit(v)
  for x in vars(self).values():visit(x)
  return sum(seen.values())
