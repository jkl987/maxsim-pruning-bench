"""Matched-state ablations. Additive/global bounds are controls, not new methods."""
import torch,triton
import triton.language as tl
from frontier_certified_guard import GuardedDenseRankedFrontier
from frontier_rankedanchor_engine import tail_losses,choose_cover,score_thresholds,conditional_group_table
from frontier_anchor_kernels import anchor_encode
from frontier_bits import make_tiles,scatter_tiles
from gpu_candidate_kernels import compact_flags

@triton.jit
def additive_parts(Lut,Anchors,Groups,Parts,C:tl.constexpr,B:tl.constexpr):
 c=tl.program_id(0)*B+tl.arange(0,B);q=tl.arange(0,32)
 group=tl.load(Groups+q);aa=tl.load(Anchors+group)
 x=tl.load(Lut+c[:,None]*32+q[None,:],c[:,None]<C,0).to(tl.float32)
 y=tl.load(Lut+c[:,None]*32+aa[None,:],c[:,None]<C,0).to(tl.float32)
 delta=tl.max(tl.where(c[:,None]<C,x-y,float('-inf')),0)
 tl.store(Parts+tl.program_id(0)*32+q,delta)

@triton.jit
def simple_table(TopValues,Thresholds,Anchors,Groups,Delta,Table,G:tl.constexpr,L:tl.constexpr,MODE:tl.constexpr):
 g=tl.program_id(0);b=tl.program_id(1);q=tl.arange(0,32)
 active=tl.load(Groups+q)==g;anchor=tl.load(Anchors+g)
 cap=tl.load(TopValues+anchor).to(tl.float32) if b==0 else tl.load(Thresholds+(b-1)*G+g)
 globalmax=tl.load(TopValues+q).to(tl.float32)
 value=globalmax
 if MODE==1:
  # Two explicit outward allowances cover FP32 delta subtraction and cap addition
  # for bounded absolute LUT values <=2. Final accumulation guard remains4e-4.
  delta=tl.load(Delta+q)+1e-6
  value=tl.minimum(globalmax,cap+delta+1e-6)
 value=tl.where(q==anchor,cap,value)
 tl.store(Table+g*(L+1)+b,tl.sum(tl.where(active,value,0)))

class MatchedBoundFrontier(GuardedDenseRankedFrontier):
 def __init__(self,*args,bound='conditional',**kwargs):
  super().__init__(*args,**kwargs);assert bound in ['conditional','additive','unconditional'];self.bound=bound
  self.delta_parts=torch.empty((triton.cdiv(self.c,512),32),device=self.cent.device);self.delta=torch.empty(32,device=self.cent.device)
 def __call__(self,q,probe,budget):
  n=self.n;c=self.c;G=self.groups;L=self.levels;W=self.words
  lut=(self.cent@q.T).contiguous();top=lut.topk(self.prefix,dim=0)
  tail_losses[(32,)](lut,top.indices,top.values,self.loss,P=self.prefix,num_warps=8)
  choose_cover[(1,)](self.loss,self.anchors,self.assign,G=G,num_warps=4)
  score_thresholds[(1,)](top.values,self.anchors,self.fractions,self.thresholds,G=G,L=L,P=self.prefix)
  th=self.thresholds
  if self.bound=='conditional':
   conditional_group_table[(G,L+1)](lut,top.indices,top.values,th,self.anchors,self.assign,self.table,G=G,L=L,P=self.prefix,num_warps=8)
  else:
   if self.bound=='additive':
    additive_parts[(triton.cdiv(c,512),)](lut,self.anchors,self.assign,self.delta_parts,C=c,B=512,num_warps=8)
    torch.amax(self.delta_parts,dim=0,out=self.delta)
   simple_table[(G,L+1)](top.values,th,self.anchors,self.assign,self.delta,self.table,G=G,L=L,MODE=int(self.bound=='additive'))
  anchor_encode[(triton.cdiv(c,32),)](lut,th,self.anchors,self.cbits,self.cf,C=c,G=G,L=L,W=W,B=32)
  self.cn.zero_();compact_flags[(triton.cdiv(c,256),)](self.cf,self.ci,self.cn,N=c,B=256);self.dbits.zero_();self.tn.zero_()
  make_tiles[(2048,)](self.ci,self.cn,self.starts,self.tc,self.to,self.tn,T=128,B=128)
  scatter_tiles[(2048,)](self.cbits,self.tc,self.to,self.tn,self.starts,self.ivf,self.flags,self.dbits,W=W,B=128)
  self.decode(lut,top,th)
  return self.finish(lut,budget)
