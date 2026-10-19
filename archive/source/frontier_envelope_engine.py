"""Synthetic max-envelope summaries covering assigned query score functions.
An envelope is a scalar function on the codebook, not necessarily a vector.
Every group member obeys score_i(c) <= envelope_g(c) + offset_i.
"""
import torch,triton
import triton.language as tl
from frontier_coveranchor_engine import CoverAnchorFrontier,tail_losses,choose_cover
from frontier_anchor_kernels import choose_anchors,sum_anchor_bits
from frontier_bits import make_tiles,scatter_tiles
from gpu_candidate_kernels import compact_flags

@triton.jit
def build_envelopes(Lut,TopValues,Groups,Env,Offsets,C:tl.constexpr,G:tl.constexpr,P:tl.constexpr,SHIFT:tl.constexpr,B:tl.constexpr):
 g=tl.program_id(0);c=tl.program_id(1)*B+tl.arange(0,B);q=tl.arange(0,32)
 assignment=tl.load(Groups+q);offset=tl.load(TopValues+(P-1)*32+q).to(tl.float32) if SHIFT else tl.full((32,),0,tl.float32)
 values=tl.load(Lut+c[:,None]*32+q[None,:],c[:,None]<C,0).to(tl.float32)-offset[None,:]
 z=tl.max(tl.where(assignment[None,:]==g,values,float('-inf')),1)
 tl.store(Env+c*G+g,z,c<C)
 if g==0 and tl.program_id(1)==0:tl.store(Offsets+q,offset)

@triton.jit
def group_thresholds(TopValues,Offsets,Groups,Ranks,Thresholds,G:tl.constexpr,L:tl.constexpr):
 g=tl.program_id(0);b=tl.program_id(1);q=tl.arange(0,32)
 assignment=tl.load(Groups+q);r=tl.load(Ranks+b);v=tl.load(TopValues+r*32+q).to(tl.float32)-tl.load(Offsets+q)
 tl.store(Thresholds+b*G+g,tl.max(tl.where(assignment==g,v,float('-inf')),0))

@triton.jit
def envelope_table(Lut,TopIds,TopValues,Env,Offsets,Thresholds,Groups,Table,G:tl.constexpr,L:tl.constexpr,P:tl.constexpr,TIGHT:tl.constexpr):
 g=tl.program_id(0);b=tl.program_id(1);q=tl.arange(0,32);t=tl.arange(0,P)
 assignment=tl.load(Groups+q);active=assignment==g;offset=tl.load(Offsets+q)
 cap=tl.load(Thresholds+tl.maximum(b-1,0)*G+g)
 if TIGHT:
  c=tl.load(TopIds+t[:,None]*32+q[None,:],active[None,:],0)
  value=tl.load(TopValues+t[:,None]*32+q[None,:],active[None,:],float('-inf')).to(tl.float32)
  z=tl.load(Env+c*G+g,active[None,:],float('inf'))
  ok=(z<=cap) if b==0 else (z<cap)
  candidate=tl.max(tl.where(ok,value,float('-inf')),0)
  tail=tl.load(TopValues+(P-1)*32+q).to(tl.float32)
  bound=tl.minimum(tl.maximum(candidate,tail),cap+offset+1.e-6)
 else:bound=cap+offset+1.e-6
 tl.store(Table+g*(L+1)+b,tl.sum(tl.where(active,bound,0)))

@triton.jit
def envelope_encode(Env,Thresholds,CellBits,Flags,C:tl.constexpr,G:tl.constexpr,L:tl.constexpr,W:tl.constexpr,B:tl.constexpr):
 c=tl.program_id(0)*B+tl.arange(0,B);g=tl.arange(0,W*(64//L))
 v=tl.load(Env+c[:,None]*G+g[None,:],(c[:,None]<C)&(g[None,:]<G),float('-inf'))
 bits=tl.full((B,W*(64//L)),0,tl.uint64)
 for lev in tl.static_range(L):
  th=tl.load(Thresholds+lev*G+g,g<G,float('inf'))
  bits|=tl.where((v>=th[None,:])&(g[None,:]<G)&(v!=float('-inf')),1<<lev,0).to(tl.uint64)
 packed=tl.sum(tl.reshape(bits,(B,W,64//L))<<(tl.arange(0,64//L)[None,None,:]*L),2)
 tl.store(CellBits+c[:,None]*W+tl.arange(0,W)[None,:],packed,c[:,None]<C)
 tl.store(Flags+c,tl.sum((bits!=0).to(tl.int32),1)>0,c<C)

class EnvelopeFrontier(CoverAnchorFrontier):
 def __init__(self,*args,shift=True,tight=True,selection='cover',quantile=False,**kwargs):
  super().__init__(*args,**kwargs);self.shift=shift;self.tight=tight;self.selection=selection;self.quantile=quantile;self.thresholds=torch.empty((self.levels,self.groups),device=self.cent.device)
  self.envelopes=torch.empty((self.c,self.groups),device=self.cent.device)
  self.offsets=torch.empty(32,device=self.cent.device)
 def __call__(self,q,probe,budget):
  n=self.n;c=self.c;G=self.groups;L=self.levels;W=self.words
  lut=(self.cent@q.T).contiguous();top=lut.topk(self.prefix,dim=0)
  if self.selection=='cover':
   tail_losses[(32,)](lut,top.indices,top.values,self.loss,P=self.prefix,num_warps=8)
   choose_cover[(1,)](self.loss,self.anchors,self.assign,G=G,num_warps=4)
  else:
   cos=q.float()@q.float().T;choose_anchors[(1,)](cos,self.anchors,self.assign,G=G,num_warps=4)
  build_envelopes[(G,triton.cdiv(c,32))](lut,top.values,self.assign,self.envelopes,self.offsets,C=c,G=G,P=self.prefix,SHIFT=self.shift,B=32,num_warps=4)
  if self.quantile:
   group_thresholds[(G,L)](top.values,self.offsets,self.assign,self.ranks,self.thresholds,G=G,L=L);th=self.thresholds
  else:
   etop=self.envelopes.topk(self.prefix,dim=0);th=etop.values[self.ranks].contiguous()
  envelope_table[(G,L+1)](lut,top.indices,top.values,self.envelopes,self.offsets,th,self.assign,self.table,G=G,L=L,P=self.prefix,TIGHT=self.tight,num_warps=8)
  envelope_encode[(triton.cdiv(c,32),)](self.envelopes,th,self.cbits,self.cf,C=c,G=G,L=L,W=W,B=32)
  self.cn.zero_();compact_flags[(triton.cdiv(c,256),)](self.cf,self.ci,self.cn,N=c,B=256);self.dbits.zero_();self.tn.zero_()
  make_tiles[(2048,)](self.ci,self.cn,self.starts,self.tc,self.to,self.tn,T=128,B=128)
  scatter_tiles[(2048,)](self.cbits,self.tc,self.to,self.tn,self.starts,self.ivf,self.flags,self.dbits,W=W,B=128)
  sum_anchor_bits[(triton.cdiv(n,32),)](self.dbits,self.table,self.upper,N=n,G=G,L=L,W=W,B=32)
  return self.finish(lut,budget)
