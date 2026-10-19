"""Choose representatives by uncovered codebook tail-score mass, not cosine."""
import torch,triton
import triton.language as tl
from frontier_anchor_engine import AnchorFrontier
from frontier_anchor_kernels import anchor_encode,sum_anchor_bits
from frontier_bits import make_tiles,scatter_tiles
from gpu_candidate_kernels import compact_flags

@triton.jit
def tail_losses(Lut,TopIds,TopValues,Loss,P:tl.constexpr):
 a=tl.program_id(0);q=tl.arange(0,32);t=tl.arange(0,P)
 c=tl.load(TopIds+t[:,None]*32+q[None,:]);v=tl.load(TopValues+t[:,None]*32+q[None,:]).to(tl.float32)
 av=tl.load(Lut+c*32+a).to(tl.float32);cap=tl.load(TopValues+(P-1)*32+a).to(tl.float32)
 tail=tl.load(TopValues+(P-1)*32+q).to(tl.float32)
 mx=tl.max(tl.where(av<cap,v,float('-inf')),0)
 loss=tl.maximum(mx-tail,0);tl.store(Loss+a*32+q,loss)

@triton.jit
def choose_cover(Loss,Anchors,Groups,G:tl.constexpr):
 a=tl.arange(0,32);q=tl.arange(0,32);loss=tl.load(Loss+a[:,None]*32+q[None,:])
 best=tl.full((32,),float('inf'),tl.float32);selected=tl.full((32,),False,tl.int1);assignment=tl.full((32,),0,tl.int32)
 for g in tl.static_range(G):
  costs=tl.sum(tl.minimum(loss,best[None,:]),1);pick=tl.argmin(tl.where(selected,float('inf'),costs),0)
  tl.store(Anchors+g,pick);lv=tl.load(Loss+pick*32+q)
  assignment=tl.where(lv<best,g,assignment);best=tl.minimum(best,lv);selected=selected|(a==pick)
 tl.store(Groups+q,assignment)

@triton.jit
def conditional_group_table(Lut,TopIds,TopValues,Thresholds,Anchors,Groups,Table,G:tl.constexpr,L:tl.constexpr,P:tl.constexpr):
 g=tl.program_id(0);b=tl.program_id(1);q=tl.arange(0,32);t=tl.arange(0,P)
 group=tl.load(Groups+q);active=group==g;anchor=tl.load(Anchors+g)
 cap=tl.load(TopValues+anchor).to(tl.float32) if b==0 else tl.load(Thresholds+(b-1)*G+g)
 c=tl.load(TopIds+t[:,None]*32+q[None,:],active[None,:],0)
 value=tl.load(TopValues+t[:,None]*32+q[None,:],active[None,:],float('-inf')).to(tl.float32)
 av=tl.load(Lut+c*32+anchor,active[None,:],float('inf')).to(tl.float32)
 compatible=(av<=cap) if b==0 else (av<cap)
 candidate=tl.max(tl.where(compatible,value,float('-inf')),0)
 tail=tl.load(TopValues+(P-1)*32+q).to(tl.float32)
 safe=tl.maximum(candidate,tail)
 # The anchor's own contribution is directly bounded by its bin endpoint.
 safe=tl.where(q==anchor,cap,safe)
 tl.store(Table+g*(L+1)+b,tl.sum(tl.where(active,safe,0)))


@triton.jit
def score_thresholds(TopValues,Anchors,Fractions,Thresholds,G:tl.constexpr,L:tl.constexpr,P:tl.constexpr):
 g=tl.arange(0,triton.next_power_of_2(G));l=tl.arange(0,L);a=tl.load(Anchors+g,g<G,0)
 lo=tl.load(TopValues+(P-1)*32+a).to(tl.float32);hi=tl.load(TopValues+a).to(tl.float32);f=tl.load(Fractions+l)
 tl.store(Thresholds+l[:,None]*G+g[None,:],lo[None,:]+(hi-lo)[None,:]*f[:,None],g[None,:]<G)

class RankedAnchorFrontier(AnchorFrontier):
 def __init__(self,*args,score_power=0.0,**kwargs):
  super().__init__(*args,**kwargs);self.loss=torch.empty((32,32),device=self.cent.device);self.score_power=score_power
  self.fractions=torch.linspace(1,0,self.levels+1,device=self.cent.device)[1:]**max(score_power,.01);self.thresholds=torch.empty((self.levels,self.groups),device=self.cent.device)
 def decode(self,lut,top,th):
  sum_anchor_bits[(triton.cdiv(self.n,32),)](self.dbits,self.table,self.upper,N=self.n,G=self.groups,L=self.levels,W=self.words,B=32)
 def __call__(self,q,probe,budget):
  n=self.n;c=self.c;G=self.groups;L=self.levels;W=self.words
  lut=(self.cent@q.T).contiguous();top=lut.topk(self.prefix,dim=0)
  tail_losses[(32,)](lut,top.indices,top.values,self.loss,P=self.prefix,num_warps=8)
  choose_cover[(1,)](self.loss,self.anchors,self.assign,G=G,num_warps=4)
  if self.score_power>0:
   score_thresholds[(1,)](top.values,self.anchors,self.fractions,self.thresholds,G=G,L=L,P=self.prefix);th=self.thresholds
  else:th=top.values[self.ranks][:,self.anchors.long()].float().contiguous()
  conditional_group_table[(G,L+1)](lut,top.indices,top.values,th,self.anchors,self.assign,self.table,G=G,L=L,P=self.prefix,num_warps=8)
  anchor_encode[(triton.cdiv(c,32),)](lut,th,self.anchors,self.cbits,self.cf,C=c,G=G,L=L,W=W,B=32)
  self.cn.zero_();compact_flags[(triton.cdiv(c,256),)](self.cf,self.ci,self.cn,N=c,B=256);self.dbits.zero_();self.tn.zero_()
  make_tiles[(2048,)](self.ci,self.cn,self.starts,self.tc,self.to,self.tn,T=128,B=128)
  scatter_tiles[(2048,)](self.cbits,self.tc,self.to,self.tn,self.starts,self.ivf,self.flags,self.dbits,W=W,B=128)
  self.decode(lut,top,th)
  return self.finish(lut,budget)
