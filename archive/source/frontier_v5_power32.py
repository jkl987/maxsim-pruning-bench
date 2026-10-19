"""Stronger control:32independent terms with the SAME score-space bin schedule.
No representative selection or cross-term bound. Production16state code untouched.
"""
import torch,triton
import triton.language as tl
from frontier_densefinish_engine import DenseNormalFrontier
from frontier_rankedanchor_engine import score_thresholds
from frontier_anchor_kernels import anchor_encode
from frontier_certified_guard import sum_certified_anchor_bits
from frontier_bits import make_tiles,scatter_tiles
from gpu_candidate_kernels import compact_flags
@triton.jit
def independent_table(TopValues,Thresholds,Table,L:tl.constexpr):
 q=tl.arange(0,32);b=tl.program_id(0)
 v=tl.load(TopValues+q).to(tl.float32) if b==0 else tl.load(Thresholds+(b-1)*32+q)
 tl.store(Table+q*(L+1)+b,v)
class Power32Frontier(DenseNormalFrontier):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.anchors=torch.arange(32,device=self.cent.device,dtype=torch.int32);self.fractions=torch.linspace(1,0,self.levels+1,device=self.cent.device)[1:]**2;self.thresholds=torch.empty((self.levels,32),device=self.cent.device);self.table=torch.empty((32,self.levels+1),device=self.cent.device)
 def __call__(self,q,probe,budget):
  n=self.n;c=self.c;L=self.levels;lut=(self.cent@q.T).contiguous();top=lut.topk(self.prefix,dim=0)
  score_thresholds[(1,)](top.values,self.anchors,self.fractions,self.thresholds,G=32,L=L,P=self.prefix)
  independent_table[(L+1,)](top.values,self.thresholds,self.table,L=L)
  anchor_encode[(triton.cdiv(c,32),)](lut,self.thresholds,self.anchors,self.cbits,self.cf,C=c,G=32,L=L,W=L//2,B=32)
  self.cn.zero_();compact_flags[(triton.cdiv(c,256),)](self.cf,self.ci,self.cn,N=c,B=256);self.dbits.zero_();self.tn.zero_()
  make_tiles[(2048,)](self.ci,self.cn,self.starts,self.tc,self.to,self.tn,T=128,B=128)
  scatter_tiles[(2048,)](self.cbits,self.tc,self.to,self.tn,self.starts,self.ivf,self.flags,self.dbits,W=L//2,B=128)
  sum_certified_anchor_bits[(triton.cdiv(n,32),)](self.dbits,self.table,self.upper,N=n,G=32,L=L,W=L//2,B=32)
  return self.finish(lut,budget)
