"""Stronger implementation controls, not author EMVB/WARP implementations."""
import torch,triton
from frontier_engine import Frontier
from frontier_bits import encode_cells,sum_bits,make_tiles
from gpu_candidate_kernels import compact_flags,centroid_lookup
from frontier_kernels import seed_map,compact_survivors,scatter_scores,fill_prefix,sum_prefix
from frontier_factor_kernels import float_flags,fill_aos,scatter_float_union,sum_aos,forward_bits

class FinishFrontier(Frontier):
 def finish(self,lut,budget):
  n=self.n;seedsize=budget*self.factor;seed=self.upper.topk(seedsize).indices;seedids=self.ids[seed].contiguous();seedn=self.seed_counts[budget]
  centroid_lookup[(256,)](lut,self.codes,self.off,seedids,seedn,self.ss,B=128,num_warps=4,num_stages=1)
  cut=self.ss[:seedsize].topk(budget).values[-1];self.already.zero_();self.scores.fill_(-float('inf'))
  seed_map[(triton.cdiv(seedsize,256),)](seed,self.ss,self.already,self.scores,K=seedsize,B=256);self.wn.zero_()
  compact_survivors[(triton.cdiv(n,256),)](self.upper,cut,self.already,self.ids,self.count,self.wi,self.wp,self.wn,self.survivors,B=256)
  centroid_lookup[(512,)](lut,self.codes,self.off,self.wi,self.wn,self.ws,B=128,num_warps=4,num_stages=1)
  scatter_scores[(triton.cdiv(n,256),)](self.wp,self.wn,self.ws,self.scores,B=256)
  v,j=self.scores.topk(budget);return j,torch.isfinite(v)

class UnionFloatFrontier(FinishFrontier):
 def __init__(self,*args,aos=True,**kwargs):
  super().__init__(*args,**kwargs);self.aos=aos;self.matrix=torch.empty((self.n,32),device=self.cent.device)
 def __call__(self,q,probe,budget):
  n=self.n;c=self.c;lut=(self.cent@q.T).contiguous();top=lut.topk(self.prefix,dim=0);tail=top.values[-1].float()
  float_flags[(triton.cdiv(c,32),)](lut,tail,self.cf,C=c,B=32);self.cn.zero_();compact_flags[(triton.cdiv(c,256),)](self.cf,self.ci,self.cn,N=c,B=256)
  if self.aos:fill_aos[(triton.cdiv(n,32),)](self.matrix,tail,N=n,B=32)
  else:fill_prefix[(32,triton.cdiv(n,1024))](self.matrix,tail,N=n,B=1024)
  self.tn.zero_();make_tiles[(2048,)](self.ci,self.cn,self.starts,self.tc,self.to,self.tn,T=128,B=128)
  scatter_float_union[(2048,)](lut,tail,self.tc,self.to,self.tn,self.starts,self.ivf,self.matrix,N=n,AOS=self.aos,B=32,num_warps=4)
  if self.aos:sum_aos[(triton.cdiv(n,32),)](self.matrix,self.upper,N=n,B=32)
  else:sum_prefix[(triton.cdiv(n,128),)](self.matrix,self.ids,self.count,self.upper,N=n,B=128,num_warps=4)
  return self.finish(lut,budget)

class ForwardBitsFrontier(FinishFrontier):
 def __call__(self,q,probe,budget):
  n=self.n;c=self.c;L=self.levels;lut=(self.cent@q.T).contiguous();top=lut.topk(self.prefix,dim=0);th=top.values[self.ranks].float().contiguous()
  encode_cells[(triton.cdiv(c,32),)](lut,th,self.cbits,self.cf,C=c,L=L,B=32)
  forward_bits[(2048,)](self.cbits,self.codes,self.off,self.dbits,N=n,W=L//2,B=64,num_warps=4)
  sum_bits[(triton.cdiv(n,32),)](self.dbits,self.ids,self.count,th,self.upper,L=L,B=32)
  return self.finish(lut,budget)
