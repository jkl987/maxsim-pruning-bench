"""Engineering ablation: sort only compacted, actually scored candidates."""
import torch,triton
from frontier_rankedanchor_engine import RankedAnchorFrontier
from frontier_coveranchor_engine import CoverAnchorFrontier
from gpu_candidate_kernels import centroid_lookup
from frontier_kernels import seed_map,compact_survivors,scatter_scores

class DenseFinishMixin:
 def __init__(self,*args,compact_final=True,unsorted_seed=True,programs=512,block=128,**kwargs):
  super().__init__(*args,**kwargs);self.compact_final=compact_final;self.unsorted_seed=unsorted_seed;self.programs=programs;self.block=block
 def finish(self,lut,budget):
  n=self.n;seedsize=budget*self.factor;seed=self.upper.topk(seedsize,sorted=not self.unsorted_seed).indices;seedids=self.ids[seed].contiguous();seedn=self.seed_counts[budget]
  centroid_lookup[(self.programs,)](lut,self.codes,self.off,seedids,seedn,self.ss,B=self.block,num_warps=4,num_stages=1)
  if self.unsorted_seed:cut=self.ss[:seedsize].topk(budget,sorted=False).values.min()
  else:cut=self.ss[:seedsize].topk(budget).values[-1]
  self.already.zero_();self.scores.fill_(-float('inf'))
  seed_map[(triton.cdiv(seedsize,256),)](seed,self.ss,self.already,self.scores,K=seedsize,B=256);self.wn.zero_()
  compact_survivors[(triton.cdiv(n,256),)](self.upper,cut,self.already,self.ids,self.count,self.wi,self.wp,self.wn,self.survivors,B=256)
  centroid_lookup[(self.programs,)](lut,self.codes,self.off,self.wi,self.wn,self.ws,B=self.block,num_warps=4,num_stages=1)
  scatter_scores[(triton.cdiv(n,256),)](self.wp,self.wn,self.ws,self.scores,B=256)
  if self.compact_final:
   # Synchronization is deliberately inside the measured call. No hidden count.
   count=int(self.wn);values=torch.cat((self.ss[:seedsize],self.ws[:count]));ids=torch.cat((seed,self.wi[:count].long()))
   v,j=values.topk(budget);return ids[j],torch.isfinite(v)
  v,j=self.scores.topk(budget);return j,torch.isfinite(v)

class DenseRankedFrontier(DenseFinishMixin,RankedAnchorFrontier):pass
class DenseCoverFrontier(DenseFinishMixin,CoverAnchorFrontier):pass

from frontier_anchor_engine import AnchorFrontier
class DenseCosineFrontier(DenseFinishMixin,AnchorFrontier):pass

from frontier_engine import Frontier
from frontier_bits import encode_cells,sum_bits,make_tiles,scatter_tiles
from gpu_candidate_kernels import compact_flags
class DenseNormalFrontier(DenseFinishMixin,Frontier):
 def __call__(self,q,probe,budget):
  n=self.n;c=self.c;L=self.levels;lut=(self.cent@q.T).contiguous();top=lut.topk(self.prefix,dim=0);th=top.values[self.ranks].float().contiguous()
  encode_cells[(triton.cdiv(c,32),)](lut,th,self.cbits,self.cf,C=c,L=L,B=32)
  self.cn.zero_();compact_flags[(triton.cdiv(c,256),)](self.cf,self.ci,self.cn,N=c,B=256);self.dbits.zero_();self.tn.zero_()
  make_tiles[(2048,)](self.ci,self.cn,self.starts,self.tc,self.to,self.tn,T=128,B=128)
  scatter_tiles[(2048,)](self.cbits,self.tc,self.to,self.tn,self.starts,self.ivf,self.flags,self.dbits,W=L//2,B=128)
  sum_bits[(triton.cdiv(n,32),)](self.dbits,self.ids,self.count,th,self.upper,L=L,B=32)
  return self.finish(lut,budget)
from frontier_flexibleanchor_engine import FlexibleAnchorFrontier
from frontier_envelope_engine import EnvelopeFrontier
class DenseFlexibleFrontier(DenseFinishMixin,FlexibleAnchorFrontier):pass
class DenseEnvelopeFrontier(DenseFinishMixin,EnvelopeFrontier):pass
