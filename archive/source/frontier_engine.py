"""Global centroid-topB retrieval using conservative multi-level membership bounds.
Scope: exact (modulo floating point guard and boundary ties) centroid score topB.
NOT an exact retrieval guarantee for original token vectors. The downstream B
budget still approximates original-vector search. Closely related to EMVB/XTR/NRA.
"""
import numpy as np,torch,triton
from gpu_candidate_kernels import compact_flags,centroid_lookup
from frontier_bits import encode_cells,sum_bits,make_tiles,scatter_tiles
from frontier_kernels import seed_map,compact_survivors,scatter_scores
class Frontier:
 def __init__(self,cent,codes,off,starts,ivf,maxbudget=16384,prefix=1024,seed_factor=2,levels=8):
  self.cent=cent;self.codes=codes;self.off=off;self.starts=starts;self.ivf=ivf;self.n=off.numel()-1;self.c=len(cent);self.prefix=prefix;self.factor=seed_factor;self.levels=levels
  n=self.n;c=self.c;dev=cent.device
  def buf(size,dtype=torch.int32):return torch.empty(size,device=dev,dtype=dtype)
  self.ids=torch.arange(n,device=dev,dtype=torch.int32);self.count=torch.tensor([n],device=dev,dtype=torch.int32);self.flags=torch.ones(n,device=dev,dtype=torch.int32)
  self.scores=buf(n,torch.float32);self.upper=buf(n,torch.float32);self.already=buf(n);self.survivors=buf(n)
  self.cbits=buf((c,levels//2),torch.uint64);self.dbits=buf((n,levels//2),torch.uint64);self.cf=buf(c);self.ci=buf(c);self.cn=buf(1)
  capacity=len(ivf)//128+c;self.tc=buf(capacity);self.to=buf(capacity,torch.int64);self.tn=buf(1)
  self.wi=buf(n);self.wp=buf(n);self.wn=buf(1);self.ws=buf(n,torch.float32);self.ss=buf(maxbudget*seed_factor,torch.float32)
  self.seed_counts={b:torch.tensor([b*seed_factor],device=dev,dtype=torch.int32) for b in [128,256,512,1024,2048,4096,8192,16384,32768,65536] if b<=maxbudget}
  self.ranks=torch.tensor(np.geomspace(1,prefix,levels).round().astype('int64')-1,device=dev)
 def __call__(self,q,probe,budget):
  n=self.n;c=self.c;L=self.levels;seedsize=budget*self.factor;lut=(self.cent@q.T).contiguous();top=lut.topk(self.prefix,dim=0);thresholds=top.values[self.ranks].float().contiguous()
  encode_cells[(triton.cdiv(c,32),)](lut,thresholds,self.cbits,self.cf,C=c,L=L,B=32);self.cn.zero_();compact_flags[(triton.cdiv(c,256),)](self.cf,self.ci,self.cn,N=c,B=256);self.dbits.zero_();self.tn.zero_()
  make_tiles[(2048,)](self.ci,self.cn,self.starts,self.tc,self.to,self.tn,T=128,B=128);scatter_tiles[(2048,)](self.cbits,self.tc,self.to,self.tn,self.starts,self.ivf,self.flags,self.dbits,W=L//2,B=128)
  sum_bits[(triton.cdiv(n,32),)](self.dbits,self.ids,self.count,thresholds,self.upper,L=L,B=32)
  seed=self.upper.topk(seedsize).indices;seedids=self.ids[seed].contiguous();seedn=self.seed_counts[budget]
  centroid_lookup[(256,)](lut,self.codes,self.off,seedids,seedn,self.ss,B=128,num_warps=4,num_stages=1);cut=self.ss[:seedsize].topk(budget).values[-1];self.already.zero_();self.scores.fill_(-float('inf'))
  seed_map[(triton.cdiv(seedsize,256),)](seed,self.ss,self.already,self.scores,K=seedsize,B=256);self.wn.zero_()
  compact_survivors[(triton.cdiv(n,256),)](self.upper,cut,self.already,self.ids,self.count,self.wi,self.wp,self.wn,self.survivors,B=256)
  centroid_lookup[(512,)](lut,self.codes,self.off,self.wi,self.wn,self.ws,B=128,num_warps=4,num_stages=1);scatter_scores[(triton.cdiv(n,256),)](self.wp,self.wn,self.ws,self.scores,B=256)
  v,j=self.scores.topk(budget);return j,torch.isfinite(v)
class BudgetFrontier(Frontier):
 """Fixed policy selected on 32 development queries, before test evaluation.
 Chooses near-best mean with a smaller observed worst-query cost. It changes
 certificate effort only, not requested global centroid topB semantics.
 """
 def __init__(self,cent,codes,off,starts,ivf,maxbudget=16384):
  super().__init__(cent,codes,off,starts,ivf,maxbudget,1024,4,8)
  self.policy={1024:(256,4),4096:(512,2),16384:(1024,2)}
  self.rank_plans={p:torch.tensor(np.geomspace(1,p,8).round().astype('int64')-1,device=cent.device) for p in [256,512,1024]}
  self.count_plans={f:{b:torch.tensor([b*f],device=cent.device,dtype=torch.int32) for b in self.seed_counts} for f in [2,4]}
 def __call__(self,q,probe,budget):
  self.prefix,self.factor=self.policy.get(budget,(1024,2));self.ranks=self.rank_plans[self.prefix];self.seed_counts=self.count_plans[self.factor]
  return super().__call__(q,probe,budget)
