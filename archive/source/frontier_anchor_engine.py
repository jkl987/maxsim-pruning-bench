import torch,triton
from frontier_factor_engine import FinishFrontier
from frontier_anchor_kernels import choose_anchors,anchor_encode,conditional_group_table,sum_anchor_bits
from frontier_bits import make_tiles,scatter_tiles
from gpu_candidate_kernels import compact_flags

class AnchorFrontier(FinishFrontier):
 def __init__(self,*args,groups=8,**kwargs):
  super().__init__(*args,**kwargs);self.groups=groups;self.words=triton.cdiv(groups*self.levels,64)
  self.anchors=torch.empty(groups,device=self.cent.device,dtype=torch.int32);self.assign=torch.empty(32,device=self.cent.device,dtype=torch.int32)
  self.table=torch.empty((groups,self.levels+1),device=self.cent.device)
  self.cbits=torch.empty((self.c,self.words),device=self.cent.device,dtype=torch.uint64);self.dbits=torch.empty((self.n,self.words),device=self.cent.device,dtype=torch.uint64)
 def __call__(self,q,probe,budget):
  n=self.n;c=self.c;G=self.groups;L=self.levels;W=self.words
  lut=(self.cent@q.T).contiguous();top=lut.topk(self.prefix,dim=0)
  cos=q.float()@q.float().T;choose_anchors[(1,)](cos,self.anchors,self.assign,G=G,num_warps=4)
  th=top.values[self.ranks][:,self.anchors.long()].float().contiguous()
  conditional_group_table[(G,L+1)](lut,top.indices,top.values,th,self.anchors,self.assign,self.table,G=G,L=L,P=self.prefix,num_warps=8)
  anchor_encode[(triton.cdiv(c,32),)](lut,th,self.anchors,self.cbits,self.cf,C=c,G=G,L=L,W=W,B=32)
  self.cn.zero_();compact_flags[(triton.cdiv(c,256),)](self.cf,self.ci,self.cn,N=c,B=256);self.dbits.zero_();self.tn.zero_()
  make_tiles[(2048,)](self.ci,self.cn,self.starts,self.tc,self.to,self.tn,T=128,B=128)
  scatter_tiles[(2048,)](self.cbits,self.tc,self.to,self.tn,self.starts,self.ivf,self.flags,self.dbits,W=W,B=128)
  sum_anchor_bits[(triton.cdiv(n,32),)](self.dbits,self.table,self.upper,N=n,G=G,L=L,W=W,B=32)
  return self.finish(lut,budget)
