"""One small online8pair prototype; generic24B packing, matched independent24B control.
All pair choices use query/centroid responses, no document scores or observed timings.
"""
import torch,triton
import triton.language as tl
from v6_engine import DiagnosticFrontier
from frontier_rankedanchor_engine import tail_losses,score_thresholds,conditional_group_table
from frontier_bits import make_tiles,scatter_tiles
from gpu_candidate_kernels import compact_flags,centroid_lookup
from frontier_kernels import seed_map,compact_survivors,scatter_scores

@triton.jit
def select_pairs(Loss,Top,Cos,Anchors,Groups,Widths,Offsets,Trace,P:tl.constexpr):
 a=tl.arange(0,32);i=tl.arange(0,32);loss=tl.load(Loss+a[:,None]*32+i[None,:]);span=tl.maximum(tl.load(Top+i).to(tl.float32)-tl.load(Top+(P-1)*32+i).to(tl.float32),.001)
 cos=tl.load(Cos+a[:,None]*32+i[None,:]);cost=loss/span[None,:]+.001*(1-cos)
 cost=tl.where(a[:,None]!=i[None,:],cost,float('inf'));guide=i
 for _ in range(8):
  rowmin=tl.min(cost,1);aa=tl.argmin(rowmin,0);column=tl.min(tl.where(a[:,None]==aa,cost,float('inf')),0);ii=tl.argmin(column,0);guide=tl.where(i==ii,aa,guide);cost=tl.where((a[:,None]==aa)|(a[:,None]==ii)|(i[None,:]==aa)|(i[None,:]==ii),float('inf'),cost);tl.store(Trace+_*2,aa);tl.store(Trace+_*2+1,ii)
 rep=guide==i;rank=tl.cumsum(rep.to(tl.int32))-1;tl.store(Anchors+rank,i,rep)
 group=tl.sum(((a[:,None]<guide[None,:])&rep[:,None]).to(tl.int32),0);tl.store(Groups+i,group)
 tl.store(Widths+i,8);tl.store(Offsets+i,i*8)

@triton.jit
def mixed_plan(Top,Anchors,Groups,Widths,Offsets,Thresholds,Table,P:tl.constexpr):
 q=tl.arange(0,32);r=tl.arange(0,32);hi=tl.load(Top+q).to(tl.float32);lo=tl.load(Top+(P-1)*32+q).to(tl.float32);span=hi-lo
 rank=tl.sum(((span[:,None]<span[None,:])|((span[:,None]==span[None,:])&(q[:,None]>r[None,:]))).to(tl.int32),1)
 width=tl.where(rank<16,8,4);offset=tl.cumsum(width)-width;tl.store(Widths+q,width);tl.store(Offsets+q,offset);tl.store(Anchors+q,q);tl.store(Groups+q,q)
 tl.store(Table+q*9,hi)
 for lev in tl.static_range(8):
  frac=tl.maximum(1-(lev+1)/width.to(tl.float32),0);threshold=lo+(hi-lo)*frac*frac
  tl.store(Thresholds+lev*32+q,threshold);tl.store(Table+q*9+lev+1,threshold)

@triton.jit
def packed_encode(Lut,Thresholds,Anchors,Widths,Offsets,CellBits,Flags,C:tl.constexpr,G:tl.constexpr,B:tl.constexpr):
 c=tl.program_id(0)*B+tl.arange(0,B);g=tl.arange(0,32);anchor=tl.load(Anchors+g,g<G,0);width=tl.load(Widths+g,g<G,8);offset=tl.load(Offsets+g,g<G,0);word=offset//64;shift=offset%64
 value=tl.load(Lut+c[:,None]*32+anchor[None,:],(c[:,None]<C)&(g[None,:]<G),0).to(tl.float32);bits=tl.full((B,32),0,tl.uint64)
 for lev in tl.static_range(8):
  th=tl.load(Thresholds+lev*G+g,g<G,float('inf'));bits|=tl.where((value>=th[None,:])&(g[None,:]<G)&(lev<width[None,:]),1<<lev,0).to(tl.uint64)
 for w in tl.static_range(3):
  lower=tl.where(word[None,:]==w,bits<<shift[None,:],0);upper=tl.where((word[None,:]+1==w)&(shift[None,:]+width[None,:]>64),bits>>(64-shift[None,:]),0)
  packed=tl.sum(lower|upper,1);tl.store(CellBits+c*3+w,packed,c<C)
 tl.store(Flags+c,tl.sum((bits!=0).to(tl.int32),1)>0,c<C)

@triton.jit
def scatter24(CellBits,TileCells,TileOffsets,TileCount,Starts,IVF,Flags,DocBits,B:tl.constexpr):
 nc=tl.load(TileCount);t=tl.arange(0,B);w=tl.arange(0,4)
 for j in range(tl.program_id(0),nc,tl.num_programs(0)):
  cell=tl.load(TileCells+j);lo=tl.load(TileOffsets+j);hi=tl.load(Starts+cell+1);bits=tl.load(CellBits+cell*3+w,w<3,0);doc=tl.load(IVF+lo+t,lo+t<hi,0);valid=tl.load(Flags+doc,lo+t<hi,0)!=0
  tl.atomic_or(DocBits+doc[:,None]*3+w[None,:],bits[None,:],valid[:,None]&(w[None,:]<3)&(bits[None,:]!=0),sem='relaxed')

@triton.jit
def packed_decode(DocBits,Widths,Offsets,Table,Upper,N:tl.constexpr,G:tl.constexpr,B:tl.constexpr):
 d=tl.program_id(0)*B+tl.arange(0,B);g=tl.arange(0,32);width=tl.load(Widths+g,g<G,8);offset=tl.load(Offsets+g,g<G,0);word=offset//64;shift=offset%64
 low=tl.load(DocBits+d[:,None]*3+word[None,:],(d[:,None]<N)&(g[None,:]<G),0);high=tl.load(DocBits+d[:,None]*3+word[None,:]+1,(d[:,None]<N)&(g[None,:]<G)&(shift[None,:]+width[None,:]>64),0)
 bits=(((low>>shift[None,:])|(high<<(64-shift[None,:])))&((1<<width[None,:])-1)).to(tl.uint32);before=(bits&(0-bits))-1
 b=tl.inline_asm_elementwise('popc.b32 $0, $1;',constraints='=r,r',args=[before],dtype=tl.int32,is_pure=True,pack=1);b=tl.minimum(b,width[None,:]);v=tl.load(Table+g[None,:]*9+b,g[None,:]<G,0)
 tl.store(Upper+d,tl.sum(v,1)+4e-4,d<N)

class Hybrid24(DiagnosticFrontier):
 def __init__(self,*args,merge=True,**kwargs):
  super().__init__(*args,groups=32,score_power=2.,kind='independent',**kwargs);self.trace=torch.empty(16,device=self.cent.device,dtype=torch.int32);self.merge=merge;self.groups=24 if merge else 32;self.words=3
  self.cbits=torch.empty((self.c,3),device=self.cent.device,dtype=torch.uint64);self.dbits=torch.empty((self.n,3),device=self.cent.device,dtype=torch.uint64);self.widths=torch.empty(32,device=self.cent.device,dtype=torch.int32);self.offsets=torch.empty_like(self.widths)
  self.thresholds=torch.empty((8,self.groups),device=self.cent.device);self.table=torch.zeros((32,9),device=self.cent.device)
 def __call__(self,q,probe,budget,override=None,instrument=False):
  self.events=[];self.stamp('start',instrument);n=self.n;c=self.c;G=self.groups
  lut=(self.cent@q.T).contiguous();top=lut.topk(self.prefix,dim=0);self.stamp('lut_top',instrument)
  if self.merge:
   cos=q.float()@q.float().T;tail_losses[(32,)](lut,top.indices,top.values,self.loss,P=self.prefix,num_warps=8)
   select_pairs[(1,)](self.loss,top.values,cos,self.anchors,self.assign,self.widths,self.offsets,self.trace,P=self.prefix,num_warps=1)
   score_thresholds[(1,)](top.values,self.anchors,self.fractions,self.thresholds,G=G,L=8,P=self.prefix)
   conditional_group_table[(G,9)](lut,top.indices,top.values,self.thresholds,self.anchors,self.assign,self.table,G=G,L=8,P=self.prefix,num_warps=8)
  else:mixed_plan[(1,)](top.values,self.anchors,self.assign,self.widths,self.offsets,self.thresholds,self.table,P=self.prefix)
  self.stamp('selection_table',instrument)
  packed_encode[(triton.cdiv(c,32),)](lut,self.thresholds,self.anchors,self.widths,self.offsets,self.cbits,self.cf,C=c,G=G,B=32)
  self.cn.zero_();compact_flags[(triton.cdiv(c,256),)](self.cf,self.ci,self.cn,N=c,B=256);self.dbits.zero_();self.tn.zero_()
  make_tiles[(2048,)](self.ci,self.cn,self.starts,self.tc,self.to,self.tn,T=128,B=128);self.stamp('encode_tiles',instrument)
  scatter24[(2048,)](self.cbits,self.tc,self.to,self.tn,self.starts,self.ivf,self.flags,self.dbits,B=128);self.stamp('scatter',instrument)
  packed_decode[(triton.cdiv(n,32),)](self.dbits,self.widths,self.offsets,self.table,self.upper,N=n,G=G,B=32);self.stamp('decode',instrument)
  seedsize=budget*2;seed=self.upper.topk(seedsize,sorted=False).indices;seedids=self.ids[seed].contiguous();self.stamp('seed_select',instrument)
  centroid_lookup[(self.programs,)](lut,self.codes,self.off,seedids,self.seed_counts[budget],self.ss,B=self.block,num_warps=4,num_stages=1);self.stamp('seed_score',instrument)
  self.cut=self.ss[:seedsize].topk(budget,sorted=False).values.min();self.already.zero_();self.scores.fill_(-float('inf'))
  seed_map[(triton.cdiv(seedsize,256),)](seed,self.ss,self.already,self.scores,K=seedsize,B=256);self.wn.zero_()
  compact_survivors[(triton.cdiv(n,256),)](self.upper,self.cut,self.already,self.ids,self.count,self.wi,self.wp,self.wn,self.survivors,B=256);self.stamp('threshold_compact',instrument)
  centroid_lookup[(self.programs,)](lut,self.codes,self.off,self.wi,self.wn,self.ws,B=self.block,num_warps=4,num_stages=1);self.stamp('survivor_score',instrument)
  scatter_scores[(triton.cdiv(n,256),)](self.wp,self.wn,self.ws,self.scores,B=256);v,j=self.scores.topk(budget);self.stamp('final_select',instrument)
  return j,torch.isfinite(v)
