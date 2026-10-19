"""Finite-codebook prefix upper-bound screening. Related to XTR/NRA; no novelty claim."""
import triton,triton.language as tl
@triton.jit
def fill_prefix(M,Tail,N:tl.constexpr,B:tl.constexpr):
 q=tl.program_id(0);i=tl.program_id(1)*B+tl.arange(0,B);tl.store(M+q*N+i,tl.load(Tail+q),i<N)
@triton.jit
def scatter_prefix(Cells,Values,Starts,IVF,Flags,M,N:tl.constexpr,P:tl.constexpr,B:tl.constexpr):
 q=tl.program_id(0);j=tl.program_id(1);idx=q*P+j;cell=tl.load(Cells+idx);value=tl.load(Values+idx).to(tl.float32);lo=tl.load(Starts+cell);hi=tl.load(Starts+cell+1);t=tl.arange(0,B)
 for st in range(lo+tl.program_id(2)*B,hi,tl.num_programs(2)*B):
  doc=tl.load(IVF+st+t,st+t<hi,0);ok=tl.load(Flags+doc,st+t<hi,0)!=0;tl.atomic_max(M+q*N+doc,value,ok,sem='relaxed')
@triton.jit
def sum_prefix(M,IDs,Count,Upper,N:tl.constexpr,B:tl.constexpr):
 j=tl.program_id(0)*B+tl.arange(0,B);nc=tl.load(Count);d=tl.load(IDs+j,j<nc,0);q=tl.arange(0,32);v=tl.load(M+q[:,None]*N+d[None,:],j[None,:]<nc,0);tl.store(Upper+j,tl.sum(v,0)+.0001,j<nc)
@triton.jit
def seed_map(Seeds,SeedScore,Already,Score,K:tl.constexpr,B:tl.constexpr):
 j=tl.program_id(0)*B+tl.arange(0,B);k=tl.load(Seeds+j,j<K,0);s=tl.load(SeedScore+j,j<K,0);tl.store(Already+k,1,j<K);tl.store(Score+k,s,j<K)
@triton.jit
def gated_lookup(Lut,Codes,Off,IDs,Count,Upper,Cut,Already,Score,Survivors,B:tl.constexpr):
 nc=tl.load(Count);cut=tl.load(Cut);r=tl.arange(0,32);t=tl.arange(0,B)
 for j in range(tl.program_id(0),nc,tl.num_programs(0)):
  seen=tl.load(Already+j);live=tl.load(Upper+j)>=cut
  if live and seen==0:
   doc=tl.load(IDs+j);lo=tl.load(Off+doc).to(tl.int64);hi=tl.load(Off+doc+1).to(tl.int64);mx=tl.full((32,),float('-inf'),tl.float32)
   for st in range(lo,hi,B):
    tok=st+t;code=tl.load(Codes+tok,tok<hi,0);v=tl.load(Lut+code[None,:]*32+r[:,None],tok[None,:]<hi,float('-inf')).to(tl.float32);mx=tl.maximum(mx,tl.max(v,1))
   tl.store(Score+j,tl.sum(mx))
  tl.store(Survivors+j,(live or seen!=0).to(tl.int32))
@triton.jit
def compact_survivors(Upper,Cut,Already,IDs,Count,WorkIDs,WorkPos,WorkCount,Survivors,B:tl.constexpr):
 j=tl.program_id(0)*B+tl.arange(0,B);nc=tl.load(Count);v=tl.load(Upper+j,j<nc,float('-inf'));seen=tl.load(Already+j,j<nc,0);live=(j<nc)&(v>=tl.load(Cut))&(seen==0);rank=tl.cumsum(live.to(tl.int32))-1;nt=tl.sum(live.to(tl.int32));base=tl.atomic_add(WorkCount,nt,sem='relaxed');d=tl.load(IDs+j,j<nc,0);tl.store(WorkIDs+base+rank,d,live);tl.store(WorkPos+base+rank,j,live);tl.store(Survivors+j,live.to(tl.int32)+(seen!=0).to(tl.int32),j<nc)
@triton.jit
def scatter_scores(WorkPos,WorkCount,WorkScore,Score,B:tl.constexpr):
 j=tl.program_id(0)*B+tl.arange(0,B);nc=tl.load(WorkCount);pos=tl.load(WorkPos+j,j<nc,0);v=tl.load(WorkScore+j,j<nc,0);tl.store(Score+pos,v,j<nc)
@triton.jit
def prefix_tiles(Cells,Starts,TileCells,TileOffsets,TileCount,N:tl.constexpr,T:tl.constexpr,B:tl.constexpr):
 t=tl.arange(0,B)
 for j in range(tl.program_id(0),N,tl.num_programs(0)):
  cell=tl.load(Cells+j);lo=tl.load(Starts+cell);hi=tl.load(Starts+cell+1);nt=tl.cdiv(hi-lo,T);dst=tl.atomic_add(TileCount,nt,sem='relaxed')
  for st in range(0,nt,B):
   at=st+t;tl.store(TileCells+dst+at,j,at<nt);tl.store(TileOffsets+dst+at,lo+at*T,at<nt)
@triton.jit
def prefix_scatter_tiles(Cells,Values,TileCells,TileOffsets,TileCount,Starts,IVF,Flags,M,N:tl.constexpr,P:tl.constexpr,B:tl.constexpr):
 nc=tl.load(TileCount);t=tl.arange(0,B)
 for j in range(tl.program_id(0),nc,tl.num_programs(0)):
  idx=tl.load(TileCells+j);cell=tl.load(Cells+idx);val=tl.load(Values+idx).to(tl.float32);lo=tl.load(TileOffsets+j);hi=tl.load(Starts+cell+1);doc=tl.load(IVF+lo+t,lo+t<hi,0);ok=tl.load(Flags+doc,lo+t<hi,0)!=0;tl.atomic_max(M+(idx//P)*N+doc,val,ok,sem='relaxed')
