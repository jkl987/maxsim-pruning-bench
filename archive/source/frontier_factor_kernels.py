"""Factor controls: same finite-codebook bound target, alternative access patterns."""
import triton
import triton.language as tl

@triton.jit
def float_flags(Lut,Tail,Flags,C:tl.constexpr,B:tl.constexpr):
 c=tl.program_id(0)*B+tl.arange(0,B);q=tl.arange(0,32)
 v=tl.load(Lut+c[:,None]*32+q[None,:],c[:,None]<C,0).to(tl.float32)
 t=tl.load(Tail+q);tl.store(Flags+c,tl.sum((v>=t[None,:]).to(tl.int32),1)>0,c<C)

@triton.jit
def fill_aos(Matrix,Tail,N:tl.constexpr,B:tl.constexpr):
 d=tl.program_id(0)*B+tl.arange(0,B);q=tl.arange(0,32)
 t=tl.load(Tail+q);tl.store(Matrix+d[:,None]*32+q[None,:],t[None,:],d[:,None]<N)

@triton.jit
def scatter_float_union(Lut,Tail,TileCells,TileOffsets,TileCount,Starts,IVF,Matrix,N:tl.constexpr,AOS:tl.constexpr,B:tl.constexpr):
 nc=tl.load(TileCount);t=tl.arange(0,B);q=tl.arange(0,32);tail=tl.load(Tail+q)
 for j in range(tl.program_id(0),nc,tl.num_programs(0)):
  cell=tl.load(TileCells+j);lo=tl.load(TileOffsets+j);hi=tl.minimum(tl.load(Starts+cell+1),lo+128)
  v=tl.load(Lut+cell*32+q).to(tl.float32)
  for sub in range(0,128,B):
   at=lo+sub+t;doc=tl.load(IVF+at,at<hi,0)
   addr=doc[:,None]*32+q[None,:] if AOS else q[None,:]*N+doc[:,None]
   tl.atomic_max(Matrix+addr,v[None,:],(at[:,None]<hi)&(v[None,:]>tail[None,:]),sem='relaxed')

@triton.jit
def sum_aos(Matrix,Upper,N:tl.constexpr,B:tl.constexpr):
 d=tl.program_id(0)*B+tl.arange(0,B);q=tl.arange(0,32)
 v=tl.load(Matrix+d[:,None]*32+q[None,:],d[:,None]<N,0)
 tl.store(Upper+d,tl.sum(v,1)+.0001,d<N)

@triton.jit
def _or(a,b):return a|b

@triton.jit
def forward_bits(CellBits,Codes,Off,DocBits,N:tl.constexpr,W:tl.constexpr,B:tl.constexpr):
 t=tl.arange(0,B);w=tl.arange(0,W)
 for doc in range(tl.program_id(0),N,tl.num_programs(0)):
  lo=tl.load(Off+doc).to(tl.int64);hi=tl.load(Off+doc+1).to(tl.int64)
  acc=tl.full((W,),0,tl.uint64)
  for st in range(lo,hi,B):
   at=st+t;c=tl.load(Codes+at,at<hi,0)
   v=tl.load(CellBits+c[:,None]*W+w[None,:],at[:,None]<hi,0)
   acc|=tl.reduce(v,0,_or)
  tl.store(DocBits+doc*W+w,acc)
