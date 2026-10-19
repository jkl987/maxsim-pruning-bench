"""Conservative score-bin membership unions. Unary OR aggregation; EMVB-related.
This implementation alone is not a claim of scientific novelty.
"""
import triton,triton.language as tl
@triton.jit
def encode_cells(Lut,Thresholds,CellBits,Flags,C:tl.constexpr,L:tl.constexpr,B:tl.constexpr):
 c=tl.program_id(0)*B+tl.arange(0,B);q=tl.arange(0,32);v=tl.load(Lut+c[:,None]*32+q[None,:],c[:,None]<C,0).to(tl.float32);bits=tl.full((B,32),0,tl.uint64)
 for lev in tl.static_range(L):
  threshold=tl.load(Thresholds+lev*32+q);bits|=tl.where(v>=threshold[None,:],1<<lev,0).to(tl.uint64)
 W:tl.constexpr=32*L//64;Q:tl.constexpr=64//L
 packed=tl.sum(tl.reshape(bits,(B,W,Q))<<(tl.arange(0,Q)[None,None,:]*L),2)
 tl.store(CellBits+c[:,None]*W+tl.arange(0,W)[None,:],packed,c[:,None]<C);tl.store(Flags+c,tl.sum((bits!=0).to(tl.int32),1)>0,c<C)
@triton.jit
def scatter_cells(CellBits,Cells,Count,Starts,IVF,Flags,DocBits,W:tl.constexpr,B:tl.constexpr):
 nc=tl.load(Count);t=tl.arange(0,B);w=tl.arange(0,W)
 for j in range(tl.program_id(0),nc,tl.num_programs(0)):
  cell=tl.load(Cells+j);bits=tl.load(CellBits+cell*W+w);lo=tl.load(Starts+cell);hi=tl.load(Starts+cell+1)
  for st in range(lo,hi,B):
   doc=tl.load(IVF+st+t,st+t<hi,0);valid=tl.load(Flags+doc,st+t<hi,0)!=0
   tl.atomic_or(DocBits+doc[:,None]*W+w[None,:],bits[None,:],valid[:,None]&(bits[None,:]!=0),sem='relaxed')
@triton.jit
def sum_bits(DocBits,IDs,Count,Thresholds,Upper,L:tl.constexpr,B:tl.constexpr):
 j=tl.program_id(0)*B+tl.arange(0,B);nc=tl.load(Count);d=tl.load(IDs+j,j<nc,0);q=tl.arange(0,32);W:tl.constexpr=32*L//64;Q:tl.constexpr=64//L
 packed=tl.load(DocBits+d[:,None]*W+(q[None,:]//Q),j[:,None]<nc,0)
 bits=((packed>>((q[None,:]%Q)*L))&((1<<L)-1)).to(tl.uint32)
 before=(bits & (0-bits))-1
 idx=tl.inline_asm_elementwise('popc.b32 $0, $1;',constraints='=r,r',args=[before],dtype=tl.int32,is_pure=True,pack=1)
 idx=tl.minimum(tl.maximum(idx-1,0),L-1)
 vals=tl.load(Thresholds+idx*32+q[None,:]);tl.store(Upper+j,tl.sum(vals,1)+.0001,j<nc)
@triton.jit
def make_tiles(Cells,Count,Starts,TileCells,TileOffsets,TileCount,T:tl.constexpr,B:tl.constexpr):
 nc=tl.load(Count);t=tl.arange(0,B)
 for j in range(tl.program_id(0),nc,tl.num_programs(0)):
  cell=tl.load(Cells+j);lo=tl.load(Starts+cell);hi=tl.load(Starts+cell+1);nt=tl.cdiv(hi-lo,T);dst=tl.atomic_add(TileCount,nt,sem='relaxed')
  for st in range(0,nt,B):
   at=st+t;tl.store(TileCells+dst+at,cell,at<nt);tl.store(TileOffsets+dst+at,lo+at*T,at<nt)
@triton.jit
def scatter_tiles(CellBits,TileCells,TileOffsets,TileCount,Starts,IVF,Flags,DocBits,W:tl.constexpr,B:tl.constexpr):
 nc=tl.load(TileCount);t=tl.arange(0,B);w=tl.arange(0,W)
 for j in range(tl.program_id(0),nc,tl.num_programs(0)):
  cell=tl.load(TileCells+j);lo=tl.load(TileOffsets+j);hi=tl.load(Starts+cell+1);bits=tl.load(CellBits+cell*W+w);doc=tl.load(IVF+lo+t,lo+t<hi,0);valid=tl.load(Flags+doc,lo+t<hi,0)!=0
  tl.atomic_or(DocBits+doc[:,None]*W+w[None,:],bits[None,:],valid[:,None]&(bits[None,:]!=0),sem='relaxed')
