"""Allow intermediate representative counts without changing safe bounds."""
import triton
import triton.language as tl
from frontier_rankedanchor_engine import RankedAnchorFrontier

@triton.jit
def sum_flexible_bits(DocBits,Table,Upper,N:tl.constexpr,G:tl.constexpr,L:tl.constexpr,W:tl.constexpr,B:tl.constexpr):
 d=tl.program_id(0)*B+tl.arange(0,B);g=tl.arange(0,triton.next_power_of_2(G))
 packed=tl.load(DocBits+d[:,None]*W+(g[None,:]//(64//L)),(d[:,None]<N)&(g[None,:]<G),0)
 bits=((packed>>((g[None,:]%(64//L))*L))&((1<<L)-1)).to(tl.uint32)
 before=(bits&(0-bits))-1;first=tl.inline_asm_elementwise('popc.b32 $0, $1;',constraints='=r,r',args=[before],dtype=tl.int32,is_pure=True,pack=1)
 first=tl.minimum(first,L);v=tl.load(Table+g[None,:]*(L+1)+first,g[None,:]<G,0)
 tl.store(Upper+d,tl.sum(v,1)+.0001,d<N)

class FlexibleAnchorFrontier(RankedAnchorFrontier):
 def decode(self,lut,top,th):
  sum_flexible_bits[(triton.cdiv(self.n,32),)](self.dbits,self.table,self.upper,N=self.n,G=self.groups,L=self.levels,W=self.words,B=32)
