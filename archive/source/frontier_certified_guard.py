"""Conservative FP32 accumulation allowance under an explicit bounded-LUT contract.
For 32 query terms, 16 groups, finite LUT/table terms with magnitude <=2,
no overflow, round-to-nearest FP32 sums (any association), 4e-4 covers the
worst-case error bound recorded in evidence-v4-float-contract.json. This is a
conditional coarse-score guarantee, not a guarantee for raw-token global TopK.
"""
import triton
import triton.language as tl
from frontier_densefinish_engine import DenseRankedFrontier

@triton.jit
def sum_certified_anchor_bits(DocBits,Table,Upper,N:tl.constexpr,G:tl.constexpr,L:tl.constexpr,W:tl.constexpr,B:tl.constexpr):
    d=tl.program_id(0)*B+tl.arange(0,B);g=tl.arange(0,G)
    packed=tl.load(DocBits+d[:,None]*W+(g[None,:]//(64//L)),d[:,None]<N,0)
    bits=((packed>>((g[None,:]%(64//L))*L))&((1<<L)-1)).to(tl.uint32)
    before=(bits&(0-bits))-1
    first=tl.inline_asm_elementwise('popc.b32 $0, $1;',constraints='=r,r',args=[before],dtype=tl.int32,is_pure=True,pack=1)
    first=tl.minimum(first,L);v=tl.load(Table+g[None,:]*(L+1)+first)
    tl.store(Upper+d,tl.sum(v,1)+4e-4,d<N)

class GuardedDenseRankedFrontier(DenseRankedFrontier):
    def decode(self,lut,top,th):
        if self.groups!=16:raise ValueError('The documented numerical contract requires 16 groups')
        sum_certified_anchor_bits[(triton.cdiv(self.n,32),)](self.dbits,self.table,self.upper,N=self.n,G=self.groups,L=self.levels,W=self.words,B=32)
