"""Query anchor summaries with finite-codebook conditional bounds.
Not token deletion: every original query contribution remains upper bounded.
"""
import triton
import triton.language as tl

@triton.jit
def choose_anchors(Cos,Anchors,Groups,G:tl.constexpr):
 q=tl.arange(0,32);r=tl.arange(0,32);cs=tl.load(Cos+q[:,None]*32+r[None,:])
 mean=tl.sum(cs,1);a=tl.argmin(mean,0);near=tl.full((32,),float('-inf'),tl.float32)
 for g in tl.static_range(G):
  tl.store(Anchors+g,a);v=tl.load(Cos+a*32+q);near=tl.maximum(near,v);a=tl.argmin(near,0)
 # Scalar stores and vector loads can be owned by different warps. Ensure the
 # current representatives are visible before assigning terms to them.
 tl.debug_barrier()
 gs=tl.arange(0,G);ai=tl.load(Anchors+gs);similarity=tl.load(Cos+q[:,None]*32+ai[None,:]);assignment=tl.argmax(similarity,1);tl.store(Groups+q,assignment)

@triton.jit
def anchor_encode(Lut,Thresholds,Anchors,CellBits,Flags,C:tl.constexpr,G:tl.constexpr,L:tl.constexpr,W:tl.constexpr,B:tl.constexpr):
 c=tl.program_id(0)*B+tl.arange(0,B);g=tl.arange(0,W*(64//L));ai=tl.load(Anchors+g,g<G,0)
 v=tl.load(Lut+c[:,None]*32+ai[None,:],(c[:,None]<C)&(g[None,:]<G),0).to(tl.float32)
 bits=tl.full((B,W*(64//L)),0,tl.uint64)
 for lev in tl.static_range(L):
  th=tl.load(Thresholds+lev*G+g,g<G,float('inf'))
  bits|=tl.where((v>=th[None,:])&(g[None,:]<G),1<<lev,0).to(tl.uint64)
 packed=tl.sum(tl.reshape(bits,(B,W,64//L))<<(tl.arange(0,64//L)[None,None,:]*L),2)
 tl.store(CellBits+c[:,None]*W+tl.arange(0,W)[None,:],packed,c[:,None]<C)
 tl.store(Flags+c,tl.sum((bits!=0).to(tl.int32),1)>0,c<C)

@triton.jit
def conditional_group_table(Lut,TopIds,TopValues,Thresholds,Anchors,Groups,Table,G:tl.constexpr,L:tl.constexpr,P:tl.constexpr):
 g=tl.program_id(0);b=tl.program_id(1);q=tl.arange(0,32);t=tl.arange(0,P)
 group=tl.load(Groups+q);active=group==g;anchor=tl.load(Anchors+g)
 cap=tl.load(Thresholds+tl.maximum(b-1,0)*G+g)
 c=tl.load(TopIds+t[:,None]*32+q[None,:],active[None,:],0)
 value=tl.load(TopValues+t[:,None]*32+q[None,:],active[None,:],float('-inf')).to(tl.float32)
 av=tl.load(Lut+c*32+anchor,active[None,:],float('inf')).to(tl.float32)
 compatible=(av<=cap) if b==0 else (av<cap)
 candidate=tl.max(tl.where(compatible,value,float('-inf')),0)
 tail=tl.load(TopValues+(P-1)*32+q).to(tl.float32)
 safe=tl.maximum(candidate,tail)
 # The anchor's own contribution is directly bounded by its bin endpoint.
 safe=tl.where(q==anchor,cap,safe)
 tl.store(Table+g*(L+1)+b,tl.sum(tl.where(active,safe,0)))

@triton.jit
def sum_anchor_bits(DocBits,Table,Upper,N:tl.constexpr,G:tl.constexpr,L:tl.constexpr,W:tl.constexpr,B:tl.constexpr):
 d=tl.program_id(0)*B+tl.arange(0,B);g=tl.arange(0,G)
 packed=tl.load(DocBits+d[:,None]*W+(g[None,:]//(64//L)),d[:,None]<N,0)
 bits=((packed>>((g[None,:]%(64//L))*L))&((1<<L)-1)).to(tl.uint32)
 before=(bits&(0-bits))-1
 first=tl.inline_asm_elementwise('popc.b32 $0, $1;',constraints='=r,r',args=[before],dtype=tl.int32,is_pure=True,pack=1)
 first=tl.minimum(first,L);v=tl.load(Table+g[None,:]*(L+1)+first)
 tl.store(Upper+d,tl.sum(v,1)+.0001,d<N)
