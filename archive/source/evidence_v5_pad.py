import triton
import triton.language as tl
@triton.jit
def pad_codes(Codes,Off,Out,N:tl.constexpr,L:tl.constexpr,B:tl.constexpr):
 d=tl.program_id(0);t=tl.arange(0,B);lo=tl.load(Off+d);hi=tl.load(Off+d+1)
 for st in range(0,L,B):
  x=st+t;v=tl.load(Codes+lo+x,(x<L)&(lo+x<hi),0);tl.store(Out+d*L+x,v,x<L)
