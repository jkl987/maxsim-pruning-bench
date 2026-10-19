"""GPU implementation of public PLAID IVF + centroid-MaxSim candidate selection.
Algorithm follows lightonai/fast-plaid, not a novel retrieval algorithm.
"""
import triton,triton.language as tl
@triton.jit
def mark_ivf(Cells,Starts,IVF,Flags,P:tl.constexpr,B:tl.constexpr):
 cell=tl.load(Cells+tl.program_id(0));lo=tl.load(Starts+cell);hi=tl.load(Starts+cell+1);t=tl.arange(0,B)
 for st in range(lo+tl.program_id(1)*B,hi,tl.num_programs(1)*B):
  ids=tl.load(IVF+st+t,st+t<hi,0);tl.atomic_or(Flags+ids,1,st+t<hi,sem='relaxed')
@triton.jit
def compact_flags(Flags,IDs,Count,N:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);ok=tl.load(Flags+i,i<N,0)!=0;rank=tl.cumsum(ok.to(tl.int32));num=tl.sum(ok.to(tl.int32),0);base=tl.atomic_add(Count,num,sem='relaxed');tl.store(IDs+base+rank-1,i,ok)
@triton.jit
def centroid_lookup(Lut,Codes,Off,IDs,Count,Score,B:tl.constexpr):
 nc=tl.load(Count);r=tl.arange(0,32);t=tl.arange(0,B)
 for j in range(tl.program_id(0),nc,tl.num_programs(0)):
  doc=tl.load(IDs+j);lo=tl.load(Off+doc).to(tl.int64);hi=tl.load(Off+doc+1).to(tl.int64);mx=tl.full((32,),float('-inf'),tl.float32)
  for st in range(lo,hi,B):
   tok=st+t;code=tl.load(Codes+tok,tok<hi,0);v=tl.load(Lut+code[None,:]*32+r[:,None],tok[None,:]<hi,float('-inf')).to(tl.float32);mx=tl.maximum(mx,tl.max(v,1))
  tl.store(Score+j,tl.sum(mx,0))
@triton.jit
def corrected_centroid_lookup(Lut,Codes,Off,IDs,Count,Score,Side,QC,RANK:tl.constexpr,STRIDE:tl.constexpr,B:tl.constexpr,CONSTANT:tl.constexpr):
 nc=tl.load(Count);r=tl.arange(0,32);t=tl.arange(0,B)
 for j in range(tl.program_id(0),nc,tl.num_programs(0)):
  doc=tl.load(IDs+j);lo=tl.load(Off+doc).to(tl.int64);hi=tl.load(Off+doc+1).to(tl.int64);mx=tl.full((32,),float('-inf'),tl.float32)
  for st in range(lo,hi,B):
   tok=st+t;code=tl.load(Codes+tok,tok<hi,0);v=tl.load(Lut+code[None,:]*32+r[:,None],tok[None,:]<hi,float('-inf')).to(tl.float32)
   for f in tl.static_range(RANK):
    side=tl.load(Side+tok*STRIDE+f,tok<hi,0).to(tl.float32);qc=tl.load(QC+r*STRIDE+f);v+=qc[:,None]*side[None,:]
   mx=tl.maximum(mx,tl.max(v,1))
  tl.store(Score+j,tl.sum(mx,0))
@triton.jit
def calibrated_centroid_lookup(Lut,Codes,Off,IDs,Count,Score,Side,QC,Meta,RANK:tl.constexpr,MODE:tl.constexpr,B:tl.constexpr):
 nc=tl.load(Count);r=tl.arange(0,32);t=tl.arange(0,B)
 for j in range(tl.program_id(0),nc,tl.num_programs(0)):
  doc=tl.load(IDs+j);lo=tl.load(Off+doc).to(tl.int64);hi=tl.load(Off+doc+1).to(tl.int64);mx=tl.full((32,),float('-inf'),tl.float32)
  for st in range(lo,hi,B):
   tok=st+t;code=tl.load(Codes+tok,tok<hi,0);v=tl.load(Lut+code[None,:]*32+r[:,None],tok[None,:]<hi,float('-inf')).to(tl.float32)
   if MODE==1:v*=tl.load(Meta+tok*8,tok<hi,0)[None,:]
   elif MODE==2:
    v*=tl.load(Meta+tok*8+1,tok<hi,0)[None,:];aa=tl.load(Meta+tok*8+2,tok<hi,0);qa=tl.load(QC+r*16);v+=qa[:,None]*aa[None,:]
   else:
    for f in tl.static_range(RANK):
     side=tl.load(Side+tok*16+f,tok<hi,0).to(tl.float32);qc=tl.load(QC+r*16+f);v+=qc[:,None]*side[None,:]
    if MODE>=3:v*=tl.load(Meta+tok*8+MODE,tok<hi,0)[None,:]
   v=tl.where(tok[None,:]<hi,v,float('-inf'));mx=tl.maximum(mx,tl.max(v,1))
  tl.store(Score+j,tl.sum(mx,0))
