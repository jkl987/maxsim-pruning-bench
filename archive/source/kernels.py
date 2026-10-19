import torch,triton,triton.language as tl
@triton.jit
def compressed(Q,C,Scale,Err,Center,Off,Coarse,Upper,D:tl.constexpr,BITS:tl.constexpr,BD:tl.constexpr,AFFINE:tl.constexpr=0):
 doc=tl.program_id(0);a=tl.load(Off+doc).to(tl.int64);b=tl.load(Off+doc+1).to(tl.int64)
 r=tl.arange(0,32);d=tl.arange(0,128);j=tl.arange(0,BD);pack=8//BITS
 q=tl.load(Q+r[:,None]*D+d[None,:],d[None,:]<D,0)
 qn=tl.sqrt(tl.sum(q.to(tl.float32)*q.to(tl.float32),1))
 center=tl.load(Center+(doc*D if AFFINE else 0)+d,d<D,0)
 if AFFINE==2:channelscale=tl.load(Scale+doc*D+d,d<D,0)
 mx=tl.full((32,),float('-inf'),tl.float32);up=mx
 for st in range(a,b,BD):
  t=st+j
  byte=tl.load(C+t[:,None]*(D//pack)+d[None,:]//pack,(t[:,None]<b)&(d[None,:]<D),0).to(tl.int32)
  code=(byte>>((d[None,:]%pack)*BITS))&((1<<BITS)-1)
  if AFFINE==2:
   x=((code.to(tl.float32)-((1<<BITS)-1)*0.5)*channelscale[None,:]+center[None,:]).to(tl.float16)
  else:
   scale=tl.load(Scale+t,t<b,0)
   x=((code.to(tl.float32)-((1<<BITS)-1)*0.5)*scale[:,None]+center[None,:]).to(tl.float16)
  x=tl.where(d[None,:]<D,x,0)
  v=tl.dot(q,tl.trans(x));e=tl.load(Err+t,t<b,0)
  mx=tl.maximum(mx,tl.max(tl.where(t[None,:]<b,v,float('-inf')),1))
  up=tl.maximum(up,tl.max(tl.where(t[None,:]<b,v+qn[:,None]*e[None,:],float('-inf')),1))
 tl.store(Coarse+doc,tl.sum(mx));tl.store(Upper+doc,tl.sum(up)+0.001)

@triton.jit
def exact(Q,X,Off,IDs,Upper,Cut,S,D:tl.constexpr,BD:tl.constexpr,MODE:tl.constexpr):
 i=tl.program_id(0);doc=i
 if MODE==1:doc=tl.load(IDs+i)
 keep=True
 if MODE==2:keep=tl.load(Upper+doc)>=tl.load(Cut)
 score=float('-inf')
 if keep:
  a=tl.load(Off+doc).to(tl.int64);b=tl.load(Off+doc+1).to(tl.int64);r=tl.arange(0,32);d=tl.arange(0,128);j=tl.arange(0,BD)
  q=tl.load(Q+r[:,None]*D+d[None,:],d[None,:]<D,0);mx=tl.full((32,),float('-inf'),tl.float32)
  for st in range(a,b,BD):
   t=st+j;x=tl.load(X+t[:,None]*D+d[None,:],(t[:,None]<b)&(d[None,:]<D),0)
   v=tl.dot(q,tl.trans(x));mx=tl.maximum(mx,tl.max(tl.where(t[None,:]<b,v,float('-inf')),1))
  score=tl.sum(mx)
 tl.store(S+i,score)

@triton.jit
def ball_bounds(Q,Centers,Radii,Norms,Coarse,Upper,D:tl.constexpr,K:tl.constexpr,BK:tl.constexpr):
 doc=tl.program_id(0);r=tl.arange(0,32);d=tl.arange(0,128);j=tl.arange(0,BK)
 q=tl.load(Q+r[:,None]*D+d[None,:],d[None,:]<D,0)
 c=tl.load(Centers+(doc*K+j[:,None])*D+d[None,:],(j[:,None]<K)&(d[None,:]<D),0)
 rad=tl.load(Radii+doc*K+j,j<K,0);norm=tl.load(Norms+doc*K+j,j<K,0)
 val=tl.dot(q,tl.trans(c));qn=tl.sqrt(tl.sum(q.to(tl.float32)*q.to(tl.float32),1))
 # Euclidean ball plus unit-norm bound. Radius includes fp16-center conversion.
 u=tl.minimum(val+qn[:,None]*rad[None,:],qn[:,None]*norm[None,:])
 tl.store(Coarse+doc,tl.sum(tl.max(tl.where(j[None,:]<K,val,float('-inf')),1)))
 tl.store(Upper+doc,tl.sum(tl.max(tl.where(j[None,:]<K,u,float('-inf')),1))+0.001)

@triton.jit
def grouped(Q,X,C,Scale,Err,Center,Off,IDs,Coarse,Upper,Cut,Bias,COUNT:tl.constexpr,D:tl.constexpr,BITS:tl.constexpr,L:tl.constexpr,G:tl.constexpr,FILTER:tl.constexpr=False,BIAS:tl.constexpr=False):
 g=tl.arange(0,G);i=tl.program_id(0)*G+g;docs=tl.load(IDs+i,i<COUNT,0)
 starts=tl.load(Off+docs).to(tl.int64);ends=tl.load(Off+docs+1).to(tl.int64);keep=i<COUNT
 if FILTER:
  keep=keep&(tl.load(Upper+docs)>=tl.load(Cut))
  tl.store(Coarse+docs,float('-inf'),i<COUNT)
 if tl.sum(keep.to(tl.int32),0)>0:
  r=tl.arange(0,32);d=tl.arange(0,128);j=tl.arange(0,G*L)
  # Repeat each start over its fixed L token slots.
  tt=tl.reshape(tl.broadcast_to(starts[:,None]+tl.arange(0,L)[None,:],(G,L)),(G*L,))
  valid=tl.reshape((starts[:,None]+tl.arange(0,L)[None,:]<ends[:,None])&keep[:,None],(G*L,))
  q=tl.load(Q+r[:,None]*D+d[None,:],d[None,:]<D,0)
  if BITS>0:
   pack=8//BITS;cc=tl.load(C+tt[:,None]*(D//pack)+d[None,:]//pack,valid[:,None]&(d[None,:]<D),0).to(tl.int32)
   cc=(cc>>((d[None,:]%pack)*BITS))&((1<<BITS)-1);sc=tl.load(Scale+tt,valid,0);cen=tl.load(Center+d,d<D,0)
   xx=((cc.to(tl.float32)-((1<<BITS)-1)*.5)*sc[:,None]+cen[None,:]).to(tl.float16);xx=tl.where(d[None,:]<D,xx,0)
  else:xx=tl.load(X+tt[:,None]*D+d[None,:],valid[:,None]&(d[None,:]<D),0)
  dot=tl.dot(q,tl.trans(xx))
  if BIAS:dot+=tl.load(Bias+tt,valid,0)[None,:]
  v=tl.reshape(tl.where(valid[None,:],dot,float('-inf')),(32,G,L));scor=tl.sum(tl.max(v,2),0)
  tl.store(Coarse+docs,scor,keep)
  if BITS>0:
   er=tl.load(Err+tt,valid,0);qn=tl.sqrt(tl.sum(q.to(tl.float32)*q.to(tl.float32),1));u=tl.reshape(tl.where(valid[None,:],dot+qn[:,None]*er[None,:],float('-inf')),(32,G,L));ub=tl.sum(tl.max(u,2),0)+.001
   tl.store(Upper+docs,ub,keep)

@triton.jit
def group_int(Q,QS,QN,C,Scale,Err,Bias,Off,IDs,Coarse,Upper,COUNT:tl.constexpr,D:tl.constexpr,BITS:tl.constexpr,L:tl.constexpr,G:tl.constexpr):
 g=tl.arange(0,G);i=tl.program_id(0)*G+g;docs=tl.load(IDs+i,i<COUNT,0);starts=tl.load(Off+docs).to(tl.int64);ends=tl.load(Off+docs+1).to(tl.int64)
 r=tl.arange(0,32);d=tl.arange(0,128);tt=tl.reshape(starts[:,None]+tl.arange(0,L)[None,:],(G*L,));valid=tl.reshape((starts[:,None]+tl.arange(0,L)[None,:]<ends[:,None])&(i[:,None]<COUNT),(G*L,))
 q=tl.load(Q+r[:,None]*D+d[None,:],d[None,:]<D,0);pack=8//BITS
 code=tl.load(C+tt[:,None]*(D//pack)+d[None,:]//pack,valid[:,None]&(d[None,:]<D),0).to(tl.int32)
 code=(code>>((d[None,:]%pack)*BITS))&((1<<BITS)-1)
 xx=tl.where(d[None,:]<D,code*2-((1<<BITS)-1),0).to(tl.int8)
 dot=tl.dot(q,tl.trans(xx)).to(tl.float32)
 scale=tl.load(Scale+tt,valid,0);qs=tl.load(QS+r);qn=tl.load(QN+r);bias=tl.load(Bias+tt,valid,0)
 val=dot*qs[:,None]*scale[None,:]*.5+bias[None,:]
 er=tl.load(Err+tt,valid,0)
 v=tl.reshape(tl.where(valid[None,:],val,float('-inf')),(32,G,L));up=tl.reshape(tl.where(valid[None,:],val+qn[:,None]*er[None,:],float('-inf')),(32,G,L))
 tl.store(Coarse+docs,tl.sum(tl.max(v,2),0),i<COUNT);tl.store(Upper+docs,tl.sum(tl.max(up,2),0)+.001,i<COUNT)

@triton.jit
def grouped_loop(Q,X,C,Scale,Err,Center,Off,IDs,Coarse,Upper,Cut,Bias,Weights,COUNT:tl.constexpr,D:tl.constexpr,BITS:tl.constexpr,L:tl.constexpr,G:tl.constexpr,QR:tl.constexpr=32,FILTER:tl.constexpr=False,BIAS:tl.constexpr=False,WEIGHTED:tl.constexpr=False):
 g=tl.arange(0,G);i=tl.program_id(0)*G+g;docs=tl.load(IDs+i,i<COUNT,0);starts=tl.load(Off+docs).to(tl.int64);ends=tl.load(Off+docs+1).to(tl.int64);keep=i<COUNT
 if FILTER:
  keep=keep&(tl.load(Upper+docs)>=tl.load(Cut))
  tl.store(Coarse+docs,float('-inf'),i<COUNT)
 if tl.sum(keep.to(tl.int32),0)>0:
  T:tl.constexpr=min(L,128//G)
  r=tl.arange(0,QR);d=tl.arange(0,128);q=tl.load(Q+r[:,None]*D+d[None,:],d[None,:]<D,0.0);qn=tl.sqrt(tl.sum(q.to(tl.float32)*q.to(tl.float32),1));wm=tl.full((QR,),1,tl.float32)
  if WEIGHTED:wm=tl.load(Weights+r)
  mx=tl.full((QR,G),float('-inf'),tl.float32);um=mx
  for st in range(0,L,T):
   tt=tl.reshape(starts[:,None]+st+tl.arange(0,T)[None,:],(G*T,));valid=tl.reshape((starts[:,None]+st+tl.arange(0,T)[None,:]<ends[:,None])&keep[:,None],(G*T,))
   if BITS>0:
    pack=8//BITS;cc=tl.load(C+tt[:,None]*(D//pack)+d[None,:]//pack,valid[:,None]&(d[None,:]<D),0).to(tl.int32);cc=(cc>>((d[None,:]%pack)*BITS))&((1<<BITS)-1);sc=tl.load(Scale+tt,valid,0);cen=tl.load(Center+d,d<D,0)
    xx=((cc.to(tl.float32)-((1<<BITS)-1)*.5)*sc[:,None]+cen[None,:]).to(tl.float16);xx=tl.where(d[None,:]<D,xx,0)
   else:xx=tl.load(X+tt[:,None]*D+d[None,:],valid[:,None]&(d[None,:]<D),0)
   dot=tl.dot(q,tl.trans(xx))
   if BIAS:dot+=wm[:,None]*tl.load(Bias+tt,valid,0)[None,:]
   v=tl.reshape(tl.where(valid[None,:],dot,float('-inf')),(QR,G,T));mx=tl.maximum(mx,tl.max(v,2))
   if BITS>0:
    er=tl.load(Err+tt,valid,0);u=tl.reshape(tl.where(valid[None,:],dot+qn[:,None]*er[None,:],float('-inf')),(QR,G,T));um=tl.maximum(um,tl.max(u,2))
  tl.store(Coarse+docs,tl.sum(mx,0),keep)
  if BITS>0:tl.store(Upper+docs,tl.sum(um,0)+.001,keep)

@triton.jit
def group_fp4(Q,C,Scale,Err,Bias,Off,IDs,Coarse,Upper,COUNT:tl.constexpr,D:tl.constexpr,L:tl.constexpr,G:tl.constexpr,FMT:tl.constexpr='fp16',CORRECT:tl.constexpr=False,Correction=None):
 g=tl.arange(0,G);i=tl.program_id(0)*G+g;docs=tl.load(IDs+i,i<COUNT,0);starts=tl.load(Off+docs).to(tl.int64);ends=tl.load(Off+docs+1).to(tl.int64)
 T:tl.constexpr=min(L,128//G)
 r=tl.arange(0,32);d=tl.arange(0,128);dk=tl.arange(0,64);sb=tl.arange(0,4)
 q=tl.load(Q+r[:,None]*D+d[None,:],d[None,:]<D,0.0);qn=tl.sqrt(tl.sum(q.to(tl.float32)*q.to(tl.float32),1));mx=tl.full((32,G),float('-inf'),tl.float32);um=mx
 for st in range(0,L,T):
  tt=tl.reshape(starts[:,None]+st+tl.arange(0,T)[None,:],(G*T,));valid=tl.reshape((starts[:,None]+st+tl.arange(0,T)[None,:]<ends[:,None])&(i[:,None]<COUNT),(G*T,))
  cc=tl.load(C+tt[:,None]*(D//2)+dk[None,:],valid[:,None]&(dk[None,:]<D//2),0)
  sc=tl.load(Scale+tt[:,None]*(D//32)+sb[None,:],valid[:,None]&(sb[None,:]<D//32),127)
  dot=tl.dot_scaled(q,None,FMT,tl.trans(cc),sc,'e2m1')+tl.load(Bias+tt,valid,0)[None,:];er=tl.load(Err+tt,valid,0)
  v=tl.reshape(tl.where(valid[None,:],dot,float('-inf')),(32,G,T));u=tl.reshape(tl.where(valid[None,:],dot+qn[:,None]*er[None,:],float('-inf')),(32,G,T));mx=tl.maximum(mx,tl.max(v,2));um=tl.maximum(um,tl.max(u,2))
 shift=0.;guard=.001
 if CORRECT:
  shift=tl.load(Correction);guard=tl.load(Correction+1)
 tl.store(Coarse+docs,tl.sum(mx,0)+shift,i<COUNT);tl.store(Upper+docs,tl.sum(um,0)+shift+guard,i<COUNT)

@triton.jit
def prepare_query(Q,A,B,QR,Correction,D:tl.constexpr,PROXY:tl.constexpr=False,QProxy=None):
 r=tl.arange(0,32);d=tl.arange(0,128)
 q=tl.load(Q+r[:,None]*D+d[None,:],d[None,:]<D,0).to(tl.float32);a=tl.load(A+d,d<D,0);b=tl.load(B+d,d<D,0);res=q-a[None,:];rounded=res.to(tl.float16)
 tl.store(QR+r[:,None]*D+d[None,:],rounded,d[None,:]<D)
 delta=res-rounded.to(tl.float32);guard=tl.sum(tl.sqrt(tl.sum(delta*delta,1)),0)*3+.003;shift=tl.sum(tl.sum(q*b[None,:],1),0)
 tl.store(Correction,shift);tl.store(Correction+1,guard)
 if PROXY:tl.store(QProxy+d,tl.sum(res,0)/32,d<D)

@triton.jit
def compact_survivors(Upper,Cut,IDs,Count,N:tl.constexpr,B:tl.constexpr=256):
 ix=tl.program_id(0)*B+tl.arange(0,B);keep=(ix<N)&(tl.load(Upper+ix,ix<N,float('-inf'))>=tl.load(Cut));rank=tl.cumsum(keep.to(tl.int32),0)-1;n=tl.sum(keep.to(tl.int32),0)
 if n>0:
  base=tl.atomic_add(Count,n);tl.store(IDs+base+rank,ix,keep)

@triton.jit
def persistent_refine(Q,X,Off,IDs,Count,S,D:tl.constexpr,BD:tl.constexpr):
 count=tl.load(Count);r=tl.arange(0,32);d=tl.arange(0,128);j=tl.arange(0,BD);q=tl.load(Q+r[:,None]*D+d[None,:],d[None,:]<D,0)
 for ii in range(tl.program_id(0),count,tl.num_programs(0)):
  doc=tl.load(IDs+ii);a=tl.load(Off+doc).to(tl.int64);b=tl.load(Off+doc+1).to(tl.int64);mx=tl.full((32,),float('-inf'),tl.float32)
  for st in range(a,b,BD):
   tt=st+j;x=tl.load(X+tt[:,None]*D+d[None,:],(tt[:,None]<b)&(d[None,:]<D),0);dot=tl.dot(q,tl.trans(x));mx=tl.maximum(mx,tl.max(tl.where(tt[None,:]<b,dot,float('-inf')),1))
  tl.store(S+doc,tl.sum(mx))

@triton.jit
def fused_fp4(Q,C,Scale,Err,Bias,Off,IDs,Coarse,Upper,X,QOriginal,Cut,Scores,COUNT:tl.constexpr,D:tl.constexpr,L:tl.constexpr,G:tl.constexpr,FMT:tl.constexpr='fp16',CORRECT:tl.constexpr=False,Correction=None):
 g=tl.arange(0,G);i=tl.program_id(0)*G+g;docs=tl.load(IDs+i,i<COUNT,0);starts=tl.load(Off+docs).to(tl.int64);ends=tl.load(Off+docs+1).to(tl.int64)
 T:tl.constexpr=min(L,128//G)
 r=tl.arange(0,32);d=tl.arange(0,128);dk=tl.arange(0,64);sb=tl.arange(0,4)
 q=tl.load(Q+r[:,None]*D+d[None,:],d[None,:]<D,0.0);qn=tl.sqrt(tl.sum(q.to(tl.float32)*q.to(tl.float32),1));mx=tl.full((32,G),float('-inf'),tl.float32);um=mx
 for st in range(0,L,T):
  tt=tl.reshape(starts[:,None]+st+tl.arange(0,T)[None,:],(G*T,));valid=tl.reshape((starts[:,None]+st+tl.arange(0,T)[None,:]<ends[:,None])&(i[:,None]<COUNT),(G*T,))
  cc=tl.load(C+tt[:,None]*(D//2)+dk[None,:],valid[:,None]&(dk[None,:]<D//2),0)
  sc=tl.load(Scale+tt[:,None]*(D//32)+sb[None,:],valid[:,None]&(sb[None,:]<D//32),127)
  dot=tl.dot_scaled(q,None,FMT,tl.trans(cc),sc,'e2m1')+tl.load(Bias+tt,valid,0)[None,:];er=tl.load(Err+tt,valid,0)
  v=tl.reshape(tl.where(valid[None,:],dot,float('-inf')),(32,G,T));u=tl.reshape(tl.where(valid[None,:],dot+qn[:,None]*er[None,:],float('-inf')),(32,G,T));mx=tl.maximum(mx,tl.max(v,2));um=tl.maximum(um,tl.max(u,2))
 shift=0.;guard=.001
 if CORRECT:
  shift=tl.load(Correction);guard=tl.load(Correction+1)
 tl.store(Coarse+docs,tl.sum(mx,0)+shift,i<COUNT);tl.store(Upper+docs,tl.sum(um,0)+shift+guard,i<COUNT)

 keep=(i<COUNT)&(tl.sum(um,0)+shift+guard>=tl.load(Cut))
 tl.store(Scores+docs,float('-inf'),i<COUNT)
 if tl.sum(keep.to(tl.int32),0)>0:
  qfull=tl.load(QOriginal+r[:,None]*D+d[None,:],d[None,:]<D,0)
  fm=tl.full((32,G),float('-inf'),tl.float32)
  for st in range(0,L,T):
   tt=tl.reshape(starts[:,None]+st+tl.arange(0,T)[None,:],(G*T,));valid=tl.reshape((starts[:,None]+st+tl.arange(0,T)[None,:]<ends[:,None])&keep[:,None],(G*T,))
   xx=tl.load(X+tt[:,None]*D+d[None,:],valid[:,None]&(d[None,:]<D),0)
   v=tl.dot(qfull,tl.trans(xx));v=tl.reshape(tl.where(valid[None,:],v,float('-inf')),(32,G,T));fm=tl.maximum(fm,tl.max(v,2))
  tl.store(Scores+docs,tl.sum(fm,0),keep)
