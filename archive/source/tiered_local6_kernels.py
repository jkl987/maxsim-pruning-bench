"""Quantized screening and token-level conservative refinement experiments.
These kernels alone are not a novelty claim. Certification is restricted to
the input candidate set; PLAID candidate generation remains approximate.
"""
import triton
import triton.language as tl

@triton.jit
def quant_bounds(Q, Codes, Scale, Err, Off, IDs, Coarse, Lower, Upper, RowLower,
                 Center,Bias,QP,QR,D:tl.constexpr, B:tl.constexpr, GUARD:tl.constexpr,CENTERED:tl.constexpr=False,CodeIDs=None):
    i=tl.program_id(0);doc=tl.load(IDs+i)
    lo=tl.load(Off+doc).to(tl.int64);hi=tl.load(Off+doc+1).to(tl.int64)
    r=tl.arange(0,32);d=tl.arange(0,128);j=tl.arange(0,B)
    q=tl.load(Q+r[:,None]*D+d[None,:],d[None,:]<D,0)
    qn=tl.sqrt(tl.sum(q.to(tl.float32)*q.to(tl.float32),1))
    if CENTERED:
        qp=tl.load(QP+r);qn=tl.load(QR+r)
    mx=tl.full((32,),float('-inf'),tl.float32);lm=mx;um=mx
    for st in range(lo,hi,B):
        t=st+j;valid=t<hi
        byte=tl.load(Codes+t[:,None]*(D*3//4)+(d[None,:]*6//8),valid[:,None]&(d[None,:]<D),0).to(tl.int32)
        shift=(d%4)*6%8
        tail=tl.load(Codes+t[:,None]*(D*3//4)+(d[None,:]*6//8)+1,valid[:,None]&(d[None,:]<D)&(shift[None,:]>2),0).to(tl.int32)
        c=(((byte|(tail<<8))>>shift[None,:])&63).to(tl.float32)-32
        cid=tl.load(CodeIDs+t,valid,0).to(tl.int64)
        center=tl.load(Center+cid[:,None]*D+d[None,:],valid[:,None]&(d[None,:]<D),0).to(tl.float32)
        s=tl.load(Scale+t,valid,0).to(tl.float32)
        if CENTERED:x=tl.where(d[None,:]<D,c*s[:,None]+center,0).to(tl.float16)
        else:x=(c*s[:,None]).to(tl.float16)
        v=tl.dot(q,tl.trans(x));e=tl.load(Err+t,valid,0).to(tl.float32)
        if CENTERED:v+=qp[:,None]*tl.load(Bias+t,valid,0)[None,:]
        radius=qn[:,None]*e[None,:]+GUARD
        mx=tl.maximum(mx,tl.max(tl.where(valid[None,:],v,float('-inf')),1))
        lm=tl.maximum(lm,tl.max(tl.where(valid[None,:],v-radius,float('-inf')),1))
        um=tl.maximum(um,tl.max(tl.where(valid[None,:],v+radius,float('-inf')),1))
    tl.store(Coarse+i,tl.sum(mx));tl.store(Lower+i,tl.sum(lm)-0.0001)
    tl.store(Upper+i,tl.sum(um)+0.0001);tl.store(RowLower+i*32+r,lm)

@triton.jit
def possible_tokens(Q,Codes,Scale,Err,Off,IDs,Selected,RowLower,TokenIDs,Counts,Center,Bias,QP,QR,
                    D:tl.constexpr,B:tl.constexpr,MAXL:tl.constexpr,
                    GUARD:tl.constexpr,FULLDOC:tl.constexpr,CENTERED:tl.constexpr=False,CodeIDs=None):
    i=tl.program_id(0);ci=tl.load(Selected+i);doc=tl.load(IDs+ci)
    lo=tl.load(Off+doc).to(tl.int64);hi=tl.load(Off+doc+1).to(tl.int64)
    r=tl.arange(0,32);d=tl.arange(0,128);j=tl.arange(0,B)
    q=tl.load(Q+r[:,None]*D+d[None,:],d[None,:]<D,0)
    qn=tl.sqrt(tl.sum(q.to(tl.float32)*q.to(tl.float32),1))
    if CENTERED:
        qp=tl.load(QP+r);qn=tl.load(QR+r)
    lm=tl.load(RowLower+ci*32+r);count=0
    for st in range(lo,hi,B):
        t=st+j;valid=t<hi
        if FULLDOC:keep=valid
        else:
            byte=tl.load(Codes+t[:,None]*(D*3//4)+(d[None,:]*6//8),valid[:,None]&(d[None,:]<D),0).to(tl.int32)
            shift=(d%4)*6%8
            tail=tl.load(Codes+t[:,None]*(D*3//4)+(d[None,:]*6//8)+1,valid[:,None]&(d[None,:]<D)&(shift[None,:]>2),0).to(tl.int32)
            c=(((byte|(tail<<8))>>shift[None,:])&63).to(tl.float32)-32
            cid=tl.load(CodeIDs+t,valid,0).to(tl.int64)
            center=tl.load(Center+cid[:,None]*D+d[None,:],valid[:,None]&(d[None,:]<D),0).to(tl.float32)
            s=tl.load(Scale+t,valid,0).to(tl.float32)
            if CENTERED:x=tl.where(d[None,:]<D,c*s[:,None]+center,0).to(tl.float16)
            else:x=(c*s[:,None]).to(tl.float16)
            v=tl.dot(q,tl.trans(x))
            if CENTERED:v+=qp[:,None]*tl.load(Bias+t,valid,0)[None,:]
            e=tl.load(Err+t,valid,0).to(tl.float32)
            upper=v+qn[:,None]*e[None,:]+GUARD
            keep=(tl.max((upper>=lm[:,None]).to(tl.int32),0)>0)&valid
        prefix=tl.cumsum(keep.to(tl.int32));nn=tl.sum(keep.to(tl.int32),0)
        tl.store(TokenIDs+i*MAXL+count+prefix-1,t,keep);count+=nn
    tl.store(Counts+i,count)

@triton.jit
def pack_token_ids(Padded, Counts, Prefix, Packed,N:tl.constexpr,MAXL:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0);j=tl.arange(0,B);nn=tl.load(Counts+i);start=tl.load(Prefix+i)
    for st in range(0,nn,B):
        v=tl.load(Padded+i*MAXL+st+j,st+j<nn,0)
        tl.store(Packed+start+st+j,v,st+j<nn)

@triton.jit
def gather_host_tokens(HostRaw,IDs,Stage,N:tl.constexpr,D:tl.constexpr,T:tl.constexpr):
    t=tl.program_id(0)*T+tl.arange(0,T);d=tl.arange(0,128)
    source=tl.load(IDs+t,t<N,0).to(tl.int64)
    x=tl.load(HostRaw+source[:,None]*D+d[None,:],(t[:,None]<N)&(d[None,:]<D),0,cache_modifier='.cg')
    tl.store(Stage+t[:,None]*D+d[None,:],x,(t[:,None]<N)&(d[None,:]<D))

@triton.jit
def uva_refine(Q,HostRaw,Codes,Scale,Err,Off,IDs,Upper,Cut,RowLower,
               Center,Bias,QP,QR,Scores,Counts,DocTokens,
               N:tl.constexpr,D:tl.constexpr,B:tl.constexpr,
               GUARD:tl.constexpr,CENTERED:tl.constexpr,FULLDOC:tl.constexpr,CodeIDs=None):
    r=tl.arange(0,32);d=tl.arange(0,128);j=tl.arange(0,B)
    q=tl.load(Q+r[:,None]*D+d[None,:],d[None,:]<D,0)
    qn=tl.sqrt(tl.sum(q.to(tl.float32)*q.to(tl.float32),1))
    if CENTERED:
        qp=tl.load(QP+r);qn=tl.load(QR+r)
    cutoff=tl.load(Cut)
    for i in range(tl.program_id(0),N,tl.num_programs(0)):
        up=tl.load(Upper+i);score=float('-inf');count=0;doclen=0
        if up>=cutoff:
            doc=tl.load(IDs+i);lo=tl.load(Off+doc).to(tl.int64);hi=tl.load(Off+doc+1).to(tl.int64)
            lm=tl.load(RowLower+i*32+r);mx=tl.full((32,),float('-inf'),tl.float32);doclen=(hi-lo).to(tl.int32)
            for st in range(lo,hi,B):
                t=st+j;valid=t<hi
                if FULLDOC:keep=valid
                else:
                    byte=tl.load(Codes+t[:,None]*(D*3//4)+(d[None,:]*6//8),valid[:,None]&(d[None,:]<D),0).to(tl.int32)
                    shift=(d%4)*6%8
                    tail=tl.load(Codes+t[:,None]*(D*3//4)+(d[None,:]*6//8)+1,valid[:,None]&(d[None,:]<D)&(shift[None,:]>2),0).to(tl.int32)
                    c=(((byte|(tail<<8))>>shift[None,:])&63).to(tl.float32)-32
                    cid=tl.load(CodeIDs+t,valid,0).to(tl.int64)
                    center=tl.load(Center+cid[:,None]*D+d[None,:],valid[:,None]&(d[None,:]<D),0).to(tl.float32)
                    s=tl.load(Scale+t,valid,0).to(tl.float32)
                    if CENTERED:x=tl.where(d[None,:]<D,c*s[:,None]+center,0).to(tl.float16)
                    else:x=(c*s[:,None]).to(tl.float16)
                    v=tl.dot(q,tl.trans(x))
                    if CENTERED:v+=qp[:,None]*tl.load(Bias+t,valid,0)[None,:]
                    e=tl.load(Err+t,valid,0).to(tl.float32)
                    tokenupper=v+qn[:,None]*e[None,:]+GUARD
                    keep=(tl.max((tokenupper>=lm[:,None]).to(tl.int32),0)>0)&valid
                raw=tl.load(HostRaw+t[:,None]*D+d[None,:],keep[:,None]&(d[None,:]<D),0,cache_modifier='.cg')
                exactv=tl.dot(q,tl.trans(raw))
                mx=tl.maximum(mx,tl.max(tl.where(keep[None,:],exactv,float('-inf')),1))
                count+=tl.sum(keep.to(tl.int32),0)
            score=tl.sum(mx)
        tl.store(Scores+i,score);tl.store(Counts+i,count);tl.store(DocTokens+i,doclen)
