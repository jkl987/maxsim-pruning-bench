#!/usr/bin/env python3
"""Run the preserved independent GPU implementation on data-free inputs.

The archived implementation is not refactored or retuned here. Check coarse
scores against a float64 reference built from the same finite FP32 LUT.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from maxsim_pruning.evidence import require

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'archive/source'))
from frontier_v5_power32 import Power32Frontier

ap=argparse.ArgumentParser()
ap.add_argument('--device',type=int,default=0)
ap.add_argument('--output',type=Path)
args=ap.parse_args()
torch.cuda.set_device(args.device)
torch.backends.cuda.matmul.allow_tf32=False
device=torch.device('cuda',args.device)
config=json.loads((ROOT/'configs/validation.json').read_text())
fixture=config['gpu_demo']
rng=np.random.default_rng(config['seed'])
n,c,dim,budget=(fixture[k] for k in ['documents','centroids','terms','budget'])
require(dim==32,'Archived GPU kernels require 32 query terms')
counts=rng.integers(1,33,size=n)
off=np.concatenate(([0],np.cumsum(counts))).astype(np.int64)
codes=rng.integers(0,c,size=off[-1],dtype=np.int32)
postings=[[] for _ in range(c)]
for i in range(n):
    for code in np.unique(codes[off[i]:off[i+1]]):postings[code].append(i)
starts=np.concatenate(([0],np.cumsum([len(p) for p in postings]))).astype(np.int64)
ivf=np.concatenate([np.array(p,dtype=np.int32) for p in postings])
tensor=lambda x:torch.as_tensor(x,device=device)
offsets,code_tensor,istart,posting=map(tensor,(off,codes,starts,ivf))
rows=[]
for case in fixture['cases']:
    require(case in ('random','negative','ties'),'Unknown fixture case')
    cent=rng.uniform(-1,1,(c,dim)).astype(np.float32)
    if case=='negative':cent=-np.abs(cent)
    if case=='ties':cent[:]=0
    cent=tensor(cent);q=torch.eye(dim,device=device)
    # q=identity keeps each LUT term in [-1,1], satisfying the archived guard contract.
    lut=(cent@q.T).cpu().numpy().astype(np.float64)
    truth=np.array([lut[codes[off[i]:off[i+1]]].max(0).sum() for i in range(n)])
    cutoff=np.sort(truth)[-budget]
    for levels in fixture['levels']:
        engine=Power32Frontier(cent,code_tensor,offsets,istart,posting,
            budget,fixture['prefix'],2,levels,programs=128,block=128,compact_final=False,unsorted_seed=True)
        ids,valid=engine(q,16,budget)
        torch.cuda.synchronize()
        upper=engine.upper.cpu().numpy().astype(np.float64)
        chosen=ids.cpu().numpy()
        require(bool(valid.all()) and len(set(chosen.tolist()))==budget,'Invalid returned candidates')
        require(np.all(upper>=truth),f'{case}/{levels}: upper violation')
        # Float32 scoring may change near-ties; do not require arbitrary tied IDs.
        require(set(np.flatnonzero(truth>cutoff+4e-4)) <= set(chosen.tolist()),'Strict cutoff miss')
        require(np.all(truth[chosen]>=cutoff-4e-4),'Returned score below cutoff tolerance')
        times=[]
        for _ in range(fixture['repetitions']):
            torch.cuda.synchronize();t=time.perf_counter();engine(q,16,budget)
            torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
        rows.append({'case':case,'levels':levels,'state_bytes_per_doc':engine.dbits.numel()*engine.dbits.element_size()//n,
            'upper_violations':int(np.sum(upper<truth)),'times_ms':times})
        del engine
result={'scope':'Synthetic smoke test, no public-system speed claim','device':torch.cuda.get_device_name(),
        'torch':torch.__version__,'cuda':torch.version.cuda,'config':config,'results':rows}
text=json.dumps(result,indent=2)
if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text+'\n')
print(text)
