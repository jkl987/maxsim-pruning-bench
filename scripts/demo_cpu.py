#!/usr/bin/env python3
"""Data-free correctness demo. Its timings are not a search benchmark."""
import json
import numpy as np
from maxsim_pruning import coarse_scores, independent_upper, audit

rng = np.random.default_rng(20261019)
lut = rng.uniform(-1,1,size=(256,32))
docs = [rng.integers(0,256,size=int(n)) for n in rng.integers(1,65,size=256)]
truth = coarse_scores(lut, docs)
rows=[]
for levels in (2,4,8):
    upper, states = independent_upper(lut, docs, levels, 64)
    row = audit(truth, upper, 10)
    row.update(levels=levels, state_bytes_per_doc=len(states[0]), documents=len(docs))
    assert row['upper_violations']==0
    rows.append(row)
print(json.dumps({'scope':'Synthetic float64 correctness demo; not performance evidence','results':rows},indent=2))
