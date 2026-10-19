#!/usr/bin/env python3
"""Data-free correctness demo. Its timings are not a search benchmark."""
import json
from pathlib import Path
import numpy as np
from maxsim_pruning import coarse_scores, independent_upper, audit
from maxsim_pruning.evidence import require

config=json.loads((Path(__file__).resolve().parents[1]/'configs/validation.json').read_text())
fixture=config['cpu_demo']
rng = np.random.default_rng(config['seed'])
lut = rng.uniform(-1,1,size=(fixture['centroids'],fixture['terms']))
docs = [rng.integers(0,fixture['centroids'],size=int(n)) for n in rng.integers(1,65,size=fixture['documents'])]
truth = coarse_scores(lut, docs)
rows=[]
for levels in fixture['levels']:
    upper, states = independent_upper(lut, docs, levels, fixture['prefix'])
    row = audit(truth, upper, fixture['budget'])
    row.update(levels=levels, state_bytes_per_doc=len(states[0]), documents=len(docs))
    require(row['upper_violations']==0,'CPU reference upper violation')
    rows.append(row)
print(json.dumps({'scope':'Synthetic float64 correctness demo; not performance evidence','results':rows},indent=2))
