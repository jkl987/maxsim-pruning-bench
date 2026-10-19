#!/usr/bin/env python3
"""Optional real pinned-memory test; does not benchmark historical RSS savings."""
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from maxsim_pruning.loading import copy_pinned_source

with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'source.npy'
    values=np.random.default_rng(20261019).integers(0,65536,size=(2048,128),dtype=np.uint16).view(np.float16)
    np.save(path,values)
    source=np.load(path,mmap_mode='r')
    target=copy_pinned_source(source,chunk_bytes=131072)
    if not target.is_pinned() or target.numpy().tobytes()!=values.tobytes():
        raise RuntimeError('Pinned-memory copy parity failed')
    # Include a real host-to-device transfer and readback.
    restored=target.to('cuda',non_blocking=True).cpu()
    if restored.numpy().tobytes()!=values.tobytes():
        raise RuntimeError('CUDA transfer parity failed')
    source._mmap.close()
    print(json.dumps({'passed':True,'bytes':target.numel()*target.element_size(),
        'pinned':target.is_pinned(),'bitwise_parity':True,'cuda_roundtrip':True,
        'device':torch.cuda.get_device_name(),'torch':torch.__version__,
        'scope':'Correctness only; no new RSS or speedup measurement'},indent=2))
