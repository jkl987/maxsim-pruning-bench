"""Bound temporary read-only file pages when materializing an FP16 pinned corpus.
Query algorithms and the resulting pinned tensor bytes are unchanged.
"""
import mmap
import numpy as np
import torch

def copy_pinned_source(source, chunk_bytes=2**28):
    if source.dtype != np.float16 or source.ndim != 2:
        raise ValueError('Expected a two-dimensional FP16 source mmap')
    if not hasattr(source, '_mmap') or not hasattr(source._mmap, 'madvise'):
        raise RuntimeError('This bounded-page loader requires mmap.madvise on Linux')
    pinned = torch.empty(source.shape, dtype=torch.float16, pin_memory=True)
    target = pinned.numpy()
    row_bytes = source.shape[1] * source.dtype.itemsize
    chunk_rows = max(1, chunk_bytes // row_bytes)
    page_bytes = mmap.PAGESIZE
    for first_row in range(0, source.shape[0], chunk_rows):
        last_row = min(source.shape[0], first_row + chunk_rows)
        np.copyto(target[first_row:last_row], source[first_row:last_row])
        # File mappings are read-only and separate from pinned output storage.
        first_byte = first_row * row_bytes // page_bytes * page_bytes
        last_byte = last_row * row_bytes
        source._mmap.madvise(mmap.MADV_DONTNEED, first_byte, last_byte-first_byte)
    source._mmap.madvise(mmap.MADV_DONTNEED)
    return pinned
