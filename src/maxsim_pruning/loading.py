"""Chunked loading of a direct read-only FP16 memmap into owned output storage.

MADV_DONTNEED advises release of consumed source mapping pages. It does not
reduce corpus size or guarantee a fixed RSS on every operating system.
"""
import mmap
from numbers import Integral
import numpy as np


def _validate(source, chunk_bytes):
    if not isinstance(source, np.memmap) or not isinstance(source.base, mmap.mmap):
        raise ValueError('Expected a direct memmap; sliced and reshaped views are unsupported')
    if source.mode != 'r' or source.flags.writeable:
        raise ValueError('Source must be opened read-only (mode="r")')
    if source.ndim != 2 or source.dtype != np.dtype(np.float16) or not source.flags.c_contiguous:
        raise ValueError('Expected a C-contiguous two-dimensional native FP16 mapping')
    if min(source.shape) == 0:
        raise ValueError('Empty dimensions are unsupported')
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, Integral) or chunk_bytes <= 0:
        raise ValueError('chunk_bytes must be a positive integer')
    if source._mmap.closed:
        raise ValueError('Source mapping is closed')
    if not hasattr(source._mmap, 'madvise') or not hasattr(mmap, 'MADV_DONTNEED'):
        raise RuntimeError('This loader requires mmap.madvise and MADV_DONTNEED')


def copy_into(source, target, *, chunk_bytes=2**28):
    """Copy into a caller-owned ndarray, then release consumed mapping pages.

    Offset mappings (including .npy headers) are supported. Direct mappings only:
    rejecting views avoids guessing their byte displacement inside the mapping.
    Source files must remain unmodified for the duration of the copy.
    """
    _validate(source, chunk_bytes)
    if not isinstance(target, np.ndarray) or isinstance(target, np.memmap):
        raise ValueError('Target must be an owned-memory ndarray, not a memmap')
    if target.shape != source.shape or target.dtype != source.dtype or not target.flags.c_contiguous:
        raise ValueError('Target must match source shape/dtype and be C-contiguous')
    if not target.flags.writeable or np.shares_memory(source,target):
        raise ValueError('Target must be writable and independent of source')
    row_bytes = source.shape[1]*source.itemsize
    chunk_rows = max(1, int(chunk_bytes)//row_bytes)
    data_start = int(source.offset) % mmap.ALLOCATIONGRANULARITY
    for first in range(0, len(source), chunk_rows):
        last = min(len(source), first+chunk_rows)
        np.copyto(target[first:last],source[first:last])
        begin = (data_start+first*row_bytes)//mmap.PAGESIZE*mmap.PAGESIZE
        end = data_start+last*row_bytes
        source._mmap.madvise(mmap.MADV_DONTNEED, begin, end-begin)
    return target


def copy_pinned_source(source, *, chunk_bytes=2**28):
    """Return a PyTorch pinned CPU FP16 tensor; requires CUDA-enabled PyTorch."""
    _validate(source, chunk_bytes)
    import torch
    target = torch.empty(source.shape, dtype=torch.float16, pin_memory=True)
    copy_into(source,target.numpy(),chunk_bytes=chunk_bytes)
    return target
