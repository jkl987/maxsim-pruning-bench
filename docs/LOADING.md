# Chunked read-only mapped loading

```python
import numpy as np
from maxsim_pruning.loading import copy_into, copy_pinned_source

source = np.load("vectors.npy", mmap_mode="r")  # native FP16, [tokens, dimension]

# CPU-only output: NumPy is the only dependency.
output = np.empty(source.shape, dtype=np.float16)
copy_into(source, output, chunk_bytes=256 * 2**20)

# Optional alternative: pinned CPU tensor (CUDA-enabled PyTorch required).
pinned = copy_pinned_source(source, chunk_bytes=256 * 2**20)
```

Choose either output path; creating both consumes both allocations. The output
still occupies the full corpus size. The loader bounds copying chunks and
advises release of consumed read-only file-mapping pages via `MADV_DONTNEED`.
Actual RSS and I/O effects depend on the operating system and cache state.

The mapping must be opened directly in mode `r`, be C-contiguous, nonempty,
two-dimensional and native FP16. The source file must not change during the
copy. Do not use writable/copy-on-write mappings, sliced or reshaped views.
Reopen a suitable contiguous source explicitly rather than guessing a view's
offset. Chunk size must be a positive integer; the minimum work unit is a row.
The CPU destination must be writable, contiguous and independent of source.
Platforms without `mmap.madvise/MADV_DONTNEED` fail explicitly.

The release accounts for `.npy` headers and mapping offsets beyond one OS
allocation granularity. Tests include signed zero, infinity and NaN payloads,
compare bytes rather than numerical equality, and verify the source file hash
is unchanged. Run CPU tests with `python -m unittest discover -s tests -v`.
On a CUDA host, run `python scripts/validate_pinned_loader.py` for real pinned
allocation, exact copy and GPU transfer/readback verification.

The historical loader and measurements remain unchanged in `archive/source`.
The earlier NQ peak-RSS reduction was measured for that historical system;
this portable utility's new validation is correctness evidence only. It is not
a new retrieval algorithm, a bound on RSS, or an upstream integration claim.
