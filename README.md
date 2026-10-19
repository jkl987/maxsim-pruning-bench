# MaxSim pruning audits and reference controls

Reproducible diagnostics for **fixed-codebook coarse MaxSim filtering**. This
repository preserves useful independent-bin controls and the negative results
of a conditional-summary investigation. It is a research artifact, not a
production search engine or a claim of a new state-of-the-art algorithm.

## Run without data or a GPU

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/demo_cpu.py
python scripts/recompute.py
```

The CPU demo is a small float64 audit reference. It checks independent unary
membership bounds at 8, 16 and 32 bytes per document for 32 query terms.
`recompute.py` verifies archived SHA256 hashes and recomputes comparisons from
per-query raw timings. No embeddings, models, credentials or server are needed.

## Run the preserved GPU implementation

Install a CUDA-enabled PyTorch build appropriate for your machine, then:

```sh
python -m pip install -e '.[gpu]'
python scripts/demo_gpu.py --device 0 --output run-output/gpu-smoke.json
```

This runs the original `Power32Frontier` implementation on 1,024 synthetic
documents, with random, negative and tied scores at three state budgets. It
checks upper bounds and tie-aware coarse Top128 membership. Reported demo
timings are smoke-test measurements, not the historical corpus benchmark.
No kernel retuning is performed by this release.

The historical GPU code lives in `archive/source/` and retains its original
hashes and import structure. Experimental conditional variants are present for
inspection; the runnable GPU demo selects only the independent control.

## What the measurements support

Both NQ and HotpotQA were observed during development. These are local
comparisons with our own matched controls, not upstream-system speedups.

| Observed full refinement pipeline | Independent 8 bins, 32B | Independent 4 bins, 16B | Time change |
|---|---:|---:|---:|
| NQ | 1.33009 ms | 1.21143 ms | −8.92% |
| HotpotQA | 2.46803 ms | 1.96476 ms | −20.39% |

These are means of per-query medians over five paired passes, 64 fixed queries,
one process session. Final Top100 IDs matched for the observed sample. They do
not establish raw-vector exactness or held-out generalization.

The local conditional hybrid did **not** beat the simple same-budget control:

| Coarse-only pipeline, 24B per document for both | Hybrid time change |
|---|---:|
| NQ, three process sessions | +2.10% |
| HotpotQA, three process sessions | +17.26% |

These coarse-only timings must not be mixed with the full-pipeline table.
The conditional route was stopped; its failed variants and diagnostics remain
in `evidence/failed/`. The third recorded hybrid correctness failure duplicates
the second failure JSON; it is not an independent third measurement.

## Scope and correctness

- Coarse scores are the sum, over query terms, of maximum lookup-table values
  among a document's centroid codes. This is not raw-token global Top100.
- An upper bound equal to the cutoff survives. Scores strictly above the
  cutoff are mandatory; arbitrary tied document IDs need not be identical.
- The GPU implementation's floating-point guard assumes a bounded finite LUT.
  The synthetic GPU fixture explicitly satisfies that contract. Do not use the
  archived guard for arbitrary-magnitude inputs without a new error analysis.
- State bytes describe the packed per-document bit fields, not total workspace
  or GPU memory. Historical diagnostics separately record shared indexes,
  workspace and temporary allocations where measured.
- Phase profiling and formal synchronized wall-clock timings are separate.
  Repetitions and query bootstrap do not measure machine-to-machine variation.

## Layout

- `src/maxsim_pruning/`: portable float64 reference and audit helpers.
- `tests/`: boundary, negative-value, duplicate-code and input-contract checks.
- `scripts/`: data-free CPU/GPU demos and raw-result recomputation.
- `evidence/raw/`: original protocol, provenance, per-query results and timings.
- `evidence/failed/`: retained failures and invalid/pilot configurations.
- `archive/source/`: unchanged historical experiment source, including the
  pinned-memory loader and conditional diagnostics.
- `docs/REPRODUCIBILITY.md`: what can and cannot be reproduced from this release.

## Contributing

Useful contributions include numerical edge cases, clear memory accounting,
portable loaders and matched-budget baselines. Include a minimal reproducer and
exact environment. Performance changes should report paired query timings,
quality, total memory and the full measured boundary; preserve regressions.
Please avoid claiming novelty or SOTA from synthetic fixtures or own-control
comparisons. Run the CPU checks above before proposing a change.

## Related work and attribution

This work studies established ideas: lookup-table MaxSim, membership summaries,
score bounds, threshold pruning and GPU data layout. See
[PLAID](https://arxiv.org/abs/2205.09707),
[FastPLAID](https://github.com/lightonai/fast-plaid),
[late-interaction-kernels](https://github.com/hcompai/late-interaction-kernels),
and [GIGP+](https://github.com/DBGroup-SUSTech/multi-vector-retrieval).
Third-party engines and datasets are not bundled. See `NOTICE` and `LICENSE`.
