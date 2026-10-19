# Reproducibility levels

## Included and directly runnable

1. Verify every archived evidence/source SHA256 with `python scripts/recompute.py`.
2. Recompute the two full-pipeline engineering comparisons and the two pooled
   same-byte hybrid comparisons from raw per-query measurements.
3. Run the CPU reference tests and data-free demo on any supported Python host.
4. Run the original independent GPU control on synthetic data with CUDA,
   PyTorch and Triton. This checks executability and coarse-score contracts.

## Included for audit, requiring original corpus caches for a full rerun

`archive/source/v6_diagnose.py`, `v6_engineering_stream.py` and the other historical
drivers retain their original paths and protocol. They are not a portable
one-command rebuild of the full NQ/HotpotQA experiment. They expect trained
codebooks, inverted indexes, corpus embeddings, query embeddings, qrels and
the existing refinement caches. Those large inputs are not bundled here.
The archived source includes dynamic loading of earlier experiment drivers;
it is preserved for inspection rather than advertised as a clean public API.

The records in `evidence/raw/model-provenance.json` document the model revisions
and encoding protocols; `protocol.json` documents fixed sampling and budgets.
Do not re-encode an entire corpus merely to run this repository's smoke tests.
The original full diagnostic package remains a separate local research artifact.

## Negative results and exclusions

The archive retains pilot results, invalid local-oracle comparisons, failed
hybrid implementations and associated failure records. They are not included
in successful performance aggregates. The conditional hypothesis failed its
same-budget development-data gate; this release does not reinterpret that as
a positive new algorithm. Full four-cell corpus/encoder comparisons, unseen
data generalization, hardware-counter traffic, concurrency and public-system
integration are not established by this release.

## Timing interpretation

Historical full-pipeline comparisons use CPU query embedding to returned CPU
Top100 IDs/scores, including unchanged raw refinement. The hybrid table measures
coarse retrieval only. The synthetic demo times a synchronized GPU coarse call;
it is a different workload and must not be used to reproduce the corpus table.
Warmup/compilation are outside the demo's timed repetitions. Correctness checks
and CPU copies are also outside those repetitions; no end-to-end claim is made.

The measured GPU guard is specific to finite bounded LUT values. The CPU audit
is float64 and checks one computed reference result; neither is a proof for
all unbounded inputs. Use strict-above-cutoff misses separately from arbitrary
tie IDs, and record selected-below-cutoff results as well.
