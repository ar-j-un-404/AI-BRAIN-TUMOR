# Reproducing and preserving experiments

The unified app runs from this repository using the original local trained
project and classifier directory. It does not start any training job.

`research_suite/` contains research training/evaluation code. Its original
project manifests, configurations, split locks, scans and checkpoints remain
necessary. Preserve their hashes and cohort membership.

`experiments/` collects source for BTC/IXI, physical-space, extraction, mask
review, input audits, BTC-only comparisons, nnU-Net and modality comparisons.
These are experiment source snapshots, not automatically runnable replacements
for their original folders. They import sibling modules and may require local
`reference.json`, `btc_plan.json`, `ixi_development.csv`, `ixi_split_lock.json`,
`extraction_source_lock.json`, `prior_review.json`, or `verified_inputs/`.
Those private/runtime files are excluded from publication. Use the original
completed experiment folder with its corresponding configuration to reproduce
an exact run; do not invent missing manifests or copy a different cohort's lock.
See `experiments/catalog.json` for the available source files.

Changing code may invalidate original source-hash locks. This release preserves
app classifier preprocessing behavior. Its integrity check accepts the exact
reviewed comment-free source as equivalent to the original training source.
Portability edits to old launcher
paths mean archived experiment source is for review and explicit new runs, not
silent resumption of old signed checkpoints. Keep original folders unchanged.

Training, all-data fitting and test evaluation are distinct operations. Do not
repeatedly select models using the reserved test cohort. An external evaluation
requires verified independent subjects and compatible annotations.

## Packaging validation

The lightweight tests check configuration, routing, results integrity, private
artifact exclusion and preservation of classifier source. Full model tests need
the existing PyTorch/nibabel runtime. Local packaging checks do not establish
clinical validity or successful CUDA inference on the user's computer.
