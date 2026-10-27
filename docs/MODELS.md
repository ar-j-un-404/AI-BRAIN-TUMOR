# Local model connections

`python launch.py configure` writes ignored `app/settings.json`. No models are
copied or replaced. Paths below are relative to the configured original project.

- nnU-Net: `nnunet_reference_v1/results/Dataset501_MSDReviewed/nnUNetTrainer_250epochs__nnUNetPlans_2GB__3d_fullres/fold_0/checkpoint_final.pth`.
- Frozen 2D: `runs/stage6_internal_test_001/` evaluation plan and frozen checkpoint;
  the original adapter preserves its selection checks and configured fallback.
- Small 3D: `runs/3d_full_all4_seed42/best.pt`.
- Original classifier: separately configured results directory containing
  `experiment.json`, both variant folders, and five completed fold checkpoints.

Classifier code must match the training source hash or the exact reviewed
comment-free equivalent explicitly permitted by the adapter. Do not replace
`app/classifier_core.py` with a different preprocessing implementation.
Known development scans use their held-out fold; unknown scans use an explicitly
unvalidated five-fold ensemble. Reserved IXI names are blocked using the original
split lock; this is not a universal detector for renamed subjects.

nnU-Net verifies reviewed plans, dataset schema and checkpoint hash. Its final
checkpoint SHA256 is
`ff740475f8d6ec14b49056fd5180d275da8cb3dca609061a5789dc208473d6a3`.
Expected plans/schema are configuration, not scans or weights.

Use only trusted local checkpoints. Checkpoint existence alone does not verify
compatibility, provenance or quality.
