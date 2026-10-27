# Dataset scope

This source release contains no MRI scans or patient-level manifests.

- **MSD Task01 BrainTumour:** the actual primary segmentation cohort used here,
  derived from earlier BraTS data. It must not be mislabeled as BraTS 2021.
  484 labeled cases were partitioned 338/73/73 by case, with no case overlap.
  Case identifiers do not independently prove patient independence.
  Four modalities: FLAIR, T1w, contrast-enhanced T1, T2w.
  Labels: background=0, edema=1, non-enhancing tumor=2, enhancing tumor=3.
  WT={1,2,3}, TC={2,3}, ET={3}. It provides no healthy-control classifier cohort.
- **BTC local cohort:** 36 preoperative T1 scans, 25 tumor and 11 control.
  Preserve its original documentation and provenance locally. A public source
  identifier/license is not established in this release; do not redistribute it.
- **IXI:** 581 downloaded T1 scans, 464 development and 117 reserved participants.
  Healthy-only multi-site cohort. Metadata duplicate/conflict review is required;
  never resolve conflicting demographic rows by silently retaining the first.
- **UCSF-PDGM:** proposed independent evaluation; no completed external result.
- **BraTS-PEDs:** proposed separate pediatric generalization experiment, not pooled
  with adult data in the reported results.

Consult the source dataset's release-specific access and licensing documentation
before downloading or redistributing. Code licensing does not grant dataset or
third-party checkpoint rights. Scanner, preprocessing, source and demographic
confounding remain material classification limitations.

For nnU-Net use, cite Isensee et al. (2021), "nnU-Net: a self-configuring method
for deep learning-based biomedical image segmentation", Nature Methods.
HD-BET extraction experiments require their upstream source/weight attribution
and terms; no HD-BET assets are included here.
