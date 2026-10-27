# Results and unfinished work

Source: prior user-provided experiment reports, summarized in
`app/results_summary.json`. No new clinical evaluation was performed while
assembling this repository.

## Classification: 36 BTC development participants

| Experiment | Accuracy | Balanced accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| Original BTC + IXI | 75.00% | 59.09% | 100.00% | 18.18% |
| Physical-space BTC + IXI | 58.33% | 42.00% | 84.00% | 0.00% |
| Brain-extracted BTC + IXI | 52.78% | 40.55% | 72.00% | 9.09% |
| Brain-extracted BTC only | 50.00% | 38.55% | 68.00% | 9.09% |

All reuse a small development cohort. Repeated exploration makes these results
unsuitable as an untouched final test. Participant identity, labels, acquisition
and preprocessing need continued review. Brain extraction masks are not all
clinically verified. Worse variants are not connected as replacement app models.

The original classifier correctly called all 464 development IXI scans healthy.
This is healthy-only specificity, not tumor detection accuracy. The 117 reserved
IXI participants were not used in these experiments.

## Segmentation

The nnU-Net model obtained validation WT/TC/ET Dice of 90.83/83.41/76.76 percent
on 73 cases. The 2D internal-test result is 90.78/81.36/75.58 percent on a
previously evaluated 73-case set. Small 3D validation was 82.38/66.86/65.12.
Different cohorts/checkpoint-selection procedures prevent a direct ranking.
Patch pseudo-Dice and single-case viewer metrics are separate quantities.

## Remaining

- Reliable tumor-versus-healthy classification, especially healthy specificity.
- Independent compatible external validation and frozen model selection.
- Review of acquisition/source shortcuts, subject identity and extraction masks.
- Separate pediatric experiment if data terms, annotations and cohort justify it.
- End-to-end runtime verification of this packaging on the original GPU machine.

The 3D rendering displays a segmentation in MRI physical space; visual realism
is not evidence of prediction correctness. Optional reference overlays support
inspection but do not establish clinical fitness.
