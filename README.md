# BrainTumor AI

A local MRI research workspace connecting 2D segmentation, 3D segmentation,
interactive 3D visualization, and exploratory T1 classification.

**Research prototype, not a diagnostic system.** Segmentation overlap is not
classification accuracy. The classifier has low healthy-control specificity.

## Start with your existing trained project

Extract or clone this repository into a new folder. Keep the original project,
its virtual environments, model checkpoints, and datasets where they are.
Double-click **START_APP.bat**. On first launch, select the inner project folder
containing `research_suite` and `runs`, then optionally select the original
`BTC_IXI_Training/results` folder. The launcher uses that project's existing
Python environment and opens the app at http://localhost:8507.

Alternatively, in Windows Command Prompt:

```bat
python launch.py configure --project "C:\path\to\BrainTumor_AI_Stage2" --classifier "C:\path\to\BTC_IXI_Training\results"
python launch.py check
python launch.py start
```

The classifier option is optional. Run `configure` again to change locations.
The nnU-Net adapter expects the existing Windows `.venv_nnunet` environment;
the main app uses the main `.venv`. Do not mix their dependencies.

## Workspace pages

- **MRI segmentation & 3D:** four-channel MRI or four assigned modalities,
  optional reference mask, model inference, separate 2D/3D viewer tabs,
  interactive rotation, zoom, region controls, and report/mask exports.
- **T1 classification:** original classifier checkpoints with fold membership
  and preprocessing checks; scores are not calibrated disease probabilities.
- **Models & results:** connection checks and evaluation context.
- **Experiment comparisons:** aggregate results including unsuccessful variants.

For MSD scans, modality order is FLAIR, T1w, contrast-enhanced T1, T2w.
Reference mask labels are 0 background, 1 edema, 2 non-enhancing tumor,
3 enhancing tumor. MRI inputs must already satisfy the model's preprocessing
and alignment requirements. The app does not perform registration.

## Reported performance

| Segmentation model | Evaluation | Whole tumor Dice | Tumor core Dice | Enhancing tumor Dice |
|---|---|---:|---:|---:|
| nnU-Net 3D | 73 validation cases | 90.83% | 83.41% | 76.76% |
| 2D U-Net | 73 previously evaluated internal-test cases | 90.78% | 81.36% | 75.58% |
| Small 3D U-Net | 73 validation cases | 82.38% | 66.86% | 65.12% |

These rows use different evaluation contexts and are not a paired ranking.

The retained BTC + IXI classification baseline achieved **75.00% BTC accuracy**,
**59.09% balanced accuracy**, **100% sensitivity**, and **18.18% specificity**
on 36 development participants (25 tumor, 11 healthy). Nine healthy participants
were false positives. It is not reliable tumor detection.

Values are aggregate results from prior reports, not newly computed by this
repository build. See [results and limitations](docs/RESULTS.md).

## Repository layout

```text
app/                 Connected viewer and inference adapters
research_suite/      Original research training and evaluation source
experiments/         Comparison, extraction, audit and benchmark source
launch.py            Single configuration/check/start entry point
tests/               Synthetic contract and packaging checks
docs/                Results, model requirements, data and reproduction notes
```

## Dependencies and development

Use the existing working environment first. `requirements.txt` records the
research environment's package pins; PyTorch is installed separately because
CUDA support depends on the machine. Existing setup used PyTorch
2.10.0+cu128 and nnunetv2 2.8.1 in a separate environment.
No automatic package installation, model download, or training occurs on launch.

```bat
python -m unittest discover -s tests -p "test_repository.py"
python -m unittest discover -s tests -p "test_connections.py"
```

The second command needs the full model runtime but uses synthetic scans and
mock external inference, not private training data. See `docs/REPRODUCING.md`
for what can and cannot be rerun from the source-only release.

## Data and weights

MRI scans, patient-level manifests, checkpoints, cached arrays and local
settings are deliberately not distributed. This repository cannot produce
trained predictions without the corresponding local weights and configuration.
Source code is MIT licensed; dataset and third-party model terms remain separate.
See [data notes](docs/DATASETS.md) and [model connections](docs/MODELS.md).
