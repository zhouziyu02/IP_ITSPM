# ITSPM — MSc Independent Projects

ITSPM is a PyTorch framework for irregular multivariate time series using event,
gap, and variable-summary tokens. This repository supports eight MSc Independent
Project (IP) topics. Each student develops one task on a single GPU (for example,
a Colab T4), reporting **mean ± sample standard deviation over three seeds**
with matched data splits across methods.

Start with the [Colab notebook](https://colab.research.google.com/github/zhouziyu02/IP_ITSPM/blob/main/notebooks/quickstart.ipynb),
[project milestones](docs/PROJECTS.md), [data guide](docs/DATASETS.md), and
[historical reference results](docs/REFERENCE_RESULTS.md).

## Setup and first run

Use **Python 3.11**. On Colab, select a GPU runtime and retain its matching
`torch`/`torchvision` pair. Locally, install a pair appropriate for your device
before installing the remaining dependencies.

```bash
git clone https://github.com/zhouziyu02/IP_ITSPM.git
cd IP_ITSPM
python3 -m pip install -r requirements.txt
python3 scripts/check_data.py --dataset bundled --verify

# Small real-data pipeline check, not a benchmark reproduction.
python3 regression.py --model ITSPM --dataset activity --task forecasting \
  --history 3000 -n 8 --epoch 1 --batch_size 8 --d_model 16 \
  --n_ref_points 8 --n_scales 2 --n_mixer_layers 1 \
  --max_event_tokens 16 --max_gap_tokens 8 --seed 0 --state quickstart
```

Activity, Taxi, and regenerated MuJoCo (length 36) are bundled; Sines is generated
locally. Download other datasets separately. CPU execution is available for small
checks (`--gpu -1` for classification/regression/generation, or `GPU=-1` for
Taxi). Full training time depends on the dataset and budget; start small and
save checkpoints to persistent storage when using Colab.

## Projects and dataset sources

The first five tasks have training pipelines. **Tasks 1.6–1.8 require student
extensions.** Baselines, ablations, and analyses in the milestones are project
work unless explicitly described as implemented.

| Project | Datasets and where to get them | Starting point / current status |
| --- | --- | --- |
| **1.1 Classification** | P12: [processed data + splits](https://doi.org/10.6084/m9.figshare.19514341.v1); PAM/PAMAP2: [processed data + splits](https://doi.org/10.6084/m9.figshare.19514347.v1). Raw: [PhysioNet 2012](https://physionet.org/content/challenge-2012/1.0.0/), [UCI PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring). | [classification.py](classification.py), [Classification/scripts](Classification/scripts). P12, PAM, optional P19 supported. |
| **1.2 Interpolation** | P12 regression tensors and USHCN: [t-PatchGNN data](https://github.com/usail-hkust/t-PatchGNN/tree/main/data). P12 can also be processed from [raw archives](https://physionet.org/content/challenge-2012/1.0.0/). | [regression.py](regression.py), `--task imputation`; [Interpolation/scripts](Interpolation/scripts). |
| **1.3 Extrapolation** | P12, Human Activity, USHCN: [t-PatchGNN data](https://github.com/usail-hkust/t-PatchGNN/tree/main/data). Activity is bundled; original source: [UCI Localization Data for Person Activity](https://archive.ics.uci.edu/dataset/196/localization+data+for+person+activity). | [regression.py](regression.py), `--task forecasting`; [Extrapolation/scripts](Extrapolation/scripts). |
| **1.4 Temporal point process** | Taxi is bundled. Taxi / StackOverflow: [EasyTPP data guide](https://ant-research.github.io/EasyTemporalPointProcess/user_guide/dataset.html), [releases](https://huggingface.co/easytpp). P12 observation events must be derived from the regression tensors. | [TPP/scripts/taxi.sh](TPP/scripts/taxi.sh). Taxi wired up; StackOverflow and P12 conversion require adaptation. |
| **1.5 Generation** | Stocks and Energy: [TimeGAN data files](https://github.com/jsyoon0823/TimeGAN/tree/master/data). [P12](https://physionet.org/content/challenge-2012/1.0.0/) requires a new adapter. Sines and bundled MuJoCo are additional starting datasets. | [generation trainer](Generation/itspm_generation/train_irregular_generation.py), [Generation/scripts](Generation/scripts). Sines/Stocks/Energy/MuJoCo supported. DS, MDD, KL, RBF-MMD and an ACF diagnostic implemented; predictive score and context-FID require implementation. |
| **1.6 Early classification** | Same [P12](https://doi.org/10.6084/m9.figshare.19514341.v1) and [PAM](https://doi.org/10.6084/m9.figshare.19514347.v1) releases as 1.1. | Extend classification: truncate prefixes **before** tokenization and summary computation; add random-prefix training. |
| **1.7 Anomaly detection** | SMD: [OmniAnomaly ServerMachineDataset](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset). SMAP: [Telemanom data and labels](https://github.com/khundman/telemanom#data). | Extend forecasting with dataset adapters, fixed targets, residual scores, validation thresholds. No anomaly pipeline bundled. |
| **1.8 Prediction intervals** | P12 and USHCN: [t-PatchGNN data](https://github.com/usail-hkust/t-PatchGNN/tree/main/data), as in 1.3. | Reuse forecast checkpoints; add a disjoint calibration split and interval metrics. No calibration pipeline bundled. |

**P12 classification arrays and regression tensors have different formats.
PAMAP2 and Human Activity are different datasets.** Exact paths, schemas,
preprocessing, and optional P19/MIMIC sources are in [DATASETS.md](docs/DATASETS.md).

## Implementation map

```text
models/ITSPM.py                  Shared classification/regression/generation backbone
classification.py               Classification head, training, evaluation, checkpoints
regression.py                   Interpolation/extrapolation training and checkpoints
lib/                            Data loading, splitting, normalization, masks, metrics
Classification/scripts/         P12 / PAM / P19 training presets
Interpolation/scripts/          P12 / Activity / USHCN / MIMIC interpolation presets
Extrapolation/scripts/          Matching forecasting presets
TPP/
  EasyTemporalPointProcess/      Adapted EasyTPP subset, configs, tests, licenses
  scripts/taxi.sh                Taxi training entry point
  taxi/                         Bundled train/dev/test event sequences
Generation/
  itspm_generation/             ITSPM autoencoder + latent DDPM and starter data
  scripts/                      Training presets and optional MuJoCo regeneration
  tests/                        Generation checks
data/activity/processed/        Bundled Activity tensor
docs/                           Data guide, eight project briefs, reference results
notebooks/quickstart.ipynb       Colab setup and small runs
scripts/                        Data checks and CPU test runner
tests/                          Shared model and CLI checks
```

In **[models/ITSPM.py](models/ITSPM.py)**:

| Component | Class / method | Projects |
| --- | --- | --- |
| Values, variable IDs, continuous time, mask, elapsed time | `EventEncoder`, `SinTimeEmbedding` | 1.2, 1.4 |
| Event, gap, variable-summary tokens | `FastSparsePatternTokenizer` | 1.1, 1.4, 1.6 |
| Token mixing, type pooling, variable states, global aggregation | `PatternInteraction` | 1.5 |
| Soft temporal summaries and local multi-bandwidth readout | `KernelSummaryBranch`, `ITSPM._local_kernel_readout` | 1.3 |
| Query-time predictions | `QueryTokenReadout`, `ITSPM.forecasting` | 1.2, 1.3, 1.7, 1.8 |
| Sample representation | `ITSPM.forward` | 1.1, 1.6 |

The **separate causal TPP adaptation** is
[torch_itspm.py](TPP/EasyTemporalPointProcess/easy_tpp/model/torch_model/torch_itspm.py),
registered as `ITSPM`; it does not import the shared backbone. Generation's
`ITSPMLatentAutoencoder` does use that backbone.

Shared interfaces: values/masks `[B,L,C]`; timestamps `[B,L]`, `[B,L,1]`, or
`[B,L,C]`. `forward(...)` returns `[B,D]`; `forecasting(...)` returns
`[1,B,L_query,C]`. Sort timestamps and use mask `1` for observed entries.

**Configuration caveat:** `n_ref_points` affects default token budgets and kernel
centers. `n_scales` currently supplies the attention-head count (adjusted to
divide `d_model`), not the number of temporal bandwidths. Three trainable local
bandwidths live in `local_log_bw`; implement explicit scale control for project
1.3. Unused legacy `kernel_type` / `use_periodic_branch` CLI flags are removed.
Ablations must account for information still present through parallel summary
and kernel paths; see [PROJECTS.md](docs/PROJECTS.md).

## Training and evaluation

Run from the repository root after preparing the relevant data:

```bash
python3 scripts/check_data.py --dataset p12-classification
GPU=0 SEED=0 bash Classification/scripts/P12.sh
GPU=0 SEED=0 bash Classification/scripts/PAM.sh
GPU=0 SEED=0 bash Interpolation/scripts/physionet.sh
GPU=0 SEED=0 bash Extrapolation/scripts/ushcn.sh
GPU=0 SEED=0 bash TPP/scripts/taxi.sh
GPU=0 SEED=0 bash Generation/scripts/stocks.sh
```

Presets retain the source budgets (up to 500/1000 epochs with early stopping for
classification/regression). Trailing CLI arguments override the preset:

```bash
GPU=0 SEED=0 bash Classification/scripts/P12.sh --epoch 5 --batch_size 8
GPU=0 SEED=0 bash TPP/scripts/taxi.sh --epochs 2 --batch-size 32
SEQ_LENS=12 MISSING_RATIOS=0.5 N_SAMPLES=256 DS_ITERATIONS=10 \
  GPU=0 SEED=0 bash Generation/scripts/sines.sh \
  --ae_epochs 2 --diff_epochs 2 --batch_size 16

# Three seeds, identical data split and configuration.
for seed in 0 1 2; do
  GPU=0 SEED="$seed" bash Extrapolation/scripts/activity.sh
done
```

Set `SEED` explicitly: script defaults are historical single-run seeds.
Generation presets loop over lengths 12/24/36 and missing rates 0.3/0.5/0.7;
MuJoCo defaults to bundled length 36 and rate 0.5. Override `SEQ_LENS` and
`MISSING_RATIOS` to reduce the grid.

| Task | Outputs | Checkpoint selection |
| --- | --- | --- |
| Classification | `checkpoints/classification/`, `Classification/results/Classification_results.txt` | Validation AUPRC by default, including macro one-vs-rest AUPRC for PAM. |
| Interpolation/extrapolation | `checkpoints/regression/`, `Interpolation/results/` or `Extrapolation/results/` | Minimum validation MSE. |
| TPP | `outputs/tpp_taxi/`, including `test_metrics.json` | Maximum validation log-likelihood, then test once. |
| Generation | `Generation/itspm_generation/outputs/` | Fixed training budgets, without validation checkpoint search. |

Classification/regression support `--test_only --load_path PATH` with matching
task, data, model, and preprocessing arguments. Regression reconstructs fitted
preprocessing from the same training data; its checkpoint does not store those
statistics. Generation's `sample_from_checkpoint` Python API loads `model.pt`
and samples using saved normalization. Override result files with
`ITSPM_CLASSIFICATION_RESULTS_FILE` or `ITSPM_REGRESSION_RESULTS_FILE`.

The generator samples from noise, without evaluation observations. Its
`gaussian_baseline` sees complete training values, including artificially hidden
ones: label it as a complete-data reference. Diff-MN constants are published
reference values, not executed baselines. Stocks/Energy retain benchmark-style
global min-max scaling and random overlapping windows; introduce chronological
splits and training-only scaling if studying temporal generalization.

## Checks and provenance

```bash
bash scripts/run_checks.sh
```

Checks cover shapes, gradients, masks, metrics, data boundaries, causal TPP
states, checkpoint round trips, and tiny training/sampling runs. They do not
establish full benchmark performance or a Colab runtime budget.

This teaching snapshot starts from
[source commit 4e12080](https://github.com/zhouziyu02/HeteroNet/tree/4e12080),
renames the model and active interfaces to **ITSPM**, and adds student-facing
documentation and checks. Historical results are summarized with limitations in
[REFERENCE_RESULTS.md](docs/REFERENCE_RESULTS.md). Git ignores downloaded data,
logs, caches, and checkpoints; the three bundled datasets are listed in
[DATA_MANIFEST.json](DATA_MANIFEST.json). Attribution is retained in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and component license files.
