# Dataset preparation

Run commands from the repository root. Dataset links point to the original
provider or the authors' preprocessing release. Keep the published splits for
comparisons; do not replace a missing dataset with synthetic data. Downloaded
files are local inputs and are ignored by Git.

## Classification: P12, PAM, optional P19

Use the **processed arrays and split files** released with
[Raindrop](https://github.com/mims-harvard/Raindrop#processed-data):

| Dataset | Processed release | Raw source |
| --- | --- | --- |
| P12 | [Figshare, version 1](https://doi.org/10.6084/m9.figshare.19514341.v1) | [PhysioNet Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/) |
| PAM / PAMAP2 | [Figshare, version 1](https://doi.org/10.6084/m9.figshare.19514347.v1) | [UCI PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) |
| P19 (optional) | [Figshare, version 1](https://doi.org/10.6084/m9.figshare.19514338.v1) | [PhysioNet Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) |

Download/extract the relevant release and copy its dataset directory into
`data/`. Avoid an extra nested `P12data/P12data/` directory. Required files:

```text
data/P12data/processed_data/PTdict_list.npy
data/P12data/processed_data/arr_outcomes.npy
data/P12data/splits/phy12_split1.npy
data/PAMAP2data/processed_data/PTdict_list.npy
data/PAMAP2data/processed_data/arr_outcomes.npy
data/PAMAP2data/splits/PAMAP2_split_1.npy
data/P19data/processed_data/PT_dict_list_6.npy
data/P19data/processed_data/arr_outcomes_6.npy
data/P19data/splits/phy19_split1_new.npy
```

`lib/Dataset_MM.py` defines the contract: each record has `id`, `time`, and
`arr`; the P12 preset also enables `--use_static` and uses `extended_static`
or `static`. Outcomes must align with records; the loader uses the last outcome
column as the label. A split contains train/validation/test index arrays.
Classification masks follow the inherited processed-data convention `arr > 0`;
do not apply this convention to a new dataset with legitimate negative/zero
observations. `time == 0` after the first position marks padding in this format.

```bash
python3 scripts/check_data.py --dataset p12-classification
python3 scripts/check_data.py --dataset pam
GPU=0 SEED=0 bash Classification/scripts/P12.sh
```

For a nondefault data directory use `--data_path /path/to/data`; use `--split 2`
only if the corresponding split file is present. Three model seeds on split 1
measure seed variability, not variability across five published splits. For
project 1.6, truncate raw records before normalization/collation/tokenization as
appropriate and keep all normalization fitting confined to training data.

## Interpolation/extrapolation: P12, Activity, USHCN

The authors' [t-PatchGNN data directory](https://github.com/usail-hkust/t-PatchGNN/tree/main/data)
provides compatible processed data. Copy the relevant subdirectory into `data/`.
Extract PhysioNet's `processed.zip` if using that release.

```text
data/physionet/processed/set-a_0.0.pt
data/physionet/processed/set-b_0.0.pt
data/physionet/processed/set-c_0.0.pt
data/activity/processed/data.pt                 # already bundled
data/ushcn/processed/ushcn.pt                   # or data/ushcn/ushcn.pt
```

Alternatively, the local loaders preprocess these raw inputs on first use:

| Dataset | Required local raw inputs | Download source |
| --- | --- | --- |
| P12 | `data/physionet/raw/set-a.tar.gz`, `set-b.tar.gz`, `set-c.tar.gz` | [PhysioNet 2012 files](https://physionet.org/content/challenge-2012/1.0.0/#files-panel) |
| Activity | `data/activity/raw/ConfLongDemo_JSI.txt` | [UCI Localization Data for Person Activity](https://archive.ics.uci.edu/dataset/196/localization+data+for+person+activity) |
| USHCN | `data/ushcn/raw/small_chunked_sporadic.csv` | [t-PatchGNN USHCN files](https://github.com/usail-hkust/t-PatchGNN/tree/main/data/ushcn), or [GRU-ODE-Bayes climate preprocessing](https://github.com/edebrouwer/gru_ode_bayes/tree/master/gru_ode_bayes/datasets/Climate) |

The USHCN CSV is already benchmark-preprocessed; it is not an arbitrary NOAA
station file. It has an integer record index, `Time`, `Value_*`, and matching
`Mask_*` columns. Its loader rescales time by `48/200`. The current loaders do
not download data automatically, even where an inherited argument is called
`download=True`.

Records are sequences `(record_id, times, values, mask)` with times `[L]` and
values/masks `[L,C]`. Some source tensors include extra fields consumed by their
specific loaders; preserve those release formats. Loaders are
`lib/physionet.py`, `lib/person_activity.py`, and `lib/ushcn.py`; splitting,
masking, and batching are wired in `lib/parse_datasets.py`.

```bash
python3 scripts/check_data.py --dataset p12-regression
python3 scripts/check_data.py --dataset ushcn
GPU=0 SEED=0 bash Interpolation/scripts/physionet.sh
GPU=0 SEED=0 bash Extrapolation/scripts/ushcn.sh
```

The forecasting presets use history **24 hours for P12**, **3000 ms for
Activity** (next 1000 ms), and **24 months for USHCN** (next month, chunked
windows). The imputation protocol masks a deterministic fraction of time rows
(`--mask_rate`, default 0.3), rather than independently masking every scalar.
It differs from the observed-ratio sweeps proposed in project 1.2. Normalization
is fitted on training records, splits precede window expansion, and unusable
windows are reported/excluded. Keep target masks and record identities identical
across compared models. For project 1.8, reserve calibration identities from the
training/validation allocation **before** fitting or selecting models.

### Optional MIMIC

Use the credentialed [MIMIC-III-Ext-tPatchGNN release](https://physionet.org/content/mimic-iii-ext-tpatchgnn/)
linked by [t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN).
Follow its access process and data-use terms. Supply `data/mimic/mimic.pt`, or
`data/mimic/processed/mimic.pt`, or the compatible benchmark CSV at
`data/mimic/raw/full_dataset.csv`. This repository has no MIMIC data. MIMIC is
optional for these IP topics; P12/Activity/USHCN are the smaller starting choices.

## Temporal point process: Taxi, StackOverflow, P12 events

Taxi `train.pkl`, `dev.pkl`, and `test.pkl` are bundled under `TPP/taxi/`.
For other datasets use the [EasyTPP dataset guide](https://ant-research.github.io/EasyTemporalPointProcess/user_guide/dataset.html)
and the authors' [Hugging Face releases](https://huggingface.co/easytpp). Prefer
their Gatech pickle release for compatibility with the bundled loader; Hugging
Face JSON/Arrow data needs conversion, not a filename change.

The pickle is a dictionary with `dim_process` and a split key (`train`, `dev`,
or `test`) containing sequences of event dictionaries:

```python
{
    "time_since_start": 1.25,
    "time_since_last_event": 0.25,
    "type_event": 2,  # integer in [0, dim_process)
}
```

```bash
python3 scripts/check_data.py --dataset taxi
GPU=0 SEED=0 bash TPP/scripts/taxi.sh
```

Taxi has ten event types. The launcher and `examples/taxi_config.py` currently
support Taxi only. To add StackOverflow, extend the launcher/configuration,
point to its three split files, and set its `num_event_types` and padding ID;
do not reuse Taxi's event count. For P12, build one sequence per patient from
observed `(timestamp, variable_id)` pairs. Define simultaneous-event ordering
and zero-gap handling, use consistent time units, and split patients before
conversion. This conversion and the additional dataset launchers are student work.

The bundled EasyTPP subset contains ITSPM, not RMTPP/NHP/THP baselines. Use a
separate checkout/environment of [upstream EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess)
for these baselines; compare the exact same event files and time units.

## Generation: Stocks, Energy, Sines, MuJoCo

Download `stock_data.csv` and `energy_data.csv` from the
[TimeGAN data directory](https://github.com/jsyoon0823/TimeGAN/tree/master/data)
and place them at:

```text
Generation/itspm_generation/table1_data/stock_data.csv
Generation/itspm_generation/table1_data/energy_data.csv
Generation/itspm_generation/table1_data/mujoco_training_36.pt  # bundled
```

Both CSVs must contain numeric columns after one header row. A new source with
a date column requires explicit cleaning. The current loader reverses the rows,
applies global min-max scaling, creates overlapping windows, and shuffles them
before splitting; preserve this protocol only for comparisons using that same
benchmark setup. The later standardizer uses observed training entries only.
Sequence-level missing masks remove time rows; the realized missing rate can
differ from the requested rate because Sines/MuJoCo retain endpoints.

Sines is generated deterministically and needs no download. MuJoCo is real
HopperPhysics simulation regenerated with recorded simulator versions and seed;
it is **not** a bit-identical original release. See its adjacent provenance JSON.
Length 36 is bundled; generate other lengths explicitly:

```bash
python3 Generation/scripts/install_physics.py
python3 Generation/scripts/prepare_mujoco.py --lengths 12 24 36
SEQ_LENS="12 24 36" GPU=0 SEED=0 bash Generation/scripts/mujoco.sh
```

Physics dependencies are optional and installed only for regeneration. The
original recipe is attributed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
To use a separate generation data directory set `DATA_ROOT=/absolute/path`
for scripts, or `--data_root` for the Python trainer. P12 generation needs a new
adapter preserving values, timestamps, masks, and patient splits.

## Anomaly detection: SMD and SMAP (adapter required)

SMD's [original repository](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset)
contains `train/`, `test/`, and `test_label/` files per machine. SMAP is available
through [Telemanom](https://github.com/khundman/telemanom#data), whose current
download instructions use the [SMAP/MSL Kaggle release](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl)
(Kaggle access may require an account). Obtain train/test arrays and
`labeled_anomalies.csv`. Decode each channel's listed
anomaly intervals to pointwise labels and check inclusive endpoints against the
source definition. Both datasets are regularly sampled; the project introduces
irregularity by thinning historical observations.

Keep downloaded originals under ignored `data/SMD/` and `data/SMAP/`. Before
experiments, record a fixed one- or two-entity selection per dataset and the
normal-validation partition. Implement conversion to forecasting windows and
label alignment; these formats are **not** accepted by `regression.py` yet.
Retain identical prediction targets/labels across thinning levels and methods.
Use normal validation data for residual scaling and threshold choice, report
test false-positive rate, and do not use point adjustment.

## Data checks

`python3 scripts/check_data.py --dataset NAME` checks file availability, including
supported raw/processed alternatives. `--dataset all` checks all wired datasets,
including optional MIMIC/P19, so missing optional data is expected on a starter
checkout. `--dataset bundled --verify` checks SHA-256 for every bundled file in
`DATA_MANIFEST.json`. Presence checks do not validate downloaded schemas or
establish reproducibility of an external data release.
