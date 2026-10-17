# MSc IP project briefs

These briefs follow the supplied ITSPM project plan. Each student owns one
downstream task. Use small-to-medium public datasets and a single GPU, start with
small pipeline checks, and report mean ± sample standard deviation over three
explicit seeds. Use matched splits, preprocessing, training budgets, and target
masks across methods; save the commit, command, environment, split IDs, seeds,
checkpoints, metrics, and plots needed to reproduce each table.

The five existing trainers provide a starting point. They do not implement all
baselines or ablations below. Reference results are historical context, with
limitations explained in [REFERENCE_RESULTS.md](REFERENCE_RESULTS.md). Data
sources and formats are in [DATASETS.md](DATASETS.md).

## 1.1 Irregular time series classification

**Question:** Which sparse pattern tokens help distinguish classes, and how
robust are representations to missing observations or entire variables?

**Start:** `classification.py`, `lib/Dataset_MM.py`,
`models/ITSPM.py::FastSparsePatternTokenizer`. Datasets: P12 and PAM.

1. Reproduce classification with the provided presets. Report P12 AUROC/AUPRC
   and PAM accuracy/macro-F1 over three seeds; compare with historical references
   only after checking the preprocessing and selection protocol.
2. Train two or three lightweight baselines, such as GRU-D, mTAND, and a GRU
   with mask/time-delta inputs, using the same splits. Compare in one table.
3. Remove event, gap, and variable-summary tokens one at a time. At test time
   remove 10%–50% of observations or entire variables; plot degradation curves.
4. Plot t-SNE/UMAP representations and compare a baseline. Inspect variable
   contributions and discuss agreement with domain knowledge. The model uses
   gated pooling and query attention; attention weights are not an automatic
   causal importance measure.

## 1.2 Irregular time series interpolation

**Question:** How do continuous-time event features help reconstruct missing
values, especially inside long gaps or at low observation density?

**Start:** `regression.py --task imputation`, `lib/parse_datasets.py::task_mask`,
`models/ITSPM.py::EventEncoder`. Datasets: P12 and USHCN.

1. Train under the existing deterministic masked-time-row protocol; report
   MSE/MAE over three seeds. Keep context and target sets disjoint.
2. Compare linear interpolation, last observation carried forward, and
   per-variable training mean; add one or two neural baselines such as mTAND
   or GRU-D with an appropriate interpolation head.
3. Remove timestamp encoding, observation-status embedding, and elapsed-time
   feature separately. Vary context ratios from 10%–90% and embedding dimension;
   keep evaluation targets fixed for a clean density comparison.
4. Plot ground truth/context/reconstructions, mark long gaps, and stratify errors
   by gap length and per-variable sampling frequency.

Removing the learned observation-status embedding does not mean removing the
mask used to exclude missing values from losses and pooling.

## 1.3 Irregular time series extrapolation

**Question:** When do soft temporal patterns improve future prediction, and how
does their usefulness depend on horizon and sampling characteristics?

**Start:** `regression.py --task forecasting`, `lib/parse_datasets.py`,
`models/ITSPM.py::KernelSummaryBranch` and `_local_kernel_readout`.
Datasets: two or three of P12, Human Activity, and USHCN.

1. Train with the provided observation/forecast windows and report MSE/MAE over
   three seeds, recording time units and normalization.
2. Compare last-value persistence, per-variable linear extrapolation, and
   lightweight GRU-D/mTAND forecasting adaptations.
3. Remove the soft kernel branch; define whether this removes the summary
   branch, local kernel readout, or both. Add an explicit temporal-scale control
   and vary scale counts/bandwidths. Sweep history length and forecast horizon
   and plot error versus horizon. `n_scales` currently controls attention heads,
   so it cannot serve as the requested kernel-scale ablation.
4. Plot history/forecasts/targets and kernel shapes/bandwidths. Relate results to
   sampling rate or periodicity and show representative failures.

## 1.4 Temporal point process modeling

**Question:** Can observation timing and variable identity be modeled as a
marked point process, and what do gap tokens and elapsed time contribute?

**Start:** `TPP/scripts/taxi.sh`,
`TPP/EasyTemporalPointProcess/easy_tpp/model/torch_model/torch_itspm.py`.
Datasets: Taxi plus StackOverflow and/or P12 observation events. Only Taxi is
currently connected to the launcher; add other dataset configurations/adapters.

1. Train on two or three datasets and report test log-likelihood, next-event
   time RMSE, and event-type accuracy over three seeds. Use validation
   log-likelihood for model selection and record event/time normalization.
2. Use upstream EasyTPP to run RMTPP, NHP, and THP on the same event splits.
   Discuss differences between likelihood and prediction quality.
3. Ablate gap tokens and elapsed-time features. Vary hidden size and maximum
   context/sequence length; measure training time and peak GPU memory under
   comparable conditions.
4. Plot conditional intensities with observed events. Integrate intensities for
   time-rescaling QQ diagnostics and compare predicted/empirical inter-event
   distributions. These diagnostic plots need implementation.

Never construct a TPP history from future events. Resolve ties and zero gaps
explicitly when converting simultaneous P12 measurements.

## 1.5 Irregular time series generation

**Question:** Does pattern interaction preserve temporal and cross-variable
structure in synthetic sequences under missing observations?

**Start:** `Generation/itspm_generation/train_irregular_generation.py`,
`models/ITSPM.py::PatternInteraction`. Use two datasets, preferably Stocks and
Energy; Sines/MuJoCo are available starters. P12 needs a new adapter.

1. Train on two datasets at two or more missing rates (for example, 0.3/0.5/0.7).
   Report discriminative score, predictive score (train on synthetic, test on
   real), and MMD or context-FID over three runs. The current trainer provides
   DS/MDD/KL/RBF-MMD and an ACF diagnostic; add the predictive score, and
   context-FID if chosen. Keep a fixed evaluator protocol and budget across generators.
2. Compare a GRU VAE and/or reduced-budget TimeGAN under matched inputs and
   training budgets; report stability and cost as well as quality.
3. Ablate type-level, variable-level, and global interaction/aggregation.
   Specify all affected information paths. Sweep missing rate and sequence
   length and plot each metric.
4. Plot real/synthetic PCA or t-SNE, per-variable density estimates,
   autocorrelation, and cross-variable correlation matrices; discuss losses of
   statistical or temporal structure.

The existing sampler is unconditional latent diffusion after observed-entry
autoencoding. Synthetic data quality alone does not establish privacy. The
complete-data Gaussian reference is not a matched missing-data baseline.

## 1.6 Early classification

**Question:** How early can useful labels be predicted, and does random-prefix
training improve the quality/time trade-off?

**Start:** Extend classification and raw-record preprocessing on P12 and PAM.

1. Define an observation window before experiments and evaluate prefixes at
   20%, 40%, 60%, 80%, and 100%. Report AUROC/AUPRC (P12) and accuracy/macro-F1
   (PAM), three seeds. Truncate records before tokenization, summary statistics,
   and any time-range computation so future observations cannot leak in.
2. Compare GRU-D and a GRU with mask/time-delta inputs on identical prefixes.
   Summarize accuracy/time trade-offs in a table and curves.
3. Compare full-window and random-prefix training under matched budgets. On
   one dataset, remove an additional 20%/40% of available observations.
4. Plot class probabilities as observations arrive, including stable, late
   corrected, and persistently incorrect examples. Stratify performance by
   class and observation density.

## 1.7 Anomaly detection under irregular observations

**Question:** Can standardized forecast errors detect unusual behavior when
historical observations are incomplete?

**Start:** Extend forecasting with SMD/SMAP entity loaders and residual scores.

1. Predefine one or two entities per benchmark, train on normal training
   sequences using short windows, and thin only historical inputs. Keep target
   values and labels fixed; report pointwise AUPRC/F1 over three seeds.
2. Compare persistence and a GRU with mask/time-delta inputs. Apply identical
   residual scaling and scoring. Choose thresholds from held-out normal
   validation data at a predefined false-alarm target; report achieved test FPR.
3. Compare mean/max aggregation of per-variable errors. On one entity per
   dataset, compare 0%/30%/50% thinning and random versus contiguous missingness.
4. Plot observations, forecasts, scores, anomaly intervals, and per-variable
   contributions. Analyze false alarms and misses by missingness/duration.

Use pointwise metrics **without point adjustment**. Do not tune thresholds or
residual normalization on test anomaly labels.

## 1.8 Prediction intervals and uncertainty calibration

**Question:** Are forecast intervals calibrated across variables, horizons, and
observation densities without becoming excessively wide?

**Start:** Extend P12/USHCN forecasting evaluation; reuse trained checkpoints.

1. Reserve a disjoint calibration subset respecting identities/time ordering.
   Construct intervals from per-variable absolute-residual quantiles. Report
   empirical coverage, mean width, and interval score at 80%/90%/95% nominal
   coverage, three seeds.
2. Compare residual-quantile and Gaussian-residual intervals around the same
   ITSPM forecasts. Train one GRU with mask/time-delta inputs and apply the same
   calibration procedure; compare point errors and interval metrics.
3. Compare pooled-horizon and horizon-specific calibration. On one dataset,
   vary calibration size and observation density, reusing forecast checkpoints.
4. Plot trajectories with intervals, nominal versus empirical coverage, and
   results by variable/horizon/density; analyze overconfidence and excess width.

Define quantile conventions and sparse-variable fallback rules in advance.
Temporal dependence and distribution shift can limit exchangeability-based
coverage claims; distinguish empirical coverage from a formal guarantee.

## Implementation notes for ablations

An ablation must remove the intended information, not just change its name or
token budget. The shared model has parallel sparse-token, kernel-summary,
classification-fusion, and local-readout paths. `PatternInteraction` assumes
variable-summary tokens occupy the final `C` positions; removing/reordering
those tokens needs corresponding interaction/readout changes. Simply masking
them in type pooling does not remove the separately sliced variable states.
Similarly, disabling `token_mixer` alone does not remove all interaction levels.

Implement explicit flags and document the changed computation for your task.
Keep the mask defining valid observations/losses even when removing a learned
mask feature. Check finite losses/gradients, all-missing variables, checkpoint
reloads, and prediction-target isolation before full runs. Prefix and horizon
changes should not silently alter the evaluation population.
