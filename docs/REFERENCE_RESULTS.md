# Historical reference results

These numbers were transcribed from the supervisor-provided reference checkout's
`reproduce_2gpu_results_20260701_093340.txt` (run dated 2026-07-01). They are
**single-seed historical observations**, not three-seed estimates and not newly
reproduced results for this repository. Machine paths, process logs, duplicate
trial results, and per-student identifiers are intentionally omitted.

The current source incorporates later fixes to normalization, masks, metrics,
checkpoint loading, and TPP pooling. The original checkpoints, dataset hashes,
and full environment for that historical run are not supplied here. Differences
from these values must therefore be investigated, rather than presented as a
failed exact reproduction. Establish a current three-seed reference on the same
protocol before evaluating your extensions.

## Classification

| Dataset | Seed | Historical test metrics |
| --- | ---: | --- |
| P12 | 26 | AUROC 0.87329; AUPRC 0.53458 |
| PAM | 41 | Accuracy 0.97004; macro-F1 0.97414 |
| P19 (optional) | 88 | AUROC 0.92173; AUPRC 0.63064 |

P12 used `d_model=128`, 48 reference points, three mixer layers, learning rate
0.0004, batch size 32, static inputs, positive weight 3.0, and disabled
classification fusion. PAM used 64 reference points, learning rate 0.0005, and
batch size 32. The training presets retain these configurations and historical
default seeds. Set three explicit seeds for your report.

## Interpolation and extrapolation

| Task | Dataset | Seed | MSE | MAE |
| --- | --- | ---: | ---: | ---: |
| Interpolation | P12 / physionet | 44 | 0.00446 | 0.03296 |
| Interpolation | Activity | 15 | 0.00176 | 0.02282 |
| Interpolation | USHCN | 35 | 0.51299 | 0.28596 |
| Interpolation | MIMIC (optional) | 7 | 0.01139 | 0.05529 |
| Extrapolation | P12 / physionet | 24 | 0.00509 | 0.03656 |
| Extrapolation | Activity | 70 | 0.00255 | 0.03069 |
| Extrapolation | USHCN | 44 | 0.50692 | 0.29238 |
| Extrapolation | MIMIC (optional) | 3 | 0.01645 | 0.07048 |

The historical scripts used hidden dimension 128, 64 reference points, three
mixer layers, learning rate 0.0005, and batch size 32. Errors depend on the
dataset's normalization and masking/window protocol; do not compare these
numbers with raw-unit errors or different splits as if they were equivalent.

## Temporal point process

**There is no valid measured TPP result in that historical summary.** Its Taxi,
Taobao, Amazon, and Retweet rows contain `NA` metrics. The reference checkout
documents a missing OmegaConf dependency; the old summary's overall `OK` flag
does not establish successful TPP training. Its `ExpectedRMSE` constants are
not measurements and are not reference scores for this release.

The current Taxi trainer selects a checkpoint by validation log-likelihood,
reloads it, evaluates test once, and writes `test_metrics.json`. Establish new
three-seed Taxi results. Additional datasets require configuration/adapter work.

## Generation

The historical summary reports seed 1 with 10,000 requested samples, sequence
lengths 12/24/36, missing rates 0.3/0.5/0.7, and 2,000 discriminative-evaluator
iterations. Values below were aggregated across lengths in that summary.

| Dataset | Missing rate | Historical DS | Historical MDD | Historical KL |
| --- | ---: | ---: | ---: | ---: |
| Sines | 0.3 | 0.0156 | 0.4108 | 0.0056 |
| Sines | 0.5 | 0.0168 | 0.4671 | 0.0084 |
| Sines | 0.7 | 0.0156 | 0.5183 | 0.0161 |
| Stocks | 0.3 | 0.1101 | 0.1161 | 0.0380 |
| Stocks | 0.5 | 0.1041 | 0.1614 | 0.0683 |
| Stocks | 0.7 | 0.0626 | 0.2155 | 0.1540 |
| Energy | 0.3 | 0.0721 | 0.1162 | 0.0089 |
| Energy | 0.5 | 0.1217 | 0.1537 | 0.0117 |
| Energy | 0.7 | 0.1582 | 0.2233 | 0.0197 |
| MuJoCo | 0.3 | 0.0195 | 0.1902 | 0.0005 |
| MuJoCo | 0.5 | 0.0270 | 0.2325 | 0.0011 |
| MuJoCo | 0.7 | 0.0652 | 0.2989 | 0.0035 |

The current generator uses observed-entry autoencoding followed by latent
diffusion. Its head/checkpoints and regenerated MuJoCo data are not established
as identical to those used for this table. These values therefore provide
historical context only. MDD/KL do not replace the predictive and distribution
metrics requested in project 1.5. Diff-MN constants in the trainer are copied
literature comparisons, not reproduced baseline experiments.

There are no supplied reference results for early classification, anomaly
detection, or prediction intervals. Their first milestone establishes the
evaluation protocol and current baseline results.
