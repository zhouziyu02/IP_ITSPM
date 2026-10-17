#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

GPU="${GPU:-0}"
SEED="${SEED:-41}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"

python3 -u classification.py \
  --model ITSPM --gpu "$GPU" --task PAM --seed "$SEED" \
  --epoch 170 --patience 40 --sample_rate 1.0 --batch_size 32 \
  --d_model 128 --dropout 0.15 --lr 5e-4 --weight_decay 1e-5 \
  --n_ref_points 64 --n_scales 4 --n_mixer_layers 3 --collate indseq \
  --split 1 --select_metric auprc \
  --state "itspm_pam_best_seed${SEED}_${RUN_ID}" "$@"
