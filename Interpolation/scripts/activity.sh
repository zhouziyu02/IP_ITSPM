#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

GPU="${GPU:-0}"
SEED="${SEED:-15}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"

python3 -u regression.py \
  --model ITSPM --gpu "$GPU" --dataset activity --task imputation --seed "$SEED" \
  --epoch 1000 --patience 10 --batch_size 32 --d_model 128 \
  --dropout 0.05 --lr 5e-4 --weight_decay 1e-5 \
  --n_ref_points 64 --n_scales 4 --n_mixer_layers 3 --history 3000 \
  --state "itspm_interpolation_activity_best_seed${SEED}_${RUN_ID}" "$@"
