#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

GPU="${GPU:-0}"
SEED="${SEED:-26}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"

python3 -u classification.py \
  --model ITSPM --gpu "$GPU" --task P12 --seed "$SEED" \
  --epoch 500 --patience 80 --sample_rate 1.0 --batch_size 32 \
  --d_model 128 --dropout 0.18 --lr 4e-4 --weight_decay 1e-5 \
  --n_ref_points 48 --n_scales 4 --n_mixer_layers 3 --collate indseq \
  --split 1 --select_metric auprc --use_static --no_auto_class_weights \
  --positive_weight 3.0 --disable_cls_fusion \
  --state "itspm_p12_best_seed${SEED}_${RUN_ID}" "$@"
