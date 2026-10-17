#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO/Generation/itspm_generation"

GPU="${GPU:-0}"
SEED="${SEED:-1}"
N_SAMPLES="${N_SAMPLES:-10000}"
DS_ITERATIONS="${DS_ITERATIONS:-2000}"

SEQ_LENS="${SEQ_LENS:-12 24 36}"
MISSING_RATIOS="${MISSING_RATIOS:-0.3 0.5 0.7}"

for seq_len in $SEQ_LENS; do
  for missing in $MISSING_RATIOS; do
    out_dir="outputs/sines/len${seq_len}_miss${missing}_seed${SEED}"
    python3 train_irregular_generation.py \
      --gpu "$GPU" --seed "$SEED" --dataset sines \
      --generator itspm_latent_diffusion --data_root table1_data \
      --n_samples "$N_SAMPLES" --seq_len "$seq_len" --channels 5 --missing "$missing" \
      --batch_size 128 --diff_epochs 120 --diffusion_steps 50 --diffusion_hidden 192 \
      --d_model 64 --n_ref_points 24 --max_event_tokens 48 --max_gap_tokens 24 \
      --ds_iterations "$DS_ITERATIONS" --ds_batch_size 128 \
      --out_dir "$out_dir" "$@"
  done
done
