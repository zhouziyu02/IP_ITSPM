#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO/Generation/itspm_generation"

GPU="${GPU:-0}"
SEED="${SEED:-1}"
N_SAMPLES="${N_SAMPLES:-10000}"
DS_ITERATIONS="${DS_ITERATIONS:-2000}"
DATA_ROOT="${DATA_ROOT:-table1_data}"

if [[ ! -f "$DATA_ROOT/stock_data.csv" ]]; then
  printf 'Missing real Stocks dataset: %s/stock_data.csv\n' "$DATA_ROOT" >&2
  printf 'Provide the Diff-MN numeric CSV with one header row; no synthetic substitute is generated.\n' >&2
  exit 2
fi

SEQ_LENS="${SEQ_LENS:-12 24 36}"
MISSING_RATIOS="${MISSING_RATIOS:-0.3 0.5 0.7}"

for seq_len in $SEQ_LENS; do
  for missing in $MISSING_RATIOS; do
    out_dir="outputs/stocks/len${seq_len}_miss${missing}_seed${SEED}"
    python3 train_irregular_generation.py \
      --gpu "$GPU" --seed "$SEED" --dataset stocks \
      --generator itspm_latent_diffusion --data_root "$DATA_ROOT" \
      --n_samples "$N_SAMPLES" --seq_len "$seq_len" --missing "$missing" \
      --batch_size 128 --diff_epochs 200 --diffusion_steps 100 --diffusion_hidden 256 \
      --d_model 96 --n_ref_points 32 --max_event_tokens 64 --max_gap_tokens 32 \
      --ds_iterations "$DS_ITERATIONS" --ds_batch_size 128 \
      --out_dir "$out_dir" "$@"
  done
done
