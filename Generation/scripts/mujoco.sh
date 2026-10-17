#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO/Generation/itspm_generation"

GPU="${GPU:-0}"
SEED="${SEED:-1}"
N_SAMPLES="${N_SAMPLES:-10000}"
DS_ITERATIONS="${DS_ITERATIONS:-2000}"
DATA_ROOT="${DATA_ROOT:-table1_data}"
SEQ_LENS="${SEQ_LENS:-36}"
MISSING_RATIOS="${MISSING_RATIOS:-0.5}"

# Only length 36 is bundled. Prepare other lengths explicitly before running.
for seq_len in $SEQ_LENS; do
  case "$seq_len" in
    12|24|36) ;;
    *) printf 'Unsupported MuJoCo length: %s (choose 12, 24, or 36).\n' "$seq_len" >&2; exit 2 ;;
  esac
  if [[ ! -f "$DATA_ROOT/mujoco_training_${seq_len}.pt" ]]; then
    printf 'Missing real MuJoCo data: %s/mujoco_training_%s.pt\n' "$DATA_ROOT" "$seq_len" >&2
    printf 'From the repository root run: python Generation/scripts/install_physics.py\n' >&2
    printf 'Then: python Generation/scripts/prepare_mujoco.py --lengths %s\n' "$seq_len" >&2
    exit 2
  fi
done

for seq_len in $SEQ_LENS; do
  for missing in $MISSING_RATIOS; do
    out_dir="outputs/mujoco/len${seq_len}_miss${missing}_seed${SEED}"
    python3 train_irregular_generation.py \
      --gpu "$GPU" --seed "$SEED" --dataset mujoco \
      --generator itspm_latent_diffusion --data_root "$DATA_ROOT" \
      --n_samples "$N_SAMPLES" --seq_len "$seq_len" --missing "$missing" \
      --batch_size 128 --diff_epochs 200 --diffusion_steps 100 --diffusion_hidden 256 \
      --d_model 96 --n_ref_points 32 --max_event_tokens 64 --max_gap_tokens 32 \
      --ds_iterations "$DS_ITERATIONS" --ds_batch_size 128 \
      --out_dir "$out_dir" "$@"
  done
done
