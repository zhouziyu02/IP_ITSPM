#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU="${GPU:-0}"
SEED="${SEED:-1}"
if [[ "$GPU" == "-1" ]]; then
  DEVICE="cpu"
else
  DEVICE="cuda:${GPU}"
fi

exec "${PYTHON:-python3}" \
  "$ROOT/TPP/EasyTemporalPointProcess/examples/train_itspm.py" \
  --device "$DEVICE" --seed "$SEED" \
  --output-dir "${OUTPUT_DIR:-$ROOT/outputs/tpp_taxi}" "$@"
