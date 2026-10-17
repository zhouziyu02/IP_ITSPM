#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/itspm-matplotlib}"
"$PYTHON" scripts/check_data.py --dataset bundled --verify
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" -m unittest discover -s Generation/tests -v
"$PYTHON" -m unittest discover -s TPP/EasyTemporalPointProcess/tests -v
