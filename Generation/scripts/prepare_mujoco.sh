#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

if [[ $# -eq 0 ]]; then set -- 36; fi
python3 Generation/scripts/install_physics.py
python3 Generation/scripts/prepare_mujoco.py --lengths "$@"
