#!/usr/bin/env bash
# E04-01 must execute on the Jetson target; the local Mac is edit-only.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-${REPO_ROOT}/docs/stage_experiments/S04/E04-01/raw}"

mkdir -p "${OUT}"
python3 "${REPO_ROOT}/scripts/audit/run_e04_01_capability_failure.py" run \
  --output-dir "${OUT}"
