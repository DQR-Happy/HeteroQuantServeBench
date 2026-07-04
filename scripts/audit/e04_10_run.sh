#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
OUT="${1:-docs/stage_experiments/S04/E04-10/raw}"

mkdir -p "${OUT}"
python3 scripts/audit/run_e04_10_cross_architecture_transfer.py \
  collect --output-dir "${OUT}"
