#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
OUT="${1:-docs/stage_experiments/S05/E05-03/raw}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python3 scripts/audit/run_e05_03_calibration_generalization.py spec-data \
  --output-dir "$OUT"
python3 scripts/audit/run_e05_03_calibration_generalization.py collect-stats \
  --output-dir "$OUT"
python3 scripts/audit/run_e05_03_calibration_generalization.py build-artifacts \
  --output-dir "$OUT"
python3 scripts/audit/run_e05_03_calibration_generalization.py evaluate \
  --output-dir "$OUT"
python3 scripts/audit/run_e05_03_calibration_generalization.py verify \
  --output-dir "$OUT"
