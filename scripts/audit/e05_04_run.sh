#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
OUT="${1:-docs/stage_experiments/S05/E05-04/raw}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python3 scripts/audit/run_e05_04_industrial_method_adapters.py collect \
  --output-dir "$OUT"
python3 scripts/audit/run_e05_04_industrial_method_adapters.py verify \
  --output-dir "$OUT"
