#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
OUT="${1:-docs/stage_experiments/S05/E05-02/raw}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

for method in fp16 rtn_w8 rtn_w4; do
  for run_index in 0 1 2; do
    run_file="$OUT/runs/${method}_run_${run_index}.json"
    if [ -f "$run_file" ]; then
      echo "skip existing $run_file"
      continue
    fi
    python3 scripts/audit/run_e05_02_weight_only_model_baseline.py collect \
      --output-dir "$OUT" \
      --method "$method" \
      --run-index "$run_index" \
      --cpu-staging \
      --row-chunk 8
  done
done

python3 scripts/audit/run_e05_02_weight_only_model_baseline.py verify \
  --output-dir "$OUT"
