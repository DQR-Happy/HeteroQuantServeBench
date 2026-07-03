#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
OUT="docs/stage_experiments/S04/E04-07/raw"
SCRIPT="scripts/audit/run_e04_07_ir_sass_resource.py"

mkdir -p "${OUT}"
python3 "${SCRIPT}" initialize --output-dir "${OUT}"
for process in 0 1 2; do
  python3 "${SCRIPT}" ordinary-triton --output-dir "${OUT}" --process-index "${process}"
done
for process in 0 1 2; do
  python3 "${SCRIPT}" ordinary-cutlass --output-dir "${OUT}" --process-index "${process}"
done
python3 "${SCRIPT}" profile --output-dir "${OUT}"
python3 "${SCRIPT}" analyze --output-dir "${OUT}"
