#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
OUT="docs/stage_experiments/S04/E04-06/raw"
SCRIPT="scripts/audit/run_e04_06_triton_autotune_cache.py"

mkdir -p "${OUT}"
export TRITON_CACHE_DIR="${OUT}/compiler_cache"

python3 "${SCRIPT}" initialize --output-dir "${OUT}"
python3 "${SCRIPT}" cold-import --output-dir "${OUT}"
python3 "${SCRIPT}" transaction --output-dir "${OUT}"
for process in 0 1 2; do
  python3 "${SCRIPT}" collect --output-dir "${OUT}" --split train --process-index "${process}"
done
for process in 0 1 2; do
  python3 "${SCRIPT}" collect --output-dir "${OUT}" --split validation --process-index "${process}"
done
python3 "${SCRIPT}" freeze --output-dir "${OUT}"
python3 "${SCRIPT}" cache-build --output-dir "${OUT}"
python3 "${SCRIPT}" unknown-policy --output-dir "${OUT}"
python3 "${SCRIPT}" cache-hit --output-dir "${OUT}"
python3 "${SCRIPT}" invalidation --output-dir "${OUT}"
python3 "${SCRIPT}" corruption --output-dir "${OUT}"
python3 "${SCRIPT}" concurrency --output-dir "${OUT}"
for process in 0 1 2; do
  python3 "${SCRIPT}" collect --output-dir "${OUT}" --split holdout --process-index "${process}"
done
python3 "${SCRIPT}" summarize --output-dir "${OUT}"
