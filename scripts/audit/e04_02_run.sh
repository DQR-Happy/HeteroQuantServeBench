#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
OUT="docs/stage_experiments/S04/E04-02/raw"

mkdir -p "${OUT}"
python3 scripts/audit/run_e04_02_rmsnorm_backend_comparison.py initialize --output-dir "${OUT}"
python3 scripts/audit/run_e04_02_rmsnorm_backend_comparison.py cold --output-dir "${OUT}"
python3 scripts/audit/run_e04_02_rmsnorm_backend_comparison.py layout-stream --output-dir "${OUT}"
python3 scripts/audit/run_e04_02_rmsnorm_backend_comparison.py correctness-matrix --output-dir "${OUT}"
python3 scripts/audit/run_e04_02_rmsnorm_backend_comparison.py collect --output-dir "${OUT}" --process-index 0
python3 scripts/audit/run_e04_02_rmsnorm_backend_comparison.py collect --output-dir "${OUT}" --process-index 1
python3 scripts/audit/run_e04_02_rmsnorm_backend_comparison.py collect --output-dir "${OUT}" --process-index 2
python3 scripts/audit/run_e04_02_rmsnorm_backend_comparison.py summarize --output-dir "${OUT}"
