#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
OUT="docs/stage_experiments/S04/E04-03/raw"

mkdir -p "${OUT}"
python3 scripts/audit/run_e04_03_gemm_backend_tail.py initialize --output-dir "${OUT}"
python3 scripts/audit/run_e04_03_gemm_backend_tail.py build-cutlass --output-dir "${OUT}"
python3 scripts/audit/run_e04_03_gemm_backend_tail.py correctness --output-dir "${OUT}"
python3 scripts/audit/run_e04_03_gemm_backend_tail.py cutlass --output-dir "${OUT}"
python3 scripts/audit/run_e04_03_gemm_backend_tail.py invalid --output-dir "${OUT}"
python3 scripts/audit/run_e04_03_gemm_backend_tail.py cold --output-dir "${OUT}"
python3 scripts/audit/run_e04_03_gemm_backend_tail.py sanitizer --output-dir "${OUT}"
python3 scripts/audit/run_e04_03_gemm_backend_tail.py performance --output-dir "${OUT}" --process-index 0
python3 scripts/audit/run_e04_03_gemm_backend_tail.py performance --output-dir "${OUT}" --process-index 1
python3 scripts/audit/run_e04_03_gemm_backend_tail.py performance --output-dir "${OUT}" --process-index 2
python3 scripts/audit/run_e04_03_gemm_backend_tail.py summarize --output-dir "${OUT}"
