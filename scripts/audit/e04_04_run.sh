#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
OUT="docs/stage_experiments/S04/E04-04/raw"

mkdir -p "${OUT}"
python3 scripts/audit/run_e04_04_decode_prefill_regime.py initialize --output-dir "${OUT}"
python3 scripts/audit/run_e04_04_decode_prefill_regime.py build --output-dir "${OUT}"
python3 scripts/audit/run_e04_04_decode_prefill_regime.py gemm --output-dir "${OUT}" --process-index 0
python3 scripts/audit/run_e04_04_decode_prefill_regime.py gemm --output-dir "${OUT}" --process-index 1
python3 scripts/audit/run_e04_04_decode_prefill_regime.py gemm --output-dir "${OUT}" --process-index 2
python3 scripts/audit/run_e04_04_decode_prefill_regime.py rmsnorm --output-dir "${OUT}" --process-index 0
python3 scripts/audit/run_e04_04_decode_prefill_regime.py rmsnorm --output-dir "${OUT}" --process-index 1
python3 scripts/audit/run_e04_04_decode_prefill_regime.py rmsnorm --output-dir "${OUT}" --process-index 2
python3 scripts/audit/run_e04_04_decode_prefill_regime.py profile --output-dir "${OUT}"
python3 scripts/audit/run_e04_04_decode_prefill_regime.py summarize --output-dir "${OUT}"
