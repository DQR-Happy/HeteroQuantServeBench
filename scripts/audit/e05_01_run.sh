#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
OUT="${1:-docs/stage_experiments/S05/E05-01/raw}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
exec python3 scripts/audit/run_e05_01_rtn_quant_artifact.py --output-dir "${OUT}"
