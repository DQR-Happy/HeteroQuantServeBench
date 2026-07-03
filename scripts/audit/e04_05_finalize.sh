#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
OUT="docs/stage_experiments/S04/E04-05/raw"

python3 scripts/audit/run_e04_05_cutlass_configuration_space.py safety --output-dir "${OUT}"
python3 scripts/audit/run_e04_05_cutlass_configuration_space.py summarize --output-dir "${OUT}"
python3 scripts/audit/run_e04_05_cutlass_configuration_space.py profile --output-dir "${OUT}"
python3 scripts/audit/run_e04_05_cutlass_configuration_space.py finalize --output-dir "${OUT}"
