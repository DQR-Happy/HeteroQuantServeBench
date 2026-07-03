#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
OUT="docs/stage_experiments/S04/E04-05/raw"

mkdir -p "${OUT}"
python3 scripts/audit/run_e04_05_cutlass_configuration_space.py initialize --output-dir "${OUT}"
python3 scripts/audit/run_e04_05_cutlass_configuration_space.py build --output-dir "${OUT}"
python3 scripts/audit/run_e04_05_cutlass_configuration_space.py status --output-dir "${OUT}"
python3 scripts/audit/run_e04_05_cutlass_configuration_space.py correctness --output-dir "${OUT}"
python3 scripts/audit/run_e04_05_cutlass_configuration_space.py screen --output-dir "${OUT}"
python3 scripts/audit/run_e04_05_cutlass_configuration_space.py freeze --output-dir "${OUT}"
python3 scripts/audit/run_e04_05_cutlass_configuration_space.py precise --output-dir "${OUT}" --process-index 0
python3 scripts/audit/run_e04_05_cutlass_configuration_space.py precise --output-dir "${OUT}" --process-index 1
python3 scripts/audit/run_e04_05_cutlass_configuration_space.py precise --output-dir "${OUT}" --process-index 2
python3 scripts/audit/run_e04_05_cutlass_configuration_space.py safety --output-dir "${OUT}"
python3 scripts/audit/run_e04_05_cutlass_configuration_space.py summarize --output-dir "${OUT}"
python3 scripts/audit/run_e04_05_cutlass_configuration_space.py profile --output-dir "${OUT}"
python3 scripts/audit/run_e04_05_cutlass_configuration_space.py finalize --output-dir "${OUT}"
