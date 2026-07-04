#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
OUT="docs/stage_experiments/S04/E04-08/raw"
SCRIPT="scripts/audit/run_e04_08_dispatcher_forced_auto_replay.py"

mkdir -p "${OUT}"
python3 "${SCRIPT}" initialize --output-dir "${OUT}"
for process in 0 1 2; do
  python3 "${SCRIPT}" runtime --output-dir "${OUT}" --process-index "${process}"
done
python3 "${SCRIPT}" analyze --output-dir "${OUT}"
