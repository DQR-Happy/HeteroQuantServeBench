#!/usr/bin/env bash
# Remote-only E03-08 gate audit.  The Python driver refuses to execute any
# dispatcher case while E03-06/E03-07 are not PASS.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-${REPO_ROOT}/docs/stage_experiments/S03/E03-08/raw}"

mkdir -p "${OUT}"
python3 "${REPO_ROOT}/scripts/audit/run_e03_08_dispatcher.py" \
  --repo-root "${REPO_ROOT}" \
  --output-dir "${OUT}"
