#!/usr/bin/env bash
# Remote-only E03-09 dependency gate audit. Dynamic CUDA work is forbidden
# until E02-09 and the ordered E03-08 dispatcher prerequisite are both PASS.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-${REPO_ROOT}/docs/stage_experiments/S03/E03-09/raw}"

mkdir -p "${OUT}"
python3 "${REPO_ROOT}/scripts/audit/run_e03_09_second_hotspot.py" \
  --repo-root "${REPO_ROOT}" \
  --output-dir "${OUT}"
