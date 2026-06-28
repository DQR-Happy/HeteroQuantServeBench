#!/usr/bin/env bash
# Remote-only orchestration for E03-04.  Usage:
#   bash scripts/audit/e03_04_run.sh [output-dir]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-${REPO_ROOT}/docs/stage_experiments/S03/E03-04/raw}"
RUNNER="${REPO_ROOT}/scripts/audit/run_e03_04_launch_ncu.py"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

LIB=""
for candidate in "${REPO_ROOT}"/build/*/ops/cuda/rmsnorm/libhqsb_rmsnorm_shared.so; do
  if [[ -f "${candidate}" ]]; then LIB="${candidate}"; break; fi
done
if [[ -z "${LIB}" ]]; then
  echo "RMSNorm shared library not found" >&2
  exit 2
fi

python3 "${RUNNER}" plan --output-dir "${OUT}"
for process_index in 0 1 2; do
  python3 "${RUNNER}" collect --output-dir "${OUT}" --process-index "${process_index}" --library "${LIB}"
done
# First summary resolves the pre-registered best/worst selection rules.
python3 "${RUNNER}" summarize --output-dir "${OUT}" || true
python3 "${RUNNER}" ncu --output-dir "${OUT}" --library "${LIB}"
python3 "${RUNNER}" summarize --output-dir "${OUT}"
python3 "${RUNNER}" verify --output-dir "${OUT}"
