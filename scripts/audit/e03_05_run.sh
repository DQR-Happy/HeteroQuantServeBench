#!/usr/bin/env bash
# Remote-only orchestration for E03-05.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-${REPO_ROOT}/docs/stage_experiments/S03/E03-05/raw}"
RUNNER="${REPO_ROOT}/scripts/audit/run_e03_05_fused_residual_rmsnorm.py"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

LIB=""
for candidate in "${REPO_ROOT}"/build/*/ops/cuda/fused_residual_rmsnorm/libhqsb_fused_residual_rmsnorm_shared.so; do
  if [[ -f "${candidate}" ]]; then LIB="${candidate}"; break; fi
done
if [[ -z "${LIB}" ]]; then
  echo "fused residual RMSNorm shared library not found" >&2
  exit 2
fi

python3 "${RUNNER}" plan --output-dir "${OUT}"
python3 "${RUNNER}" negative --output-dir "${OUT}" --library "${LIB}"
for process_index in 0 1 2; do
  python3 "${RUNNER}" collect --output-dir "${OUT}" --process-index "${process_index}" --library "${LIB}"
done
# Jetson locks CUPTI performance counters for unprivileged users.  Timeline
# capture is diagnostic/read-only; use the same non-interactive profiler
# privilege as NCU, then restore artifact ownership for ordinary reruns.
sudo -n -E python3 "${RUNNER}" timeline --output-dir "${OUT}" --library "${LIB}"
sudo -n chown -R "$(id -u):$(id -g)" "${OUT}/timeline"
python3 "${RUNNER}" ncu --output-dir "${OUT}" --library "${LIB}"
python3 "${RUNNER}" summarize --output-dir "${OUT}"
python3 "${RUNNER}" verify --output-dir "${OUT}"
