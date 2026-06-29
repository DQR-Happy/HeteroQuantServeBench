#!/usr/bin/env bash
# Remote-only orchestration for E03-06.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-${REPO_ROOT}/docs/stage_experiments/S03/E03-06/raw}"
RUNNER="${REPO_ROOT}/scripts/audit/run_e03_06_stream_semantics.py"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

RMS_LIB=""
for candidate in "${REPO_ROOT}"/build/*/ops/cuda/rmsnorm/libhqsb_rmsnorm_shared.so; do
  if [[ -f "${candidate}" ]]; then RMS_LIB="${candidate}"; break; fi
done
FUSED_LIB=""
for candidate in "${REPO_ROOT}"/build/*/ops/cuda/fused_residual_rmsnorm/libhqsb_fused_residual_rmsnorm_shared.so; do
  if [[ -f "${candidate}" ]]; then FUSED_LIB="${candidate}"; break; fi
done
if [[ -z "${RMS_LIB}" || -z "${FUSED_LIB}" ]]; then
  echo "required RMSNorm shared libraries not found" >&2
  exit 2
fi

sudo -n -E python3 "${RUNNER}" collect --output-dir "${OUT}" \
  --rms-library "${RMS_LIB}" --fused-library "${FUSED_LIB}"
sudo -n chown -R "$(id -u):$(id -g)" "${OUT}"
python3 "${RUNNER}" summarize --output-dir "${OUT}"
