#!/usr/bin/env bash
# Remote-only runner for E03-03 (dtype/alignment/layout/vector-load/tail).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -z "${HQSB_CUDA_RMSNORM_LIB:-}" ]]; then
  for candidate in "${REPO_ROOT}"/build/*/ops/cuda/rmsnorm/libhqsb_rmsnorm_shared.so; do
    if [[ -f "${candidate}" ]]; then
      export HQSB_CUDA_RMSNORM_LIB="${candidate}"
      break
    fi
  done
fi

exec python3 "${REPO_ROOT}/scripts/audit/run_e03_03_dtype_alignment.py" "$@"
