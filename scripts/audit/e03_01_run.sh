#!/usr/bin/env bash
# Remote runner for E03-01 (RMSNorm semantics / dual oracle / correctness boundary).
#
# Needs a CUDA device and the built shared library, but **no privilege**:
# it never calls ncu, nvpmodel or jetson_clocks. It only loads
# ``libhqsb_rmsnorm_shared.so`` through ctypes and runs the pre-registered
# correctness matrix.
#
# Usage (on the Jetson, normally via ./scripts/remote_run.sh):
#   bash scripts/audit/e03_01_run.sh collect   --output-dir <dir>
#   bash scripts/audit/e03_01_run.sh verify    --output-dir <dir>
#   bash scripts/audit/e03_01_run.sh summarize --output-dir <dir>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Locate the freshly built shared library (the operator under test).
if [[ -z "${HQSB_CUDA_RMSNORM_LIB:-}" ]]; then
  for candidate in "${REPO_ROOT}"/build/*/ops/cuda/rmsnorm/libhqsb_rmsnorm_shared.so; do
    if [[ -f "${candidate}" ]]; then
      export HQSB_CUDA_RMSNORM_LIB="${candidate}"
      break
    fi
  done
fi

exec python3 "${REPO_ROOT}/scripts/audit/run_e03_01_rmsnorm_semantics.py" "$@"
