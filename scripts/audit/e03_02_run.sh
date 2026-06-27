#!/usr/bin/env bash
# Remote runner for E03-02 (RMSNorm shape performance heatmap).
#
# Needs a CUDA device and the built shared library, but **no privilege**: it
# never calls ncu, nvpmodel or jetson_clocks (it only *reads* their state, and
# reads the GPU clock from devfreq sysfs). The only tool it executes besides
# python3 is `cuobjdump`, which is a static read of the ELF.
#
# Usage (on the Jetson, normally via ./scripts/remote_run.sh):
#   bash scripts/audit/e03_02_run.sh pilot     --output-dir <dir>
#   bash scripts/audit/e03_02_run.sh collect   --output-dir <dir>   # 3 processes
#   bash scripts/audit/e03_02_run.sh summarize --output-dir <dir>
#   bash scripts/audit/e03_02_run.sh verify    --output-dir <dir>   # exit 0 == PASS
#
# The order matters: `pilot` freezes protocol.json before any timed run, and
# `verify` reads the artifacts written by `summarize`.
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

exec python3 "${REPO_ROOT}/scripts/audit/run_e03_02_shape_heatmap.py" "$@"
