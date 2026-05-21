#!/usr/bin/env bash
# Privileged remote runner for E02-07.
#
# Two privileges are needed and neither is optional:
#
#   * CUPTI (PyTorch profiler / Nsight Systems / Nsight Compute) refuses to
#     collect CUDA activity for an unprivileged process on this Jetson, which
#     would leave every `cuda_time_us` at zero.
#   * Nsight Compute refuses to launch a target for profiling without root
#     ("Insufficient privileges to launch app for profiling").
#
# `sudo -E` preserves HOME (so the user site-packages with transformers and
# modelscope stay importable) and the explicit `env` re-adds the two variables
# that sudo's env_reset would otherwise drop:
#
#   PYTORCH_NO_CUDA_MEMORY_CACHING=1  required to load the model at all on this
#                                     8 GiB board (E02-06 section 4.13)
#   PYTHONPATH                        the repository root
#
# Usage (on the Jetson, normally via ./scripts/remote_run.sh):
#   scripts/audit/e02_07_run.sh audit   --output-dir <dir>
#   scripts/audit/e02_07_run.sh collect --output-dir <dir> --run-index 0
#   scripts/audit/e02_07_run.sh nsys    --output-dir <dir> --run-index 0
#   scripts/audit/e02_07_run.sh ncu     --output-dir <dir> --run-index 0 --deep
#   scripts/audit/e02_07_run.sh verify  --output-dir <dir>
#   scripts/audit/e02_07_run.sh summarize --output-dir <dir>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

exec sudo -E env \
  PYTHONPATH="${PYTHONPATH}" \
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
  python3 "${REPO_ROOT}/scripts/audit/run_e02_07_multilevel_profiling.py" "$@"
