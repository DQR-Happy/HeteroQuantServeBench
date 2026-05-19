#!/usr/bin/env bash
# Privileged remote runner for E02-02.
#
# The PyTorch profiler needs CUPTI, which on this Jetson requires elevated
# privileges; without them every `cuda_time_us` is 0 and the per-scope shares
# cannot close to 1.0. `sudo -E` preserves HOME (so the user site-packages with
# transformers/modelscope stay importable) while `env PYTHONPATH=...` re-adds
# the repo root that sudo's env_reset would otherwise drop.
#
# Usage (on the Jetson, typically via ./scripts/remote_run.sh):
#   scripts/audit/e02_02_run.sh audit   --output-dir <dir>
#   scripts/audit/e02_02_run.sh collect --output-dir <dir> --run-index 0
#   scripts/audit/e02_02_run.sh verify  --output-dir <dir>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

exec sudo -E env PYTHONPATH="${PYTHONPATH}" \
  python3 "${REPO_ROOT}/scripts/audit/run_e02_02_shape_census.py" "$@"
