#!/usr/bin/env bash
# Privileged remote runner for E02-08.
#
# Root is required for three separate reasons on this board:
#
#   * `jetson_clocks --show` / `--store` / `--restore` refuse to run as a
#     non-root user, and without them the fixed-vs-dynamic clock contrast
#     cannot be observed or undone.
#   * `nvpmodel -m <id>` changes the whole-module power mode.
#   * the EMC clock (`/sys/kernel/debug/bpmp/debug/clk/emc/rate`) is only
#     readable as root, and EMC frequency is part of the throttle evidence.
#
# `sudo -E` preserves HOME (user site-packages with transformers/modelscope
# stay importable) and the explicit `env` re-adds the variables that sudo's
# env_reset would otherwise drop:
#
#   PYTORCH_NO_CUDA_MEMORY_CACHING=1  required to load the model at all on this
#                                     8 GiB board (E02-06 section 4.13); it is
#                                     the established S02 allocator protocol and
#                                     is recorded in every run record
#   PYTHONPATH                        the repository root
#
# Usage (on the Jetson, normally via ./scripts/remote_run.sh):
#   scripts/audit/e02_08_run.sh probe     --output-dir <dir>
#   scripts/audit/e02_08_run.sh audit     --output-dir <dir>
#   scripts/audit/e02_08_run.sh collect   --output-dir <dir> --run-index 0
#   scripts/audit/e02_08_run.sh verify    --output-dir <dir>
#   scripts/audit/e02_08_run.sh summarize --output-dir <dir>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

exec sudo -E env \
  PYTHONPATH="${PYTHONPATH}" \
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
  python3 "${REPO_ROOT}/scripts/audit/run_e02_08_power_thermal.py" "$@"
