#!/usr/bin/env bash
# Remote runner for E02-09 (Amdahl / Roofline hotspot decision).
#
# Unlike E02-07/E02-08 this experiment needs **no privilege at all**: it does
# not touch CUDA, the profiler, `nvpmodel` or `jetson_clocks`. It only reads the
# already-stored S02 raw evidence and recomputes shares, ceilings, roofline
# points and the Hotspot Decision Record. The only thing it does need is the
# repository root on `PYTHONPATH`, because `hqsb` is imported as a package
# while the script lives in `scripts/audit/`.
#
# Usage (on the Jetson, normally via ./scripts/remote_run.sh):
#   bash scripts/audit/e02_09_run.sh analyze   --output-dir <dir>
#   bash scripts/audit/e02_09_run.sh verify    --output-dir <dir>
#   bash scripts/audit/e02_09_run.sh summarize --output-dir <dir>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

exec python3 "${REPO_ROOT}/scripts/audit/run_e02_09_hotspot_decision.py" "$@"
