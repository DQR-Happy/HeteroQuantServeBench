#!/usr/bin/env bash
# E02-07 full collection driver.
#
# Runs the whole E02-07 evidence chain in the order the protocol requires:
#
#   1. audit      - range/annotation rehearsal (already done separately)
#   2. collect    - 3 independent processes: reference + profiled windows
#   3. nsys       - 3 independent processes of Nsight Systems capture
#   4. ncu        - shape-exact replay for every candidate in all 3 processes,
#                   plus the in-model application-replay anchor in process 0
#   5. verify     - cross-process and cross-tool verdict
#   6. summarize  - report tables recomputed from the raw evidence
#
# Everything runs through scripts/audit/e02_07_run.sh, which escalates with
# `sudo -E` because CUPTI and Nsight Compute both refuse to profile an
# unprivileged target on this board.
#
# Usage (on the Jetson, normally via ./scripts/remote_run.sh):
#   nohup ./scripts/bench/e02_07_run_all.sh > /tmp/e02_07_all.log 2>&1 &
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
RAW="docs/stage_experiments/S02/E02-07/raw"
RUNNER="./scripts/audit/e02_07_run.sh"

step() {
  echo ""
  echo "==================== $(date -u +%H:%M:%S) :: $* ===================="
  "$@"
  echo "-------------------- exit=$? :: $* --------------------"
}

mkdir -p "${RAW}"

for i in 0 1 2; do
  step "${RUNNER}" collect --output-dir "${RAW}" --run-index "${i}"
done

for i in 0 1 2; do
  step "${RUNNER}" nsys --output-dir "${RAW}" --run-index "${i}"
done

step "${RUNNER}" ncu --output-dir "${RAW}" --run-index 0 --anchor
for i in 1 2; do
  step "${RUNNER}" ncu --output-dir "${RAW}" --run-index "${i}"
done

step "${RUNNER}" verify --output-dir "${RAW}"
step "${RUNNER}" summarize --output-dir "${RAW}"

echo ""
echo "ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
