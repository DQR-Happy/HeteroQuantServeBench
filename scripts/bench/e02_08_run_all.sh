#!/usr/bin/env bash
# E02-08 full collection driver.
#
# Runs the whole evidence chain in the order the protocol requires:
#
#   1. probe      - record/protect the raw device state, enumerate the rails
#                   actually emitted and the power modes actually supported
#   2. audit      - the monitor's own correctness tests (parser counterexamples,
#                   hand-computed trapezoid oracle) plus an idle -> GPU -> idle
#                   coverage/alignment rehearsal; this is a gate
#   3. collect    - 3 independent processes, each walking a Latin-square
#                   rotation of the three mode blocks
#   4. verify     - cross-run / cross-mode verdict
#   5. summarize  - report tables recomputed from the raw evidence
#
# Every step runs through scripts/audit/e02_08_run.sh, which escalates with
# `sudo -E` because `jetson_clocks`, `nvpmodel -m` and the EMC rate node all
# need root.
#
# Usage (on the Jetson, normally via ./scripts/remote_run.sh):
#   nohup ./scripts/bench/e02_08_run_all.sh > /tmp/e02_08_all.log 2>&1 &
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
RAW="docs/stage_experiments/S02/E02-08/raw"
RUNNER="./scripts/audit/e02_08_run.sh"

step() {
  echo ""
  echo "==================== $(date -u +%H:%M:%S) :: $* ===================="
  "$@"
  echo "-------------------- exit=$? :: $* --------------------"
}

mkdir -p "${RAW}"

step "${RUNNER}" probe --output-dir "${RAW}"
step "${RUNNER}" audit --output-dir "${RAW}" || echo "WARN: audit gate failed"

for i in 0 1 2; do
  if [ -f "${RAW}/run_${i}.json" ]; then
    echo "skip: ${RAW}/run_${i}.json already exists"
    continue
  fi
  step "${RUNNER}" collect --output-dir "${RAW}" --run-index "${i}"
done

step "${RUNNER}" verify --output-dir "${RAW}"
step "${RUNNER}" summarize --output-dir "${RAW}"

echo ""
echo "ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
