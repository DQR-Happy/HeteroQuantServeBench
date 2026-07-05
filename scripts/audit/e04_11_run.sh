#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../.."
OUT="${1:-docs/stage_experiments/S04/E04-11/raw}"
RUNNER="scripts/audit/run_e04_11_tilelang_hip_portability.py"
export TILELANG_CACHE_DIR="$(pwd)/${OUT}/tilelang_cache"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p "${OUT}" "${TILELANG_CACHE_DIR}"

python3 "${RUNNER}" initialize --output-dir "${OUT}" || exit $?

collection_rc=0
for process in 0 1 2; do
  python3 "${RUNNER}" collect --output-dir "${OUT}" \
    --process-index "${process}" || collection_rc=$?
done

python3 "${RUNNER}" profile --output-dir "${OUT}"
profile_rc=$?
sudo -n chown -R "$(id -u):$(id -g)" "${OUT}" 2>/dev/null || true

python3 "${RUNNER}" analyze --output-dir "${OUT}"
analysis_rc=$?

if [[ ${collection_rc} -ne 0 ]]; then
  exit "${collection_rc}"
fi
if [[ ${profile_rc} -ne 0 ]]; then
  exit "${profile_rc}"
fi
exit "${analysis_rc}"
