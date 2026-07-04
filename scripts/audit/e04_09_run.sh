#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-${REPO_ROOT}/docs/stage_experiments/S04/E04-09/raw}"
RUNNER="${REPO_ROOT}/scripts/audit/run_e04_09_bridge_abi_stream_safety.py"
RMS="${REPO_ROOT}/build/jetson-release/ops/cuda/rmsnorm/libhqsb_rmsnorm_shared.so"
FUSED="${REPO_ROOT}/build/jetson-release/ops/cuda/fused_residual_rmsnorm/libhqsb_fused_residual_rmsnorm_shared.so"
SAN="/usr/local/cuda-12.6/bin/compute-sanitizer"
HARNESS="${REPO_ROOT}/build/jetson-release/bin/hqsb_e03_07_sanitizer_harness"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUT}/sanitizer/logs"
python3 "${RUNNER}" initialize --output-dir "${OUT}" --rms-library "${RMS}"

runtime_rc=0
for process in 0 1 2; do
  if [[ "${process}" == "0" ]]; then
    sudo -n -E python3 "${RUNNER}" runtime --output-dir "${OUT}" \
      --rms-library "${RMS}" --process-index "${process}" || runtime_rc=$?
    sudo -n chown -R "$(id -u):$(id -g)" "${OUT}"
  else
    python3 "${RUNNER}" runtime --output-dir "${OUT}" \
      --rms-library "${RMS}" --process-index "${process}" || runtime_rc=$?
  fi
done

printf 'run_id\ttool\texpectation\texit_code\telapsed_s\tlog\tcommand\n' \
  > "${OUT}/sanitizer/index.tsv"

run_sanitizer() {
  local run_id="$1" expectation="$2" mode="$3"
  local log="sanitizer/logs/${run_id}.log" start end rc command
  command="sudo -n ${SAN} --tool memcheck --error-exitcode 86 ${HARNESS} ${mode} ${RMS} ${FUSED}"
  start="$(date +%s)"
  timeout --signal=KILL 600s sudo -n "${SAN}" --tool memcheck \
    --error-exitcode 86 --leak-check full "${HARNESS}" "${mode}" "${RMS}" "${FUSED}" \
    > "${OUT}/${log}" 2>&1
  rc=$?
  end="$(date +%s)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${run_id}" memcheck "${expectation}" "${rc}" "$((end-start))" \
    "${log}" "${command}" >> "${OUT}/sanitizer/index.tsv"
}

run_sanitizer rms_matrix_clean clean rms-matrix
run_sanitizer api_negative_exposes_boundary reject_or_detect api-negative

python3 "${RUNNER}" analyze --output-dir "${OUT}"
analysis_rc=$?
if [[ ${runtime_rc} -ne 0 ]]; then
  exit "${runtime_rc}"
fi
exit "${analysis_rc}"
