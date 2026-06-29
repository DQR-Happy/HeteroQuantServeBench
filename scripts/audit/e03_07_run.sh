#!/usr/bin/env bash
# Remote-only orchestration for E03-07. Every sanitizer invocation is an
# isolated process with a hard timeout; a poisoned CUDA context is never reused.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-${REPO_ROOT}/docs/stage_experiments/S03/E03-07/raw}"
SAN="/usr/local/cuda-12.6/bin/compute-sanitizer"
HARNESS="${REPO_ROOT}/build/jetson-release/bin/hqsb_e03_07_sanitizer_harness"
RMS="${REPO_ROOT}/build/jetson-release/ops/cuda/rmsnorm/libhqsb_rmsnorm_shared.so"
FUSED="${REPO_ROOT}/build/jetson-release/ops/cuda/fused_residual_rmsnorm/libhqsb_fused_residual_rmsnorm_shared.so"
PARSER="${REPO_ROOT}/scripts/audit/run_e03_07_sanitizer.py"

mkdir -p "${OUT}/logs"
: > "${OUT}/index.tsv"
printf 'run_id\ttool\texpectation\texit_code\telapsed_s\tlog\tcommand\n' >> "${OUT}/index.tsv"

{
  date -u +'%Y-%m-%dT%H:%M:%SZ'
  uname -a
  /usr/local/cuda/bin/nvcc --version
  "${SAN}" --version
  nvidia-smi
  sha256sum "${HARNESS}" "${RMS}" "${FUSED}"
  git rev-parse HEAD
  git status --short
} > "${OUT}/provenance.txt" 2>&1

{
  echo "command=grep -R -n -E 'cuda(Malloc|Free|DeviceSynchronize|StreamSynchronize)' production_sources"
  if ! grep -R -n -E 'cuda(Malloc|Free|DeviceSynchronize|StreamSynchronize)' \
      "${REPO_ROOT}/ops/cuda/rmsnorm/src" \
      "${REPO_ROOT}/ops/cuda/fused_residual_rmsnorm/src"; then
    echo "NO_MATCHES"
  fi
} > "${OUT}/static_allocation_sync_audit.txt" 2>&1

# Preserve the target-specific permission boundary: on this Jetson build the
# tool exists for ordinary users but GPU debugging instrumentation requires
# sudo. This probe must not be mistaken for a production sanitizer failure.
set +e
"${SAN}" --tool memcheck --error-exitcode 86 "${HARNESS}" control-clean \
  > "${OUT}/nonroot_permission_probe.log" 2>&1
permission_rc=$?
set -u
printf 'nonroot_exit_code=%s\nrequired_invocation=sudo -n %s\n' \
  "${permission_rc}" "${SAN}" > "${OUT}/permission_resolution.txt"

run_plain() {
  local run_id="$1" expectation="$2" mode="$3"
  local log="logs/${run_id}.log" start end rc command
  command="${HARNESS} ${mode} ${RMS} ${FUSED}"
  start="$(date +%s)"
  timeout --signal=KILL 300s "${HARNESS}" "${mode}" "${RMS}" "${FUSED}" \
    > "${OUT}/${log}" 2>&1
  rc=$?
  end="$(date +%s)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${run_id}" ordinary "${expectation}" "${rc}" "$((end-start))" \
    "${log}" "${command}" >> "${OUT}/index.tsv"
}

run_sanitizer() {
  local run_id="$1" tool="$2" expectation="$3" mode="$4"
  local log="logs/${run_id}.log" start end rc command
  local options=(--tool "${tool}" --error-exitcode 86)
  if [[ "${tool}" == memcheck ]]; then
    options+=(--leak-check full)
  fi
  command="sudo -n ${SAN} ${options[*]} ${HARNESS} ${mode} ${RMS} ${FUSED}"
  start="$(date +%s)"
  timeout --signal=KILL 600s sudo -n "${SAN}" "${options[@]}" \
    "${HARNESS}" "${mode}" "${RMS}" "${FUSED}" \
    > "${OUT}/${log}" 2>&1
  rc=$?
  end="$(date +%s)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${run_id}" "${tool}" "${expectation}" "${rc}" "$((end-start))" \
    "${log}" "${command}" >> "${OUT}/index.tsv"
}

# Tool self-validation. Negative controls live only in the audit harness.
for tool in memcheck racecheck initcheck synccheck; do
  run_sanitizer "control_${tool}_clean" "${tool}" clean control-clean
  run_sanitizer "control_${tool}_negative" "${tool}" detect \
    "control-${tool}"
done

# Ordinary API validation is kept separate from sanitizer interpretation.
run_plain api_negative ordinary_clean api-negative
run_plain destroyed_stream ordinary_clean destroyed-stream
run_plain host_pointer ordinary_clean host-pointer
run_plain lifecycle ordinary_clean lifecycle

# Every production family/path is executed under all four complementary tools.
for tool in memcheck racecheck initcheck synccheck; do
  run_sanitizer "${tool}_rms_matrix" "${tool}" clean rms-matrix
  run_sanitizer "${tool}_fused_matrix" "${tool}" clean fused-matrix
done

# Leak reporting and dangerous pointer/stream paths receive dedicated memcheck
# subprocesses so a fatal asynchronous error cannot contaminate a clean case.
run_sanitizer memcheck_lifecycle memcheck clean lifecycle
run_sanitizer memcheck_api_negative memcheck clean api-negative
run_sanitizer memcheck_destroyed_stream memcheck clean destroyed-stream
run_sanitizer memcheck_host_pointer memcheck clean host-pointer

python3 "${PARSER}" --output-dir "${OUT}"
