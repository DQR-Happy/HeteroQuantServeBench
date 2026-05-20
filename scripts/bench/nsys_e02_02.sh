#!/usr/bin/env bash
# E02-02 NSYS cross-validation capture.
#
# Captures the timeline only inside the NVTX range `e02_02_model_core`, so
# model loading and warmup are excluded. Produces a .nsys-rep timeline and a
# machine-readable per-kernel CSV so the two-layer PyTorch-profiler attribution
# (host op vs device kernel) can be cross-checked.
#
# Usage (on the Jetson, typically via ./scripts/remote_run.sh):
#   scripts/bench/nsys_e02_02.sh --name long_prefill --isl 2048 --osl 32 \
#       --out-dir docs/stage_experiments/S02/E02-02/raw_v2/nsys
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${HOME}/models/hqsb/Qwen3-1.7B"
NAME=""
ISL=128
OSL=32
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --isl) ISL="$2"; shift 2 ;;
    --osl) OSL="$2"; shift 2 ;;
    --out-dir) OUT="$2"; shift 2 ;;
    --model-path) MODEL="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$NAME" || -z "$OUT" ]]; then
  echo "usage: nsys_e02_02.sh --name X --isl N --osl N --out-dir DIR" >&2
  exit 2
fi

# raw_v2 evidence dirs are created by privileged runs; fall back to sudo.
mkdir -p "$OUT" 2>/dev/null || sudo mkdir -p "$OUT"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "nsys capture: name=${NAME} isl=${ISL} osl=${OSL}"
sudo -E env PYTHONPATH="${PYTHONPATH}" nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output="${OUT}/${NAME}" \
  --force-overwrite=true \
  --stats=true \
  python3 "${REPO_ROOT}/scripts/bench/nsys_e02_02.py" \
    --model-path "${MODEL}" --isl "${ISL}" --osl "${OSL}"

echo "--- kernel summary (cuda_gpu_kern_sum) ---"
sudo -E env PYTHONPATH="${PYTHONPATH}" nsys stats \
  --report cuda_gpu_kern_sum \
  --format csv \
  --output "${OUT}/${NAME}_kernels" \
  "${OUT}/${NAME}.nsys-rep" || true

ls -la "${OUT}" | grep -i "${NAME}" || true
