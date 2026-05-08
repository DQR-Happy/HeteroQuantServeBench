#!/usr/bin/env bash
# Nsight Compute capture for top candidate kernels.
#
# Nsight Compute (ncu) profiles individual CUDA kernels, recording memory
# throughput, SM utilization, occupancy, stall reasons, and register/shared
# memory usage — the hardware evidence required to justify S03 kernel work.
#
# Usage:
#   ./scripts/bench/ncu_profile.sh --kernel <regex> [--isl N] [--osl N]
#
# Requires: ncu (Nsight Compute CLI). On Jetson/L4T, kernel profiling may
# require running as root and is sometimes unavailable for unified-memory
# devices; use `--launch-skip`/`--launch-count` to bound the capture region.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KERNEL=".*"     # default: profile all kernels (narrow with --kernel)
ISL=128
OSL=16
OUT_DIR="${REPO_ROOT}/reports/dev/ncu/$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel) KERNEL="$2"; shift 2 ;;
    --isl) ISL="$2"; shift 2 ;;
    --osl) OSL="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${OUT_DIR}"

echo "Nsight Compute capture: kernel='${KERNEL}' ISL=${ISL} OSL=${OSL}"

ncu \
  --kernel-name "${KERNEL}" \
  --launch-skip 1 \
  --launch-count 4 \
  --set basic \
  --export "${OUT_DIR}/ncu_report" \
  --force-overwrite \
  python3 -c "
import sys
sys.path.insert(0, '${REPO_ROOT}')
from hqsb.backends import PyTorchBackend
from hqsb.core.contracts import ModelArtifact
from hqsb.benchmark.workload import make_fixed_token_input
from hqsb.benchmark.model_core import benchmark_model_core

artifact = ModelArtifact(model_id='Qwen/Qwen3-1.7B', source='modelscope', architecture='Qwen3ForCausalLM', dtype='float16')
backend = PyTorchBackend(model_path='${HOME}/models/hqsb/Qwen3-1.7B')
backend.load(artifact)
inputs = make_fixed_token_input(backend._tokenizer, ${ISL}, device='cuda')
benchmark_model_core(backend._model, inputs, ${OSL})
backend.close()
"

echo "Report: ${OUT_DIR}/ncu_report.ncu-rep"
