#!/usr/bin/env bash
# Nsight Systems capture for a representative model-core run.
#
# Captures CPU/GPU timeline, launch gaps, synchronization, and memory
# copies so the S02 report can identify runtime overhead (launch/sync)
# versus kernel execution, and quantify CPU/GPU overlap.
#
# Usage:
#   ./scripts/bench/nsys_profile.sh [--isl N] [--osl N] [--out-dir DIR]
#
# Requires: nsys (Nsight Systems CLI) on PATH.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISL=128
OSL=32
OUT_DIR="${REPO_ROOT}/reports/dev/nsys/$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --isl) ISL="$2"; shift 2 ;;
    --osl) OSL="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${OUT_DIR}"

echo "Nsight Systems capture: ISL=${ISL} OSL=${OSL}"
echo "Output dir: ${OUT_DIR}"

# nsys CLI: profile a single model-core run. We use a tiny dedicated driver
# that loads once and runs one pass, so the timeline is not polluted by
# model loading. `--capture-range` is not used; instead the driver does a
# cudaProfilerStart/Stop around the timed region when available.
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --output="${OUT_DIR}/model_core" \
  --force-overwrite=true \
  --stats=true \
  python3 -c "
import os, sys, torch
sys.path.insert(0, '${REPO_ROOT}')
from hqsb.backends import PyTorchBackend
from hqsb.core.contracts import ModelArtifact, WorkloadSpec
from hqsb.benchmark.workload import make_fixed_token_input
from hqsb.benchmark.model_core import benchmark_model_core

artifact = ModelArtifact(model_id='Qwen/Qwen3-1.7B', source='modelscope', architecture='Qwen3ForCausalLM', dtype='float16')
backend = PyTorchBackend(model_path='${HOME}/models/hqsb/Qwen3-1.7B')
backend.load(artifact)
inputs = make_fixed_token_input(backend._tokenizer, ${ISL}, device='cuda')
result = benchmark_model_core(backend._model, inputs, ${OSL})
print('e2e_ms=', round(result['model_core_e2e_ms'], 2))
backend.close()
"

echo "Report: ${OUT_DIR}/model_core.nsys-rep"
echo "Generate a text summary with:"
echo "  nsys stats --report cuda_gpu_trace ${OUT_DIR}/model_core.nsys-rep"
