#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="$ROOT_DIR/build/jetson-release"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$ROOT_DIR/reports/jetson/$RUN_ID"

mkdir -p "$OUTPUT_DIR"

echo "Run ID: $RUN_ID"
echo "Output: $OUTPUT_DIR"

"$ROOT_DIR/scripts/env/collect_jetson_env.sh" \
  "$OUTPUT_DIR/environment.md"

cmake \
  -S "$ROOT_DIR" \
  -B "$BUILD_DIR" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  2>&1 | tee "$OUTPUT_DIR/configure.txt"

cmake \
  --build "$BUILD_DIR" \
  --parallel "$(nproc)" \
  2>&1 | tee "$OUTPUT_DIR/build.txt"

"$BUILD_DIR/bin/hqsb_device_query" \
  2>&1 | tee "$OUTPUT_DIR/device_query.txt"

: > "$OUTPUT_DIR/rmsnorm_runs.txt"

for run in 1 2 3 4 5; do
  echo "===== RUN $run =====" | tee -a "$OUTPUT_DIR/rmsnorm_runs.txt"

  "$BUILD_DIR/bin/hqsb_rmsnorm_baseline" \
    --rows 512 \
    --hidden 1024 \
    --warmup 50 \
    --iterations 500 \
    --threads 256 \
    2>&1 | tee -a "$OUTPUT_DIR/rmsnorm_runs.txt"
done

cat > "$OUTPUT_DIR/README.md" <<REPORT
# Jetson CUDA Baseline

- Run ID: $RUN_ID
- Build type: Release
- CUDA architecture: 87
- RMSNorm shape: rows=512, hidden=1024
- Warmup iterations: 50
- Measured iterations: 500
- Repetitions: 5

Files:

- environment.md
- configure.txt
- build.txt
- device_query.txt
- rmsnorm_runs.txt
- tegrastats.log, if captured separately
REPORT

echo
echo "Baseline completed."
echo "Results: $OUTPUT_DIR"
