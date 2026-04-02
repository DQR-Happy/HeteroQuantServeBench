#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${1:-$ROOT_DIR/docs/hardware/jetson_environment.md}"

mkdir -p "$(dirname "$OUTPUT")"

{
  echo "# Jetson Environment Manifest"
  echo
  echo "Generated: $(date --iso-8601=seconds)"
  echo

  echo "## Device"
  echo '```text'
  tr -d '\0' < /proc/device-tree/model 2>/dev/null || true
  echo
  uname -m
  echo '```'
  echo

  echo "## Jetson Linux / L4T"
  echo '```text'
  cat /etc/nv_tegra_release 2>/dev/null || true
  echo '```'
  echo

  echo "## Operating system"
  echo '```text'
  cat /etc/os-release
  echo '```'
  echo

  echo "## CUDA"
  echo '```text'
  echo "CUDA_HOME=${CUDA_HOME:-not-set}"
  echo "nvcc=$(command -v nvcc 2>/dev/null || echo not-found)"
  readlink -f /usr/local/cuda 2>/dev/null || true
  nvcc --version 2>/dev/null || true
  echo '```'
  echo

  echo "## Toolchain"
  echo '```text'
  gcc --version | head -n 1
  g++ --version | head -n 1
  cmake --version | head -n 1
  ninja --version 2>/dev/null || true
  python3 --version
  git --version
  echo '```'
  echo

  echo "## NVIDIA packages"
  echo '```text'
  dpkg-query -W \
    nvidia-l4t-core \
    nvidia-jetpack \
    nvidia-jetpack-runtime \
    nvidia-jetpack-dev \
    2>/dev/null || true
  echo '```'
  echo

  echo "## Memory"
  echo '```text'
  free -h
  echo '```'
  echo

  echo "## Storage"
  echo '```text'
  df -h /
  df -h "$HOME"
  lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS
  echo '```'
  echo

  echo "## Power mode"
  echo '```text'
  sudo nvpmodel -q 2>&1 || true
  echo '```'
  echo

  echo "## Clock state"
  echo '```text'
  sudo jetson_clocks --show 2>&1 || true
  echo '```'

} > "$OUTPUT"

echo "Environment manifest written to: $OUTPUT"
