#!/usr/bin/env bash
# Collect Ascend environment for S09 E09-01 compatibility manifest
#
# Usage:
#   scripts/ascend/collect_ascend_env.sh > docs/evidence/ascend_env.json
#
# This script collects the hardware/software stack information required by
# :mod:`hqsb.ascend.compatibility.CompatibilityManifest`.  On a host without
# CANN installed, it will fail with a clear message rather than producing
# fabricated data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Check for npu-smi
if ! command -v npu-smi &> /dev/null; then
    echo "" >&2
    echo "ERROR: npu-smi not found. This script requires an Ascend machine with CANN installed." >&2
    echo "" >&2
    echo "On an Ascend machine:" >&2
    echo "  1. Install CANN toolkit" >&2
    echo "  2. Ensure npu-smi is on PATH" >&2
    echo "  3. Re-run this script" >&2
    echo "" >&2
    exit 1
fi

echo "Collecting Ascend environment..."

# Hardware info
DEVICE_INFO=$(npu-smi info -t device -i 0 2>/dev/null || echo '{"error":"device_query_failed"}')

# Firmware/driver
FIRMWARE_INFO=$(npu-smi info -t product -i 0 2>/dev/null || echo '{"error":"product_query_failed"}')

# CANN components (if available).  CANN 9.x ships no `version.cfg` and exposes its
# install root through ASCEND_HOME_PATH / ASCEND_TOOLKIT_HOME (older docs say
# ASCEND_TOOLKIT_ROOT); all three are honoured so this does not report "unknown"
# on a correctly sourced board.
CANN_ROOT="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-${ASCEND_TOOLKIT_ROOT:-}}}"

CANN_VERSION=""
if [[ -n "${CANN_ROOT}" && -f "${CANN_ROOT}/compiler/version.info" ]]; then
    CANN_VERSION=$(head -5 "${CANN_ROOT}/compiler/version.info")
elif [[ -n "${CANN_ROOT}" && -f "${CANN_ROOT}/version.cfg" ]]; then
    CANN_VERSION=$(head -5 "${CANN_ROOT}/version.cfg")
elif command -v ccec &> /dev/null; then
    CANN_VERSION=$(ccec --version 2>&1 | head -1)
else
    CANN_VERSION="UNAVAILABLE"
fi

# Framework info
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not_installed")
TORCH_NPU_VERSION=$(python3 -c "import torch_npu; print(torch_npu.__version__)" 2>/dev/null || echo "not_installed")

# Output JSON
cat <<EOF
{
  "collected_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hardware": {
    "device_info": ${DEVICE_INFO},
    "firmware_info": ${FIRMWARE_INFO}
  },
  "ascend_stack": {
    "cann_version": "${CANN_VERSION}",
    "toolkit_root": "${CANN_ROOT:-unknown}"
  },
  "framework": {
    "python_version": "${PYTHON_VERSION}",
    "torch_version": "${TORCH_VERSION}",
    "torch_npu_version": "${TORCH_NPU_VERSION}"
  }
}
EOF
