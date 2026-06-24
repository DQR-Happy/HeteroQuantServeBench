#!/usr/bin/env bash
# Build Ascend C kernels for S09 E09-02/03
#
# Usage:
#   scripts/ascend/build_ascend_ops.sh -B build/ascend-sm87 --soc=910b
#
# This script checks for the CANN toolchain (ccec/bisheng compiler) and fails
# with a clear message if absent. On this host it will fail, which is honest:
# the kernels are not compiled here, they are source artifacts only.
#
# When run on an Ascend machine with CANN installed:
#   1. Detects ccec/bisheng compiler
#   2. Sets ASCEND_TOOLKIT_ROOT
#   3. Invokes cmake to compile each operator kernel
#   4. Produces .o/.so artifacts in the build directory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ASCEND_DIR="${ROOT_DIR}/ops/ascend"

# Parse arguments
BUILD_DIR=""
SOC_VERSION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -B|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --soc)
            SOC_VERSION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${BUILD_DIR}" ]]; then
    echo "Error: -B/--build-dir is required" >&2
    exit 1
fi

if [[ -z "${SOC_VERSION}" ]]; then
    echo "Error: --soc is required (e.g., 910b, 310p)" >&2
    exit 1
fi

echo "Building Ascend C kernels for SoC ${SOC_VERSION}..."
echo "Build directory: ${BUILD_DIR}"

# Check for CANN compiler
CCEC_COMPILER=""
if command -v bisheng-c++ &> /dev/null; then
    CCEC_COMPILER=$(command -v bisheng-c++)
elif command -v ccec &> /dev/null; then
    CCEC_COMPILER=$(command -v ccec)
else
    echo "" >&2
    echo "ERROR: CANN compiler (ccec or bisheng-c++) not found." >&2
    echo "" >&2
    echo "To use this script on an Ascend machine:" >&2
    echo "  1. Install CANN toolkit matching your SoC version" >&2
    echo "  2. Ensure the compiler is on PATH or set ASCEND_TOOLKIT_ROOT" >&2
    echo "  3. Re-run this script" >&2
    echo "" >&2
    echo "This host does not have CANN installed — the kernels remain as" >&2
    echo "source artifacts only. They cannot be compiled here." >&2
    echo "" >&2
    exit 1
fi

echo "Found CANN compiler: ${CCEC_COMPILER}"

# Create build directory
mkdir -p "${BUILD_DIR}"

# Configure and build
echo ""
echo "Configuring with CMake..."
cmake "${ASCEND_DIR}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DASCEND_TOOLKIT_ROOT="${ASCEND_TOOLKIT_ROOT:-}" \
    -DSOC_VERSION="${SOC_VERSION}"

echo ""
echo "Building..."
cmake --build "${BUILD_DIR}" --config Release -j

echo ""
echo "Build complete. Artifacts in: ${BUILD_DIR}"
