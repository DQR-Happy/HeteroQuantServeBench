#!/usr/bin/env bash
# Build and run the aclnn smoke harness against the installed HQSB operators.
#
# Usage:
#   scripts/ascend/run_ascend_smoke.sh
#   scripts/ascend/run_ascend_smoke.sh --install-root /path/to/custom_ops
#
# Requires:
#   * a sourced CANN environment (source <CANN>/set_env.sh), which sets
#     ASCEND_HOME_PATH and the runtime library path;
#   * the operator package installed:
#       build/ascend-opp/custom_opp_*.run --quiet --install-path=<root>
#
# The harness executes all three operators on the real device and compares the
# results with an independent CPU reference.  It is evidence that the toolchain,
# package, loader and kernels work; it is not a performance measurement.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HARNESS_SRC="${ROOT_DIR}/ops/ascend/custom/harness/aclnn_smoke.cpp"
BUILD_DIR="${ROOT_DIR}/build/ascend-opp"
INSTALL_ROOT="${HQSB_CUSTOM_OPP_ROOT:-${HOME}/ascend_custom_ops}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --install-root)
            INSTALL_ROOT="${2:-}"
            shift 2
            ;;
        --install-root=*)
            INSTALL_ROOT="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ERROR: ASCEND_HOME_PATH is not set." >&2
    echo "  source <CANN>/set_env.sh before running this script." >&2
    exit 1
fi

OP_INCLUDE="${INSTALL_ROOT}/vendors/hqsb/op_api/include"
OP_LIB="${INSTALL_ROOT}/vendors/hqsb/op_api/lib"
if [[ ! -d "${OP_INCLUDE}" || ! -d "${OP_LIB}" ]]; then
    echo "ERROR: HQSB operator package not found under ${INSTALL_ROOT}" >&2
    echo "  Install it first:" >&2
    echo "    build/ascend-opp/custom_opp_*.run --quiet --install-path=${INSTALL_ROOT}" >&2
    exit 1
fi

# The package ships its own environment hook (ASCEND_CUSTOM_OPP_PATH + op_api/lib).
# It appends to ${ASCEND_CUSTOM_OPP_PATH} without a default, which `set -u` would
# reject, so seed the variable first.
if [[ -f "${INSTALL_ROOT}/vendors/hqsb/bin/set_env.bash" ]]; then
    export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
    # shellcheck disable=SC1091
    source "${INSTALL_ROOT}/vendors/hqsb/bin/set_env.bash"
fi

CANN_INCLUDE="${ASCEND_HOME_PATH}/aarch64-linux/include"
CANN_DEVLIB="${ASCEND_HOME_PATH}/aarch64-linux/devlib"
CANN_RUNTIME_LIB="${ASCEND_HOME_PATH}/lib64"

mkdir -p "${BUILD_DIR}"

echo "==> Compiling aclnn smoke harness"
echo "    source : ${HARNESS_SRC}"
echo "    package: ${INSTALL_ROOT}/vendors/hqsb"
c++ -std=c++17 "${HARNESS_SRC}" -o "${BUILD_DIR}/aclnn_smoke" \
    -I"${CANN_INCLUDE}" \
    -I"${OP_INCLUDE}" \
    -L"${CANN_RUNTIME_LIB}" -lascendcl -lnnopbase \
    -L"${OP_LIB}" -lcust_opapi \
    -Wl,-rpath,"${CANN_RUNTIME_LIB}" \
    -Wl,-rpath,"${OP_LIB}"

echo ""
echo "==> Running aclnn smoke on device 0"
"${BUILD_DIR}/aclnn_smoke"
