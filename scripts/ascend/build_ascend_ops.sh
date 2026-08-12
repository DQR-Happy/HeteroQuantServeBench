#!/usr/bin/env bash
# Build the HQSB Ascend C custom-operator project (S09 E09-02/03/04).
#
# Usage:
#   scripts/ascend/build_ascend_ops.sh
#   scripts/ascend/build_ascend_ops.sh -B build/ascend-opp --soc 310B1
#   scripts/ascend/build_ascend_ops.sh -B build/ascend-opp --soc=310B1
#
# The operators live in ops/ascend/custom/ and are compiled by the CANN toolkit's
# own ascendc_kernel_cmake (reached through `find_package(ASC REQUIRED)`), so no
# vendor build machinery is vendored into the repository.  The output is a
# custom_opp_*.run package that can be installed with its own install.sh.
#
# Source the CANN environment first, e.g.
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#
# On a host without CANN this fails with a clear message instead of silently
# producing an artifact.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CUSTOM_DIR="${ROOT_DIR}/ops/ascend/custom"

BUILD_DIR="${ROOT_DIR}/build/ascend-opp"
SOC_VERSION="310B1"

while [[ $# -gt 0 ]]; do
    case $1 in
        -B|--build-dir)
            BUILD_DIR="${2:-}"
            shift 2
            ;;
        --build-dir=*)
            BUILD_DIR="${1#*=}"
            shift
            ;;
        --soc)
            SOC_VERSION="${2:-}"
            shift 2
            ;;
        --soc=*)
            SOC_VERSION="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${BUILD_DIR}" ]]; then
    echo "Error: -B/--build-dir must not be empty" >&2
    exit 1
fi

# The target SoC is fixed by ASCEND_COMPUTE_UNIT in
# ops/ascend/custom/CMakePresets.json (ai_core-Ascend310B1); --soc is accepted so
# a caller can state the intent explicitly, and is cross-checked below.
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "" >&2
    echo "ERROR: ASCEND_HOME_PATH is not set." >&2
    echo "  source <CANN>/set_env.sh before running this script." >&2
    echo "" >&2
    exit 1
fi

if ! command -v ccec >/dev/null 2>&1; then
    echo "" >&2
    echo "ERROR: CANN compiler (ccec) not found on PATH." >&2
    echo "  source <CANN>/set_env.sh (or export ASCEND_HOME_PATH) and retry." >&2
    echo "" >&2
    exit 1
fi

PRESET_PARSER="${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake/fwk_modules/util/preset_parse.py"
if [[ ! -f "${PRESET_PARSER}" ]]; then
    echo "ERROR: CANN cmake preset helper not found: ${PRESET_PARSER}" >&2
    exit 1
fi

echo "==> Building HQSB Ascend C operators"
echo "    sources : ${CUSTOM_DIR}"
echo "    build   : ${BUILD_DIR}"
echo "    SoC     : ${SOC_VERSION}"
echo "    CANN    : ${ASCEND_HOME_PATH}"

mkdir -p "${BUILD_DIR}"

# preset_parse.py prints all -D flags on ONE line: split on whitespace, not on
# newlines.  `mapfile -t` would put the whole line into a single argv element and
# CMake would then treat "-DENABLE_... -DASCEND_COMPUTE_UNIT=..." as the *value*
# of CMAKE_BUILD_TYPE, leaving ASCEND_COMPUTE_UNIT unset and silently falling
# back to the toolkit default (ascend910b) -- which is a wrong-target build, not
# a build error.
read -r -a CMAKE_OPTIONS <<< "$(python3 "${PRESET_PARSER}" "${CUSTOM_DIR}/CMakePresets.json")"

# The preset's install prefix points at <source>/build_out; redirect it into the
# ignored build tree so no artifact lands in the source directory.
cmake -S "${CUSTOM_DIR}" -B "${BUILD_DIR}" \
    "${CMAKE_OPTIONS[@]}" \
    "-DCMAKE_INSTALL_PREFIX=${BUILD_DIR}/install"
cmake --build "${BUILD_DIR}" --target binary package -j"$(nproc)"

echo ""
echo "==> Build complete. Packages:"
find "${BUILD_DIR}" -maxdepth 2 -name "custom_opp*.run" -print || true
