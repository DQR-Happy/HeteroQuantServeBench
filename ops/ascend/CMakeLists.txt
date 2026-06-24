# Ascend C kernel build script — not added to the CUDA-only top-level CMakeLists.txt
# because this host has no CANN toolchain.  Run directly:
#   scripts/ascend/build_ascend_ops.sh -B build/ascend-sm87 --soc=910b
# The script checks for ccec/bisheng and fails with a clear message if absent.

cmake_minimum_required(VERSION 3.18)
project(HQSB_Ascend_Ops VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CANN toolchain detection — will fail on non-Ascend hosts
find_program(CCEC_COMPILER bisheng-c++ HINTS ${ASCEND_TOOLKIT_ROOT}/compiler/bin NO_DEFAULT_PATH)
if(NOT CCEC_COMPILER)
    find_program(CCEC_COMPILER ccec HINTS /usr/local/Ascend/ccec/bin NO_DEFAULT_PATH)
endif()

if(NOT CCEC_COMPILER)
    message(FATAL_ERROR "CANN compiler (ccec/bisheng) not found; set ASCEND_TOOLKIT_ROOT or ensure CANN is installed")
endif()

message(STATUS "Found CANN compiler: ${CCEC_COMPILER}")

# Add subdirectories for each operator
add_subdirectory(add)
add_subdirectory(reduction)
add_subdirectory(rmsnorm)
