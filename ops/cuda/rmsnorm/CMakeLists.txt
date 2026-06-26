# RMSNorm operator library (V0/V1/V2 + dispatcher + reference).
#
# Public surface: ops/cuda/rmsnorm/include/hqsb/rmsnorm.h
# The library is a STATIC target so S04 (Triton) and S06 (framework) can link
# against the same compiled kernels.

# ── Build-arch provenance (consumed by the Python dispatcher) ──────────────
#
# The Python dispatcher (ops/dispatcher.py) must decide whether this
# precompiled kernel is usable on the *runtime* device. Recording the arch
# only in Python would be a second hard-coded constant that breaks on every
# other platform, so the library carries its own build arch instead and
# exposes it through the C ABI (``hqsb_rmsnorm_query_build_arch``).
#
# Accepts the CMake spellings "86", "86-real", "80;86" and "90a" style
# suffixes; the first numeric entry wins. A value CMake cannot parse (e.g.
# "native", "all") is exported as 0/0 = "unknown", which the dispatcher
# treats as "cannot prove the binary is usable" -- never as a silent match.
set(_hqsb_arch_entry "${CMAKE_CUDA_ARCHITECTURES}")
list(GET _hqsb_arch_entry 0 _hqsb_arch_entry)
string(REGEX REPLACE "[-+ ].*$" "" _hqsb_arch_entry "${_hqsb_arch_entry}")
string(REGEX MATCH "^[0-9]+" _hqsb_arch_digits "${_hqsb_arch_entry}")

if(_hqsb_arch_digits)
  math(EXPR _hqsb_arch_major "${_hqsb_arch_digits} / 10")
  math(EXPR _hqsb_arch_minor "${_hqsb_arch_digits} % 10")
else()
  set(_hqsb_arch_major 0)
  set(_hqsb_arch_minor 0)
  message(
    WARNING
    "Cannot parse CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHITECTURES}'; "
    "exporting build arch 0.0 (unknown) to the dispatcher."
  )
endif()

message(
  STATUS
  "hqsb_rmsnorm build arch = ${_hqsb_arch_major}.${_hqsb_arch_minor} "
  "(from CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHITECTURES}')"
)

add_library(hqsb_rmsnorm STATIC
  src/rmsnorm_reference.cu
  src/rmsnorm_v0.cu
  src/rmsnorm_v1.cu
  src/rmsnorm_v2.cu
  src/rmsnorm_dispatcher.cu
)

target_compile_definitions(
  hqsb_rmsnorm
  PRIVATE
  HQSB_RMSNORM_BUILD_ARCH_MAJOR=${_hqsb_arch_major}
  HQSB_RMSNORM_BUILD_ARCH_MINOR=${_hqsb_arch_minor}
)

target_include_directories(
  hqsb_rmsnorm
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(
  hqsb_rmsnorm
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
)

# Shared library exposing a stable C ABI for the Python dispatcher (S04).
# This is the *only* mechanism Python uses to reach the CUDA kernels: a thin
# ``ctypes`` binding loads ``libhqsb_rmsnorm.so`` and calls the unmangled
# ``hqsb_rmsnorm_forward_c`` symbol.
add_library(hqsb_rmsnorm_shared SHARED
  src/rmsnorm_reference.cu
  src/rmsnorm_v0.cu
  src/rmsnorm_v1.cu
  src/rmsnorm_v2.cu
  src/rmsnorm_dispatcher.cu
  src/rmsnorm_c_api.cu
)

target_include_directories(
  hqsb_rmsnorm_shared
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(
  hqsb_rmsnorm_shared
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
)

# The C ABI entry point ``hqsb_rmsnorm_query_build_arch`` reports these back
# to Python so the dispatcher can gate on the real build arch.
target_compile_definitions(
  hqsb_rmsnorm_shared
  PRIVATE
  HQSB_RMSNORM_BUILD_ARCH_MAJOR=${_hqsb_arch_major}
  HQSB_RMSNORM_BUILD_ARCH_MINOR=${_hqsb_arch_minor}
)

# ── Correctness test ──────────────────────────────────────────────────

add_executable(hqsb_rmsnorm_test tests/test_rmsnorm.cu)

target_include_directories(
  hqsb_rmsnorm_test
  PRIVATE
  ${PROJECT_SOURCE_DIR}/ops/cuda/common
)

target_link_libraries(hqsb_rmsnorm_test PRIVATE hqsb_rmsnorm)

target_compile_options(
  hqsb_rmsnorm_test
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
)

add_test(NAME rmsnorm_correctness COMMAND hqsb_rmsnorm_test)

# E03-07 standalone Compute Sanitizer harness.  It loads the production
# shared libraries with dlopen so that the injected negative-control kernels
# below can never become part of either operator library.
add_executable(
  hqsb_e03_07_sanitizer_harness
  tests/e03_07_sanitizer_harness.cu
)

target_link_libraries(
  hqsb_e03_07_sanitizer_harness
  PRIVATE
  ${CMAKE_DL_LIBS}
)

target_compile_options(
  hqsb_e03_07_sanitizer_harness
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
)

# ── Benchmark (sweeps block size via the internal launchers) ──────────

add_executable(hqsb_rmsnorm_bench bench/bench_rmsnorm.cu)

target_include_directories(
  hqsb_rmsnorm_bench
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hqsb_rmsnorm_bench PRIVATE hqsb_rmsnorm)

target_compile_options(
  hqsb_rmsnorm_bench
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
)
