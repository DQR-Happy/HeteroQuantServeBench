# RMSNorm operator library (V0/V1/V2 + dispatcher + reference).
#
# Public surface: ops/cuda/rmsnorm/include/hqsb/rmsnorm.h
# The library is a STATIC target so S04 (Triton) and S06 (framework) can link
# against the same compiled kernels.

add_library(hqsb_rmsnorm STATIC
  src/rmsnorm_reference.cu
  src/rmsnorm_v0.cu
  src/rmsnorm_v1.cu
  src/rmsnorm_v2.cu
  src/rmsnorm_dispatcher.cu
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
