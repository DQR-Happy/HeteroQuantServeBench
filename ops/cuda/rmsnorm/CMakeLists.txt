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
