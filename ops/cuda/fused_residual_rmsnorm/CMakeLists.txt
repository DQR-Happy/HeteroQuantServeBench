# Fused Residual + RMSNorm operator (second S02 hotspot).
#
# Public surface: ops/cuda/fused_residual_rmsnorm/include/hqsb/fused_residual_rmsnorm.h
# Reuses hqsb::DType from the RMSNorm header (header-only dependency; no
# linkage to the RMSNorm kernels is required).

add_library(hqsb_fused_residual_rmsnorm STATIC
  src/fused_residual_rmsnorm.cu
)

target_include_directories(
  hqsb_fused_residual_rmsnorm
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/ops/cuda/rmsnorm/include
)

target_compile_options(
  hqsb_fused_residual_rmsnorm
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
)

# ── Correctness test ──────────────────────────────────────────────────

add_executable(
  hqsb_fused_residual_rmsnorm_test
  tests/test_fused_residual_rmsnorm.cu
)

target_include_directories(
  hqsb_fused_residual_rmsnorm_test
  PRIVATE
  ${PROJECT_SOURCE_DIR}/ops/cuda/common
)

target_link_libraries(
  hqsb_fused_residual_rmsnorm_test
  PRIVATE
  hqsb_fused_residual_rmsnorm
)

target_compile_options(
  hqsb_fused_residual_rmsnorm_test
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
)

add_test(
  NAME fused_residual_rmsnorm_correctness
  COMMAND hqsb_fused_residual_rmsnorm_test
)

# ── Benchmark ─────────────────────────────────────────────────────────

add_executable(
  hqsb_fused_residual_rmsnorm_bench
  bench/bench_fused_residual_rmsnorm.cu
)

target_link_libraries(
  hqsb_fused_residual_rmsnorm_bench
  PRIVATE
  hqsb_fused_residual_rmsnorm
)

target_compile_options(
  hqsb_fused_residual_rmsnorm_bench
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
)
