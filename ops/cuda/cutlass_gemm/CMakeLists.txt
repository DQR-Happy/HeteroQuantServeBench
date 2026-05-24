# CUTLASS GEMM comparison benchmark + correctness gate (S04).
#
# CUTLASS is an optional external dependency (header-only). This target is
# only configured when the headers are present (third_party/cutlass or an
# override via -DCUTLASS_INCLUDE_DIR); otherwise it is silently skipped so
# the core CUDA library still builds on machines without CUTLASS.

if(NOT DEFINED CUTLASS_INCLUDE_DIR)
  set(_cutlass_candidate "${PROJECT_SOURCE_DIR}/third_party/cutlass/include")
  if(IS_DIRECTORY "${_cutlass_candidate}")
    set(CUTLASS_INCLUDE_DIR "${_cutlass_candidate}")
  endif()
endif()

if(CUTLASS_INCLUDE_DIR AND IS_DIRECTORY "${CUTLASS_INCLUDE_DIR}")
  add_executable(hqsb_cutlass_gemm_bench bench_cutlass_gemm.cu)

  target_include_directories(
    hqsb_cutlass_gemm_bench
    PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    # hqsb::test metrics (ops/cuda/common/test_metrics.h) shared with the
    # RMSNorm / fused-residual correctness suites.
    ${PROJECT_SOURCE_DIR}/ops/cuda/common
  )

  target_compile_features(hqsb_cutlass_gemm_bench PRIVATE cxx_std_17)

  target_compile_options(
    hqsb_cutlass_gemm_bench
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-std=c++17>
  )

  # ── Correctness gate ──────────────────────────────────────────────────
  #
  # Registered as a CTest so the CUTLASS path is covered by the same
  # `ctest` contract as the hand-written kernels. The executable selects the
  # largest tile the *device* can hold (see bench_cutlass_gemm.cu), so this
  # test is expected to pass on sm_80 / sm_86 / sm_87 alike. A non-zero exit
  # means: 2 = kernel never ran, 3 = wrong output, 4 = no config fits.
  add_test(
    NAME cutlass_gemm_correctness
    COMMAND hqsb_cutlass_gemm_bench
            --m 64 --n 128 --k 64 --warmup 2 --iterations 3
  )

  message(STATUS "CUTLASS found at ${CUTLASS_INCLUDE_DIR}; building GEMM comparison")
else()
  message(STATUS "CUTLASS not found; skipping GEMM comparison target")
endif()
