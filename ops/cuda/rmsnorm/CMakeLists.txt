add_executable(hqsb_rmsnorm_baseline rmsnorm_baseline.cu)

target_include_directories(
  hqsb_rmsnorm_baseline
  PRIVATE
  ${PROJECT_SOURCE_DIR}/ops/cuda/common
)

target_compile_options(
  hqsb_rmsnorm_baseline
  PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
)
