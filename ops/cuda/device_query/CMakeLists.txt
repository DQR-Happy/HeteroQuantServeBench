add_executable(hqsb_device_query device_query.cu)

target_include_directories(
  hqsb_device_query
  PRIVATE
  ${PROJECT_SOURCE_DIR}/ops/cuda/common
)

target_compile_options(
  hqsb_device_query
  PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
)
