find_package(CUDAToolkit REQUIRED)
add_library(hqsb_model_kernels SHARED norm_activation.cu rope.cu linear.cu)
target_link_libraries(hqsb_model_kernels PRIVATE CUDA::cudart CUDA::cublasLt)
# Preserve FP16 intermediate rounding and avoid contraction across semantic steps.
target_compile_options(hqsb_model_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
set_target_properties(hqsb_model_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
