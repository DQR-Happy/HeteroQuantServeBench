# Reduction kernel — skeleton for S09 E09-02

add_library(ascend_reduction_kernel STATIC src/reduction_kernel.cpp)
target_include_directories(ascend_reduction_kernel PUBLIC include)
set_target_properties(ascend_reduction_kernel PROPERTIES CXX_STANDARD 17)
