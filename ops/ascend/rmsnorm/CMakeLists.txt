# RMSNorm kernel — skeleton for S09 E09-02/03

add_library(ascend_rmsnorm_kernel STATIC src/rmsnorm_kernel.cpp)
target_include_directories(ascend_rmsnorm_kernel PUBLIC include)
set_target_properties(ascend_rmsnorm_kernel PROPERTIES CXX_STANDARD 17)
