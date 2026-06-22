# Add kernel — skeleton for S09 E09-02

add_library(ascend_add_kernel STATIC src/add_kernel.cpp)
target_include_directories(ascend_add_kernel PUBLIC include)
set_target_properties(ascend_add_kernel PROPERTIES CXX_STANDARD 17)

# Placeholder test/bench targets — will be wired when CANN toolchain is available
# add_executable(test_add tests/test_add.cpp)
# target_link_libraries(test_add PRIVATE ascend_add_kernel)
