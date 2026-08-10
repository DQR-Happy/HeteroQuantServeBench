# -----------------------------------------------------------------------------------------------------------
# Device side: Ascend C kernels.
#
# Identical to the CANN-generated project except for the extra include path:
# the device kernel reads the same TilingData ABI header as the host tiling.
# -----------------------------------------------------------------------------------------------------------

if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    npu_op_kernel_options(ascendc_kernels ALL OPTIONS -g -O0)
endif()

npu_op_kernel_options(ascendc_kernels ALL OPTIONS -I${CMAKE_CURRENT_SOURCE_DIR}/../common)

npu_op_kernel_sources(ascendc_kernels
    KERNEL_DIR ./
)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY ascendc_kernels
)
