#pragma once
//
// Internal launcher declarations shared between the per-variant translation
// units and the dispatcher/benchmark harness. This header is *not* part of
// the public API; the public surface is ``hqsb/rmsnorm.h``.
//

#include "hqsb/rmsnorm.h"

#include <cuda_runtime.h>

namespace hqsb {

// V0 (FP32 only): shared-memory tree reduction.
cudaError_t rmsnorm_v0(const float* input,
                       const float* weight,
                       float* output,
                       int64_t rows,
                       int64_t hidden,
                       float epsilon,
                       int block_size,
                       cudaStream_t stream);

// V1 (FP32 only): warp-shuffle reduction.
cudaError_t rmsnorm_v1(const float* input,
                       const float* weight,
                       float* output,
                       int64_t rows,
                       int64_t hidden,
                       float epsilon,
                       int block_size,
                       cudaStream_t stream);

// V2 (FP32 float4 / FP16 half2): vectorized load + warp shuffle.
cudaError_t rmsnorm_v2(const void* input,
                       const void* weight,
                       void* output,
                       int64_t rows,
                       int64_t hidden,
                       float epsilon,
                       DType dtype,
                       int block_size,
                       cudaStream_t stream);

// Occupancy queries (theoretical max active blocks per SM) for the
// benchmark harness's occupancy-constraint reporting.
int rmsnorm_v0_occupancy(int block_size);
int rmsnorm_v1_occupancy(int block_size);
int rmsnorm_v2_occupancy(DType dtype, int block_size);

}  // namespace hqsb
