#pragma once
//
// HQSB Fused Residual + RMSNorm operator — public C/C++ API (S03).
//
// This is the second hotspot operator selected in S02: the Qwen3 decoder
// block computes ``hidden = residual + hidden`` followed by ``hidden =
// rmsnorm(hidden) * weight``. Fusing them into one kernel eliminates the
// round-trip of the intermediate ``x + residual`` tensor through global
// memory, which the S02 profile attributed to ``aten::add`` + ``aten::copy_``
// elementwise traffic.
//
// out = rmsnorm(input + residual) * weight, per row.
//

#include "hqsb/rmsnorm.h"  // reuses hqsb::DType

#include <cuda_runtime.h>

#include <cstdint>

namespace hqsb {

enum class FusedResidualVariant : int {
  kAuto = 0,
  kReference = 1,      // CPU FP64 reference (test/benchmark only)
  kV0Shared = 2,       // fused, shared-memory reduction
  kV1Vectorized = 3,   // fused, warp shuffle + float4/half2 vectorized
};

cudaError_t fused_residual_rmsnorm_forward(const void* input,
                                           const void* residual,
                                           const void* weight,
                                           void* output,
                                           int64_t rows,
                                           int64_t hidden,
                                           float epsilon,
                                           DType dtype,
                                           FusedResidualVariant variant,
                                           cudaStream_t stream);

FusedResidualVariant fused_residual_select_variant(int64_t hidden, DType dtype);

const char* fused_residual_variant_name(FusedResidualVariant variant);

void fused_residual_reference_cpu(const float* input,
                                  const float* residual,
                                  const float* weight,
                                  float* output,
                                  int64_t rows,
                                  int64_t hidden,
                                  float epsilon);

}  // namespace hqsb
