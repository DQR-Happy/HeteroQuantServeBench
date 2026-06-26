#pragma once
// HQSB fused residual-add + RMSNorm operator (E03-05).
//
// Frozen semantic A (version 1):
//   residual_out = cast_dtype(float(input) + float(residual))
//   y = cast_dtype(float(residual_out) * float(weight) *
//                  rsqrt(mean(float(residual_out)^2) + epsilon))
//
// Both outputs are required. The implementation launches on the supplied
// stream, performs no allocation/synchronisation, and uses caller-owned
// storage only. Output ranges must be disjoint from every input and from one
// another; arbitrary partial overlap is rejected before launch.

#include "hqsb/rmsnorm.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace hqsb {

enum class FusedResidualVariant : int {
  kAuto = 0,
  kReference = 1,
  kV0SharedTree = 2,
  kV1WarpShuffle = 3,
};

enum class FusedResidualSemantic : int {
  kStrictRoundedResidual = 1,
};

constexpr int64_t kFusedResidualMaxHidden = 8192;

cudaError_t fused_residual_rmsnorm_forward(
    const void* input,
    const void* residual,
    const void* weight,
    void* residual_out,
    void* output,
    int64_t rows,
    int64_t hidden,
    float epsilon,
    DType dtype,
    FusedResidualVariant variant,
    FusedResidualSemantic semantic,
    cudaStream_t stream);

// Formal two-launch baseline: residual add followed by the E03-02 V2
// RMSNorm implementation, on the same stream with no intervening sync.
cudaError_t separate_residual_rmsnorm_forward(
    const void* input,
    const void* residual,
    const void* weight,
    void* residual_out,
    void* output,
    int64_t rows,
    int64_t hidden,
    float epsilon,
    DType dtype,
    FusedResidualSemantic semantic,
    cudaStream_t stream);

cudaError_t residual_add_forward(const void* input,
                                 const void* residual,
                                 void* residual_out,
                                 int64_t rows,
                                 int64_t hidden,
                                 DType dtype,
                                 cudaStream_t stream);

FusedResidualVariant fused_residual_select_variant(int64_t hidden,
                                                   DType dtype);
const char* fused_residual_variant_name(FusedResidualVariant variant);
size_t fused_residual_dynamic_shared_bytes(int64_t hidden,
                                           DType dtype,
                                           FusedResidualVariant variant);
int fused_residual_occupancy(int64_t hidden,
                             DType dtype,
                             FusedResidualVariant variant);

}  // namespace hqsb
