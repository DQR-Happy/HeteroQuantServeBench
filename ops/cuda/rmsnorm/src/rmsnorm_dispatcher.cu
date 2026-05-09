#include "hqsb/rmsnorm.h"

#include "rmsnorm_launchers.h"

#include <cuda_runtime.h>

namespace hqsb {

namespace {

// Default block size for dispatcher-selected launches. 256 threads gives a
// good balance between occupancy and per-row work for the target hidden
// sizes (256..4096). The benchmark harness sweeps block sizes directly via
// the launchers.
constexpr int kDefaultBlockSize = 256;

// The V0/V1 kernels are FP32-only by construction (they pre-date FP16
// support and were written against the S00 FP32 baseline). FP16 paths are
// served by V2 only.
bool variant_supports_dtype(RmsNormVariant variant, DType dtype) {
  if (variant == RmsNormVariant::kV2Vectorized) {
    return true;
  }
  return dtype == DType::kFloat32;
}

}  // namespace

RmsNormVariant rmsnorm_select_variant(int64_t hidden, DType dtype) {
  // V2 (vectorized + warp shuffle) is the only variant that supports FP16,
  // and its half2 kernel handles *any* column count via a scalar tail, so
  // FP16 always routes to V2. V1 remains the FP32 fallback for non-multiple-
  // of-4 column counts (where float4 vectorization yields nothing).
  if (dtype == DType::kFloat16) {
    return RmsNormVariant::kV2Vectorized;
  }
  return (hidden % 4 == 0) ? RmsNormVariant::kV2Vectorized
                           : RmsNormVariant::kV1WarpShuffle;
}

cudaError_t rmsnorm_forward(const void* input,
                            const void* weight,
                            void* output,
                            int64_t rows,
                            int64_t hidden,
                            float epsilon,
                            DType dtype,
                            RmsNormVariant variant,
                            cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (rows < 1 || hidden < 1) {
    return cudaErrorInvalidValue;
  }

  RmsNormVariant chosen = variant;
  if (chosen == RmsNormVariant::kAuto) {
    chosen = rmsnorm_select_variant(hidden, dtype);
  }

  // The CPU reference is not a device implementation; reject it here so the
  // caller gets an explicit error rather than silently running nothing.
  if (chosen == RmsNormVariant::kReference) {
    return cudaErrorInvalidValue;
  }
  if (!variant_supports_dtype(chosen, dtype)) {
    // Unsupported combination: explicit, structured fallback signal.
    return cudaErrorInvalidValue;
  }

  switch (chosen) {
    case RmsNormVariant::kV0Shared:
      return rmsnorm_v0(static_cast<const float*>(input),
                        static_cast<const float*>(weight),
                        static_cast<float*>(output),
                        rows, hidden, epsilon, kDefaultBlockSize, stream);
    case RmsNormVariant::kV1WarpShuffle:
      return rmsnorm_v1(static_cast<const float*>(input),
                        static_cast<const float*>(weight),
                        static_cast<float*>(output),
                        rows, hidden, epsilon, kDefaultBlockSize, stream);
    case RmsNormVariant::kV2Vectorized:
      return rmsnorm_v2(input, weight, output, rows, hidden, epsilon, dtype,
                        kDefaultBlockSize, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace hqsb
