#include "hqsb/rmsnorm.h"

#include "rmsnorm_launchers.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

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
  if (variant == RmsNormVariant::kV2Vectorized ||
      variant == RmsNormVariant::kV2VectorizedStrict ||
      variant == RmsNormVariant::kScalarSafe) {
    return true;
  }
  return dtype == DType::kFloat32;
}

// E03-01: the operator contract is *rejected* rather than silently coerced
// when the caller violates it. Every predicate below corresponds to a frozen
// clause of the resolved OperatorSpec (see E03-01 §2/§8 in
// docs/stage_experiments/details/S03). Returning ``cudaErrorInvalidValue``
// here means the failure happens *before* any kernel launch, so a rejected
// call can never leave a partially written output buffer behind.
bool dtype_is_valid(DType dtype) {
  return dtype == DType::kFloat32 || dtype == DType::kFloat16;
}

// ``epsilon`` must be finite and strictly positive: a NaN/Inf/non-positive
// denominator stabilizer has no meaning in the frozen math contract and would
// otherwise surface as silent NaN output. ``!(epsilon > 0)`` also rejects NaN.
bool epsilon_is_valid(float epsilon) {
  return (epsilon > 0.0F) && std::isfinite(epsilon);
}

// The device kernels index with ``int`` (``static_cast<int>(hidden)``), so a
// hidden width beyond INT_MAX would silently truncate and read the wrong
// columns. Reject it instead of producing a wrong result.
bool hidden_is_representable(int64_t hidden) {
  return hidden <= static_cast<int64_t>(std::numeric_limits<int>::max());
}

int element_bytes(DType dtype) {
  return dtype == DType::kFloat16 ? 2 : 4;
}

int vector_width_elements(DType dtype) {
  return dtype == DType::kFloat16 ? 2 : 4;
}

int required_alignment_bytes(DType dtype) {
  return dtype == DType::kFloat16 ? 4 : 16;
}

bool pointer_is_aligned(const void* pointer, int alignment) {
  return (reinterpret_cast<uintptr_t>(pointer) %
          static_cast<uintptr_t>(alignment)) == 0;
}

}  // namespace

cudaError_t rmsnorm_resolve_dispatch(const void* input,
                                     const void* weight,
                                     const void* output,
                                     int64_t hidden,
                                     DType dtype,
                                     RmsNormVariant requested,
                                     RmsNormDispatchInfo* info) {
  if (input == nullptr || weight == nullptr || output == nullptr ||
      info == nullptr || hidden < 1 || !dtype_is_valid(dtype) ||
      !hidden_is_representable(hidden)) {
    return cudaErrorInvalidValue;
  }

  const int vector_width = vector_width_elements(dtype);
  const int alignment = required_alignment_bytes(dtype);
  uint32_t reason_mask = kDispatchEligibleVector;
  if ((hidden % vector_width) != 0) {
    reason_mask |= kDispatchHiddenTail;
  }
  if (!pointer_is_aligned(input, alignment)) {
    reason_mask |= kDispatchInputMisaligned;
  }
  if (!pointer_is_aligned(weight, alignment)) {
    reason_mask |= kDispatchWeightMisaligned;
  }
  if (!pointer_is_aligned(output, alignment)) {
    reason_mask |= kDispatchOutputMisaligned;
  }
  const int64_t row_stride_bytes = hidden * element_bytes(dtype);
  if ((row_stride_bytes % alignment) != 0) {
    reason_mask |= kDispatchRowStrideMisaligned;
  }
  const bool vector_eligible = reason_mask == kDispatchEligibleVector;

  RmsNormVariant actual = requested;
  if (requested == RmsNormVariant::kAuto) {
    if (vector_eligible) {
      actual = RmsNormVariant::kV2Vectorized;
    } else if (dtype == DType::kFloat32) {
      actual = RmsNormVariant::kV1WarpShuffle;
    } else {
      actual = RmsNormVariant::kScalarSafe;
    }
  } else if (requested == RmsNormVariant::kV2Vectorized) {
    // Compatibility mode used by the completed E03-01/E03-02 matrices:
    // preserve the requested family but make its scalar fallback explicit.
    actual = vector_eligible ? RmsNormVariant::kV2Vectorized
                             : RmsNormVariant::kScalarSafe;
  } else if (requested == RmsNormVariant::kV2VectorizedStrict) {
    actual = vector_eligible ? RmsNormVariant::kV2Vectorized
                             : RmsNormVariant::kV2VectorizedStrict;
  } else if (requested == RmsNormVariant::kV0Shared ||
             requested == RmsNormVariant::kV1WarpShuffle ||
             requested == RmsNormVariant::kScalarSafe) {
    if (!variant_supports_dtype(requested, dtype)) {
      return cudaErrorInvalidValue;
    }
  } else {
    return cudaErrorInvalidValue;
  }

  info->requested_variant = requested;
  info->actual_variant = actual;
  info->reason_mask = reason_mask;
  info->vector_width_elements = vector_width;
  info->required_alignment_bytes = alignment;
  info->actual_load_width_bytes =
      actual == RmsNormVariant::kV2Vectorized ? alignment
                                              : element_bytes(dtype);
  info->vector_eligible = vector_eligible;
  if (requested == RmsNormVariant::kV2VectorizedStrict && !vector_eligible) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

RmsNormVariant rmsnorm_select_variant(int64_t hidden, DType dtype) {
  // This shape-only helper cannot inspect concrete pointer alignment.  It
  // preserves the historical V2-family answer for every FP16 shape;
  // rmsnorm_resolve_dispatch is authoritative for the actual vector versus
  // scalar-safe load path of a concrete call.
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
  if (!dtype_is_valid(dtype)) {
    return cudaErrorInvalidValue;
  }
  if (!epsilon_is_valid(epsilon)) {
    return cudaErrorInvalidValue;
  }
  if (!hidden_is_representable(hidden)) {
    return cudaErrorInvalidValue;
  }

  RmsNormDispatchInfo dispatch{};
  const cudaError_t resolve_error = rmsnorm_resolve_dispatch(
      input, weight, output, hidden, dtype, variant, &dispatch);
  if (resolve_error != cudaSuccess) {
    return resolve_error;
  }
  const RmsNormVariant chosen = dispatch.actual_variant;

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
    case RmsNormVariant::kScalarSafe:
      return rmsnorm_scalar_safe(input, weight, output, rows, hidden, epsilon,
                                 dtype, kDefaultBlockSize, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace hqsb
