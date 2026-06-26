#include "hqsb/rmsnorm.h"

#include "rmsnorm_launchers.h"

#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────────────
// Stable C ABI for cross-language (Python ctypes) consumption.
//
// The C++ API (hqsb::rmsnorm_forward) lives in a namespace and is therefore
// subject to name mangling. This file exports an unmangled ``extern "C"``
// surface that Python can load via ``ctypes`` without hard-coding a mangled
// symbol, and without introducing a heavy binding dependency (pybind11,
// torch extension, etc.).
//
// dtype encoding matches the Python-facing dispatcher:
//   0 = float32, 1 = float16
//   anything else is an *error*, never a silent coercion to float32
//   (E03-01 case G: a silently coerced dtype would make a "forced" test lie
//   about which implementation actually ran).
// variant encoding matches hqsb::RmsNormVariant:
//   0 = auto, 1 = reference (rejected — CPU-only oracle), 2 = v0, 3 = v1,
//   4 = v2 compatibility/hybrid, 5 = scalar-safe, 6 = strict vector
//   anything else is an *error*, never a silent fallback to auto.
//
// Two entry points are exported:
//   * ``hqsb_rmsnorm_forward_ex_c`` — explicit ``cudaStream_t`` (E03-01 step 5
//     requires every forced call to name its stream); the stream is passed as
//     a raw pointer so ctypes can hand over ``torch.cuda.Stream().cuda_stream``
//     or NULL for the default stream.
//   * ``hqsb_rmsnorm_forward_c`` — the original NULL-stream (default stream)
//     signature, preserved verbatim so existing callers keep working. It is a
//     thin delegate to the ``_ex`` form.
// ─────────────────────────────────────────────────────────────────────────

namespace {

// Returns true and fills ``out`` for a valid dtype code, false otherwise.
// Deliberately strict: an unknown code must not be coerced.
bool decode_dtype(int dtype, hqsb::DType* out) {
  switch (dtype) {
    case 0:
      *out = hqsb::DType::kFloat32;
      return true;
    case 1:
      *out = hqsb::DType::kFloat16;
      return true;
    default:
      return false;
  }
}

// Returns true and fills ``out`` for a valid variant code, false otherwise.
// ``1`` (kReference, CPU FP64 oracle) decodes to kReference on purpose so the
// C++ layer rejects it explicitly instead of silently running a GPU variant.
bool decode_variant(int variant, hqsb::RmsNormVariant* out) {
  switch (variant) {
    case 0:
      *out = hqsb::RmsNormVariant::kAuto;
      return true;
    case 1:
      *out = hqsb::RmsNormVariant::kReference;
      return true;
    case 2:
      *out = hqsb::RmsNormVariant::kV0Shared;
      return true;
    case 3:
      *out = hqsb::RmsNormVariant::kV1WarpShuffle;
      return true;
    case 4:
      *out = hqsb::RmsNormVariant::kV2Vectorized;
      return true;
    case 5:
      *out = hqsb::RmsNormVariant::kScalarSafe;
      return true;
    case 6:
      *out = hqsb::RmsNormVariant::kV2VectorizedStrict;
      return true;
    default:
      return false;
  }
}

}  // namespace

extern "C" {

int hqsb_rmsnorm_forward_ex_c(const void* input,
                              const void* weight,
                              void* output,
                              long long rows,
                              long long hidden,
                              float epsilon,
                              int dtype,
                              int variant,
                              void* stream) {
  hqsb::DType dt;
  if (!decode_dtype(dtype, &dt)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  hqsb::RmsNormVariant v;
  if (!decode_variant(variant, &v)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const cudaError_t err = hqsb::rmsnorm_forward(
      input, weight, output, static_cast<int64_t>(rows),
      static_cast<int64_t>(hidden), epsilon, dt, v,
      reinterpret_cast<cudaStream_t>(stream));
  return static_cast<int>(err);
}

int hqsb_rmsnorm_forward_c(const void* input,
                           const void* weight,
                           void* output,
                           long long rows,
                           long long hidden,
                           float epsilon,
                           int dtype,
                           int variant) {
  return hqsb_rmsnorm_forward_ex_c(input, weight, output, rows, hidden,
                                   epsilon, dtype, variant,
                                   /*stream=*/nullptr);
}

// Audit-only launch surface used by E03-04.  Unlike the production dispatcher,
// this entry point makes the launch configuration explicit so every measured
// (variant, block_size) cell really reaches the corresponding launcher.  It is
// intentionally strict: only full-warp power-of-two blocks supported by all
// three teaching kernels are accepted, and V0/V1 remain FP32-only.
int hqsb_rmsnorm_forward_config_ex_c(const void* input,
                                     const void* weight,
                                     void* output,
                                     long long rows,
                                     long long hidden,
                                     float epsilon,
                                     int dtype,
                                     int variant,
                                     int block_size,
                                     void* stream) {
  hqsb::DType dt;
  hqsb::RmsNormVariant v;
  if (input == nullptr || weight == nullptr || output == nullptr || rows < 1 ||
      hidden < 1 || !(epsilon > 0.0F) || !decode_dtype(dtype, &dt) ||
      !decode_variant(variant, &v) || block_size < 32 || block_size > 1024 ||
      (block_size & (block_size - 1)) != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  switch (v) {
    case hqsb::RmsNormVariant::kV0Shared:
      return dt == hqsb::DType::kFloat32
                 ? static_cast<int>(hqsb::rmsnorm_v0(
                       static_cast<const float*>(input),
                       static_cast<const float*>(weight),
                       static_cast<float*>(output), static_cast<int64_t>(rows),
                       static_cast<int64_t>(hidden), epsilon, block_size,
                       cuda_stream))
                 : static_cast<int>(cudaErrorInvalidValue);
    case hqsb::RmsNormVariant::kV1WarpShuffle:
      return dt == hqsb::DType::kFloat32
                 ? static_cast<int>(hqsb::rmsnorm_v1(
                       static_cast<const float*>(input),
                       static_cast<const float*>(weight),
                       static_cast<float*>(output), static_cast<int64_t>(rows),
                       static_cast<int64_t>(hidden), epsilon, block_size,
                       cuda_stream))
                 : static_cast<int>(cudaErrorInvalidValue);
    case hqsb::RmsNormVariant::kV2Vectorized:
    case hqsb::RmsNormVariant::kV2VectorizedStrict:
      return static_cast<int>(hqsb::rmsnorm_v2(
          input, weight, output, static_cast<int64_t>(rows),
          static_cast<int64_t>(hidden), epsilon, dt, block_size, cuda_stream));
    default:
      return static_cast<int>(cudaErrorInvalidValue);
  }
}

// Resolve without launching.  Output fields are written only on success.
// ``reason_mask`` uses hqsb::RmsNormDispatchReason bits from the public
// header; the Python audit computes an independent oracle and compares every
// field rather than calling this helper to derive its expectation.
int hqsb_rmsnorm_resolve_dispatch_c(const void* input,
                                    const void* weight,
                                    const void* output,
                                    long long hidden,
                                    int dtype,
                                    int variant,
                                    int* actual_variant,
                                    unsigned int* reason_mask,
                                    int* vector_width_elements,
                                    int* required_alignment_bytes,
                                    int* actual_load_width_bytes) {
  hqsb::DType dt;
  hqsb::RmsNormVariant requested;
  if (!decode_dtype(dtype, &dt) || !decode_variant(variant, &requested) ||
      actual_variant == nullptr || reason_mask == nullptr ||
      vector_width_elements == nullptr ||
      required_alignment_bytes == nullptr ||
      actual_load_width_bytes == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  hqsb::RmsNormDispatchInfo info{};
  const cudaError_t err = hqsb::rmsnorm_resolve_dispatch(
      input, weight, output, static_cast<int64_t>(hidden), dt, requested,
      &info);
  *actual_variant = static_cast<int>(info.actual_variant);
  *reason_mask = info.reason_mask;
  *vector_width_elements = info.vector_width_elements;
  *required_alignment_bytes = info.required_alignment_bytes;
  *actual_load_width_bytes = info.actual_load_width_bytes;
  return static_cast<int>(err);
}

// Reports the compute capability this shared library was *compiled* for.
//
// The Python dispatcher (ops/dispatcher.py) needs the build arch to decide
// whether the precompiled kernels may be used on the runtime device. Baking
// that arch into Python would hard-code one platform and silently
// mis-dispatch on every other, so the library owns the fact instead.
//
// Returns 0 and writes (major, minor) on success; 1 for a null out-pointer;
// 2 when the build system did not record an arch (unknown -> the dispatcher
// must not assume compatibility).
int hqsb_rmsnorm_query_build_arch(int* major, int* minor) {
  if (major == nullptr || minor == nullptr) {
    return 1;
  }
#if defined(HQSB_RMSNORM_BUILD_ARCH_MAJOR) && \
    defined(HQSB_RMSNORM_BUILD_ARCH_MINOR)
  *major = HQSB_RMSNORM_BUILD_ARCH_MAJOR;
  *minor = HQSB_RMSNORM_BUILD_ARCH_MINOR;
  return 0;
#else
  *major = 0;
  *minor = 0;
  return 2;
#endif
}

// Theoretical max active blocks per SM for a (variant, dtype, block_size)
// triple. Added by E03-02 step 1 so the performance matrix can record the
// occupancy *resource* column from the same binary it times; it computes a
// launch-configuration property only and does not change any kernel, the
// dispatcher or the forward path. Returns 0 for an unsupported/invalid
// combination (never a silent fallback to another variant).
int hqsb_rmsnorm_occupancy_c(int variant, int dtype, int block_size) {
  hqsb::DType dt;
  if (!decode_dtype(dtype, &dt)) {
    return 0;
  }
  hqsb::RmsNormVariant v;
  if (!decode_variant(variant, &v)) {
    return 0;
  }
  if (block_size < 32 || block_size > 1024 || (block_size % 32) != 0) {
    return 0;
  }
  switch (v) {
    case hqsb::RmsNormVariant::kV0Shared:
      return (dt == hqsb::DType::kFloat32)
                 ? hqsb::rmsnorm_v0_occupancy(block_size)
                 : 0;
    case hqsb::RmsNormVariant::kV1WarpShuffle:
      return (dt == hqsb::DType::kFloat32)
                 ? hqsb::rmsnorm_v1_occupancy(block_size)
                 : 0;
    case hqsb::RmsNormVariant::kV2Vectorized:
    case hqsb::RmsNormVariant::kV2VectorizedStrict:
      return hqsb::rmsnorm_v2_occupancy(dt, block_size);
    case hqsb::RmsNormVariant::kScalarSafe:
      return hqsb::rmsnorm_scalar_safe_occupancy(dt, block_size);
    default:
      return 0;
  }
}

}  // extern "C"
