#include "hqsb/rmsnorm.h"

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
// variant encoding matches hqsb::RmsNormVariant:
//   0 = auto, 2 = v0, 3 = v1, 4 = v2
// All kernels launch on the default stream (stream 0); the Python binding
// layer does its own torch.cuda.synchronize() around the call.
// ─────────────────────────────────────────────────────────────────────────

extern "C" {

int hqsb_rmsnorm_forward_c(const void* input,
                           const void* weight,
                           void* output,
                           long long rows,
                           long long hidden,
                           float epsilon,
                           int dtype,
                           int variant) {
  hqsb::DType dt = (dtype == 1) ? hqsb::DType::kFloat16 : hqsb::DType::kFloat32;
  hqsb::RmsNormVariant v;
  switch (variant) {
    case 2:
      v = hqsb::RmsNormVariant::kV0Shared;
      break;
    case 3:
      v = hqsb::RmsNormVariant::kV1WarpShuffle;
      break;
    case 4:
      v = hqsb::RmsNormVariant::kV2Vectorized;
      break;
    case 0:
    default:
      v = hqsb::RmsNormVariant::kAuto;
      break;
  }

  cudaError_t err = hqsb::rmsnorm_forward(
      input, weight, output, static_cast<int64_t>(rows),
      static_cast<int64_t>(hidden), epsilon, dt, v, /*stream=*/0);
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

}  // extern "C"
