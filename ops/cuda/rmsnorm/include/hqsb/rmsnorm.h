#pragma once
//
// HQSB RMSNorm operator — public C/C++ API (S03).
//
// This header is the *only* interface a downstream consumer (S04 Triton
// wrappers, S06 framework integration, S09 Ascend C port) needs to call
// RMSNorm. It is deliberately free of Python, serving, and config
// dependencies; everything below is callable from a plain CUDA runtime.
//
// Design contract:
//   * stream-aware   — every kernel launches on the caller-provided stream.
//   * no hidden alloc — kernels never call cudaMalloc; the caller owns all
//                       device buffers and their lifetimes.
//   * no sync        — the caller decides when to synchronize.
//

#include <cuda_runtime.h>

#include <cstdint>

namespace hqsb {

// Element data type supported by the operator library.
enum class DType : int {
  kFloat32 = 0,
  kFloat16 = 1,
};

// RMSNorm implementation variants. ``kAuto`` defers to the dispatcher.
enum class RmsNormVariant : int {
  kAuto = 0,
  kReference = 1,   // CPU FP64 reference (test/benchmark only; never on device)
  kV0Shared = 2,    // V0: shared-memory tree reduction (S00 baseline)
  kV1WarpShuffle = 3,  // V1: warp-shuffle reduction + warp-level combine
  kV2Vectorized = 4,   // V2: vectorized (float4/half2) load + warp shuffle
};

// Forward RMSNorm: out = (in / rms(in)) * weight, per row.
//
// Args:
//   input    device pointer, shape (rows, hidden), row-major
//   weight   device pointer, shape (hidden,)
//   output   device pointer, shape (rows, hidden), row-major (may alias input)
//   rows     number of rows (>= 1)
//   hidden   number of columns (>= 1)
//   epsilon  denominator stabilizer (added to mean-square before rsqrt)
//   dtype    element type of input/weight/output
//   variant  which implementation to use (kAuto -> dispatcher)
//   stream   CUDA stream to launch on
//
// Returns cudaSuccess or the first CUDA error encountered.
cudaError_t rmsnorm_forward(const void* input,
                            const void* weight,
                            void* output,
                            int64_t rows,
                            int64_t hidden,
                            float epsilon,
                            DType dtype,
                            RmsNormVariant variant,
                            cudaStream_t stream);

// Choose the best variant for the given shape/dtype. Returns kV2Vectorized
// for aligned FP32/FP16, kV1WarpShuffle otherwise, and never kReference or
// kAuto.
RmsNormVariant rmsnorm_select_variant(int64_t hidden, DType dtype);

// Human-readable name for a variant (for logs and benchmark tables).
const char* rmsnorm_variant_name(RmsNormVariant variant);

// CPU reference (FP64 accumulation) — independent of the device kernels and
// of the benchmark harness, so tests never share the code under test.
void rmsnorm_reference_cpu(const float* input,
                           const float* weight,
                           float* output,
                           int64_t rows,
                           int64_t hidden,
                           float epsilon);

}  // namespace hqsb
