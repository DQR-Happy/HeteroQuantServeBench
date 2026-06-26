#include "hqsb/fused_residual_rmsnorm.h"

#include "rmsnorm_launchers.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace hqsb {
namespace {

constexpr int kBlockSize = 256;

template <typename T>
__device__ __forceinline__ float to_float(T value);
template <>
__device__ __forceinline__ float to_float<float>(float value) { return value; }
template <>
__device__ __forceinline__ float to_float<__half>(__half value) {
  return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value);
template <>
__device__ __forceinline__ float from_float<float>(float value) { return value; }
template <>
__device__ __forceinline__ __half from_float<__half>(float value) {
  return __float2half_rn(value);
}

__host__ __device__ constexpr size_t align_float(size_t bytes) {
  return (bytes + alignof(float) - 1) & ~(alignof(float) - 1);
}

template <typename T, bool kWarpShuffle>
__global__ void fused_residual_rmsnorm_semantic_a_kernel(
    const T* __restrict__ input,
    const T* __restrict__ residual,
    const T* __restrict__ weight,
    T* __restrict__ residual_out,
    T* __restrict__ output,
    int hidden,
    float epsilon) {
  extern __shared__ unsigned char raw_shared[];
  T* staged = reinterpret_cast<T*>(raw_shared);
  float* reduction = reinterpret_cast<float*>(
      raw_shared + align_float(sizeof(T) * hidden));
  const int row = static_cast<int>(blockIdx.x);
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int num_warps = blockDim.x >> 5;
  const size_t row_offset = static_cast<size_t>(row) * hidden;

  float local = 0.0F;
  for (int column = thread; column < hidden; column += blockDim.x) {
    const size_t index = row_offset + column;
    // Materialise the low-precision cast before both the square and y. This
    // is the exact rounding point of semantic A.
    const T rounded = from_float<T>(to_float<T>(input[index]) +
                                    to_float<T>(residual[index]));
    residual_out[index] = rounded;
    staged[column] = rounded;
    const float value = to_float<T>(rounded);
    local += value * value;
  }

  if constexpr (kWarpShuffle) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      local += __shfl_down_sync(0xFFFFFFFFu, local, offset);
    }
    if (lane == 0) reduction[warp] = local;
    __syncthreads();
    float total = 0.0F;
    if (warp == 0) {
      total = lane < num_warps ? reduction[lane] : 0.0F;
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_down_sync(0xFFFFFFFFu, total, offset);
      }
      if (lane == 0) reduction[0] = total;
    }
  } else {
    reduction[thread] = local;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (thread < static_cast<int>(stride)) {
        reduction[thread] += reduction[thread + stride];
      }
      __syncthreads();
    }
  }
  __syncthreads();

  const float inverse_rms =
      rsqrtf(reduction[0] / static_cast<float>(hidden) + epsilon);
  for (int column = thread; column < hidden; column += blockDim.x) {
    const size_t index = row_offset + column;
    output[index] = from_float<T>(to_float<T>(staged[column]) *
                                  to_float<T>(weight[column]) * inverse_rms);
  }
}

template <typename T>
__global__ void residual_add_kernel(const T* __restrict__ input,
                                    const T* __restrict__ residual,
                                    T* __restrict__ residual_out,
                                    size_t elements) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    residual_out[index] = from_float<T>(to_float<T>(input[index]) +
                                        to_float<T>(residual[index]));
  }
}

size_t element_bytes(DType dtype) {
  switch (dtype) {
    case DType::kFloat32: return sizeof(float);
    case DType::kFloat16: return sizeof(__half);
  }
  return 0;
}

bool valid_shape(int64_t rows, int64_t hidden, size_t* elements) {
  if (rows < 1 || hidden < 1 || hidden > kFusedResidualMaxHidden ||
      rows > std::numeric_limits<int>::max()) return false;
  const uint64_t r = static_cast<uint64_t>(rows);
  const uint64_t h = static_cast<uint64_t>(hidden);
  if (r > std::numeric_limits<size_t>::max() / h) return false;
  *elements = static_cast<size_t>(r * h);
  return true;
}

bool byte_range(uintptr_t begin, size_t bytes, uintptr_t* end) {
  if (bytes > std::numeric_limits<uintptr_t>::max() - begin) return false;
  *end = begin + bytes;
  return true;
}

bool overlaps(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
  uintptr_t ae = 0, be = 0;
  const uintptr_t ab = reinterpret_cast<uintptr_t>(a);
  const uintptr_t bb = reinterpret_cast<uintptr_t>(b);
  if (!byte_range(ab, a_bytes, &ae) || !byte_range(bb, b_bytes, &be)) {
    return true;
  }
  return ab < be && bb < ae;
}

bool outputs_disjoint(const void* input,
                      const void* residual,
                      const void* weight,
                      const void* residual_out,
                      const void* output,
                      size_t tensor_bytes,
                      size_t weight_bytes) {
  return !overlaps(residual_out, tensor_bytes, input, tensor_bytes) &&
         !overlaps(residual_out, tensor_bytes, residual, tensor_bytes) &&
         !overlaps(residual_out, tensor_bytes, weight, weight_bytes) &&
         !overlaps(output, tensor_bytes, input, tensor_bytes) &&
         !overlaps(output, tensor_bytes, residual, tensor_bytes) &&
         !overlaps(output, tensor_bytes, weight, weight_bytes) &&
         !overlaps(output, tensor_bytes, residual_out, tensor_bytes);
}

bool validate_common(const void* input,
                     const void* residual,
                     const void* weight,
                     void* residual_out,
                     void* output,
                     int64_t rows,
                     int64_t hidden,
                     float epsilon,
                     DType dtype,
                     FusedResidualSemantic semantic,
                     size_t* elements) {
  if (!input || !residual || !weight || !residual_out || !output ||
      !std::isfinite(epsilon) || epsilon <= 0.0F ||
      semantic != FusedResidualSemantic::kStrictRoundedResidual ||
      !valid_shape(rows, hidden, elements)) return false;
  const size_t width = element_bytes(dtype);
  if (width == 0 || *elements > std::numeric_limits<size_t>::max() / width ||
      static_cast<size_t>(hidden) >
          std::numeric_limits<size_t>::max() / width) return false;
  return outputs_disjoint(input, residual, weight, residual_out, output,
                          *elements * width,
                          static_cast<size_t>(hidden) * width);
}

template <typename T, bool kWarpShuffle>
cudaError_t launch_fused(const void* input,
                         const void* residual,
                         const void* weight,
                         void* residual_out,
                         void* output,
                         int64_t rows,
                         int64_t hidden,
                         float epsilon,
                         cudaStream_t stream) {
  const size_t reduction_slots = kWarpShuffle ? kBlockSize / 32 : kBlockSize;
  const size_t shared_bytes = align_float(sizeof(T) * hidden) +
                              reduction_slots * sizeof(float);
  fused_residual_rmsnorm_semantic_a_kernel<T, kWarpShuffle>
      <<<static_cast<unsigned int>(rows), kBlockSize, shared_bytes, stream>>>(
          static_cast<const T*>(input), static_cast<const T*>(residual),
          static_cast<const T*>(weight), static_cast<T*>(residual_out),
          static_cast<T*>(output), static_cast<int>(hidden), epsilon);
  return cudaGetLastError();
}

}  // namespace

FusedResidualVariant fused_residual_select_variant(int64_t, DType) {
  return FusedResidualVariant::kV1WarpShuffle;
}

const char* fused_residual_variant_name(FusedResidualVariant variant) {
  switch (variant) {
    case FusedResidualVariant::kAuto: return "auto";
    case FusedResidualVariant::kReference: return "reference";
    case FusedResidualVariant::kV0SharedTree: return "v0_shared_tree";
    case FusedResidualVariant::kV1WarpShuffle: return "v1_warp_shuffle";
  }
  return "unknown";
}

size_t fused_residual_dynamic_shared_bytes(int64_t hidden,
                                           DType dtype,
                                           FusedResidualVariant variant) {
  const size_t width = element_bytes(dtype);
  if (hidden < 1 || hidden > kFusedResidualMaxHidden || width == 0) return 0;
  if (variant == FusedResidualVariant::kAuto) {
    variant = fused_residual_select_variant(hidden, dtype);
  }
  const size_t slots = variant == FusedResidualVariant::kV0SharedTree
                           ? kBlockSize
                           : kBlockSize / 32;
  return align_float(static_cast<size_t>(hidden) * width) +
         slots * sizeof(float);
}

int fused_residual_occupancy(int64_t hidden,
                             DType dtype,
                             FusedResidualVariant variant) {
  if (variant == FusedResidualVariant::kAuto) {
    variant = fused_residual_select_variant(hidden, dtype);
  }
  const size_t shared = fused_residual_dynamic_shared_bytes(hidden, dtype, variant);
  if (shared == 0) return 0;
  int blocks = 0;
  if (dtype == DType::kFloat16) {
    if (variant == FusedResidualVariant::kV0SharedTree) {
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, fused_residual_rmsnorm_semantic_a_kernel<__half, false>,
          kBlockSize, shared);
    } else {
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, fused_residual_rmsnorm_semantic_a_kernel<__half, true>,
          kBlockSize, shared);
    }
  } else if (dtype == DType::kFloat32) {
    if (variant == FusedResidualVariant::kV0SharedTree) {
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, fused_residual_rmsnorm_semantic_a_kernel<float, false>,
          kBlockSize, shared);
    } else {
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, fused_residual_rmsnorm_semantic_a_kernel<float, true>,
          kBlockSize, shared);
    }
  }
  return blocks;
}

cudaError_t fused_residual_rmsnorm_forward(
    const void* input, const void* residual, const void* weight,
    void* residual_out, void* output, int64_t rows, int64_t hidden,
    float epsilon, DType dtype, FusedResidualVariant variant,
    FusedResidualSemantic semantic, cudaStream_t stream) {
  size_t elements = 0;
  if (!validate_common(input, residual, weight, residual_out, output, rows,
                       hidden, epsilon, dtype, semantic, &elements)) {
    return cudaErrorInvalidValue;
  }
  if (variant == FusedResidualVariant::kAuto) {
    variant = fused_residual_select_variant(hidden, dtype);
  }
  if (variant == FusedResidualVariant::kReference) return cudaErrorInvalidValue;
  if (dtype == DType::kFloat16) {
    if (variant == FusedResidualVariant::kV0SharedTree) {
      return launch_fused<__half, false>(input, residual, weight, residual_out,
                                        output, rows, hidden, epsilon, stream);
    }
    if (variant == FusedResidualVariant::kV1WarpShuffle) {
      return launch_fused<__half, true>(input, residual, weight, residual_out,
                                       output, rows, hidden, epsilon, stream);
    }
  } else if (dtype == DType::kFloat32) {
    if (variant == FusedResidualVariant::kV0SharedTree) {
      return launch_fused<float, false>(input, residual, weight, residual_out,
                                       output, rows, hidden, epsilon, stream);
    }
    if (variant == FusedResidualVariant::kV1WarpShuffle) {
      return launch_fused<float, true>(input, residual, weight, residual_out,
                                      output, rows, hidden, epsilon, stream);
    }
  }
  return cudaErrorInvalidValue;
}

cudaError_t residual_add_forward(const void* input,
                                 const void* residual,
                                 void* residual_out,
                                 int64_t rows,
                                 int64_t hidden,
                                 DType dtype,
                                 cudaStream_t stream) {
  size_t elements = 0;
  if (!input || !residual || !residual_out ||
      !valid_shape(rows, hidden, &elements)) return cudaErrorInvalidValue;
  const int threads = 256;
  const unsigned int blocks = static_cast<unsigned int>(
      (elements + static_cast<size_t>(threads) - 1) / threads);
  if (dtype == DType::kFloat16) {
    residual_add_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __half*>(input), static_cast<const __half*>(residual),
        static_cast<__half*>(residual_out), elements);
  } else if (dtype == DType::kFloat32) {
    residual_add_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(input), static_cast<const float*>(residual),
        static_cast<float*>(residual_out), elements);
  } else {
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

cudaError_t separate_residual_rmsnorm_forward(
    const void* input, const void* residual, const void* weight,
    void* residual_out, void* output, int64_t rows, int64_t hidden,
    float epsilon, DType dtype, FusedResidualSemantic semantic,
    cudaStream_t stream) {
  size_t elements = 0;
  if (!validate_common(input, residual, weight, residual_out, output, rows,
                       hidden, epsilon, dtype, semantic, &elements)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t error = residual_add_forward(input, residual, residual_out, rows,
                                           hidden, dtype, stream);
  if (error != cudaSuccess) return error;
  return rmsnorm_v2(residual_out, weight, output, rows, hidden, epsilon, dtype,
                    kBlockSize, stream);
}

}  // namespace hqsb
