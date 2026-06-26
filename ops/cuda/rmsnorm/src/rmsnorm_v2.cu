#include "hqsb/rmsnorm.h"
#include "rmsnorm_launchers.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace hqsb {

namespace {

// V2 — vectorized loads + warp-shuffle reduction.
//
// Hypotheses:
//   (a) memory-bound elementwise kernels are limited by the number of
//       memory transactions, not by arithmetic; a float4 (16 B) or half2
//       (4 B) load issues fewer transactions per element than scalar loads,
//       raising achieved bandwidth;
//   (b) FP16 halves the bytes moved and the storage, which matters when the
//       kernel is DRAM-bound.
//
// The reduction strategy is identical to V1 (warp shuffle + one shared slot
// per warp); only the load/store path is vectorized.

// ── FP32, float4-vectorized ───────────────────────────────────────────

__global__ void rmsnorm_v2_f32_kernel(const float* __restrict__ input,
                                      const float* __restrict__ weight,
                                      float* __restrict__ output,
                                      int hidden,
                                      float epsilon) {
  extern __shared__ float warp_square_sum[];

  const int row = static_cast<int>(blockIdx.x);
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int num_warps = blockDim.x >> 5;
  const size_t row_offset = static_cast<size_t>(row) * hidden;
  const float* row_in = input + row_offset;
  float* row_out = output + row_offset;

  // The host launcher admits this kernel only when input/weight/output are
  // 16-byte aligned and every row preserves that alignment.  Keeping the
  // fast kernel free of a runtime alignment branch makes its SASS auditable.
  const int vec_count = hidden >> 2;

  float local_square_sum = 0.0F;
  const float4* row_in4 = reinterpret_cast<const float4*>(row_in);

  for (int i = thread; i < vec_count; i += blockDim.x) {
    const float4 v = row_in4[i];
    local_square_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_square_sum +=
        __shfl_down_sync(0xFFFFFFFFu, local_square_sum, offset);
  }
  if (lane == 0) {
    warp_square_sum[warp] = local_square_sum;
  }
  __syncthreads();

  float total = 0.0F;
  if (warp == 0) {
    total = (lane < num_warps) ? warp_square_sum[lane] : 0.0F;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      total += __shfl_down_sync(0xFFFFFFFFu, total, offset);
    }
  }
  if (warp == 0 && lane == 0) {
    warp_square_sum[0] = total;
  }
  __syncthreads();

  const float inverse_rms =
      rsqrtf(warp_square_sum[0] / static_cast<float>(hidden) + epsilon);

  float4* row_out4 = reinterpret_cast<float4*>(row_out);
  const float4* weight4 = reinterpret_cast<const float4*>(weight);
  for (int i = thread; i < vec_count; i += blockDim.x) {
    const float4 v = row_in4[i];
    const float4 w = weight4[i];
    float4 o;
    o.x = v.x * inverse_rms * w.x;
    o.y = v.y * inverse_rms * w.y;
    o.z = v.z * inverse_rms * w.z;
    o.w = v.w * inverse_rms * w.w;
    row_out4[i] = o;
  }
}

// ── FP16, half2-vectorized ────────────────────────────────────────────

__global__ void rmsnorm_v2_f16_kernel(const __half* __restrict__ input,
                                      const __half* __restrict__ weight,
                                      __half* __restrict__ output,
                                      int hidden,
                                      float epsilon) {
  extern __shared__ float warp_square_sum[];

  const int row = static_cast<int>(blockIdx.x);
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int num_warps = blockDim.x >> 5;
  const size_t row_offset = static_cast<size_t>(row) * hidden;
  const __half* row_in = input + row_offset;
  __half* row_out = output + row_offset;

  // The host launcher admits this kernel only when all pointers and all row
  // starts meet half2's 4-byte alignment requirement.
  const int vec_count = hidden >> 1;

  float local_square_sum = 0.0F;
  const __half2* row_in2 = reinterpret_cast<const __half2*>(row_in);

  for (int i = thread; i < vec_count; i += blockDim.x) {
    const float2 v = __half22float2(row_in2[i]);
    local_square_sum += v.x * v.x + v.y * v.y;
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_square_sum +=
        __shfl_down_sync(0xFFFFFFFFu, local_square_sum, offset);
  }
  if (lane == 0) {
    warp_square_sum[warp] = local_square_sum;
  }
  __syncthreads();

  float total = 0.0F;
  if (warp == 0) {
    total = (lane < num_warps) ? warp_square_sum[lane] : 0.0F;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      total += __shfl_down_sync(0xFFFFFFFFu, total, offset);
    }
  }
  if (warp == 0 && lane == 0) {
    warp_square_sum[0] = total;
  }
  __syncthreads();

  const float inverse_rms =
      rsqrtf(warp_square_sum[0] / static_cast<float>(hidden) + epsilon);

  __half2* row_out2 = reinterpret_cast<__half2*>(row_out);
  const __half2* weight2 = reinterpret_cast<const __half2*>(weight);
  for (int i = thread; i < vec_count; i += blockDim.x) {
    const float2 v = __half22float2(row_in2[i]);
    const float2 w = __half22float2(weight2[i]);
    float2 o;
    o.x = v.x * inverse_rms * w.x;
    o.y = v.y * inverse_rms * w.y;
    row_out2[i] = __float22half2_rn(o);
  }
}

// ── Scalar-safe FP32 / FP16 fallback ─────────────────────────────────

template <typename T>
__device__ __forceinline__ float to_float(T value);

template <>
__device__ __forceinline__ float to_float<float>(float value) {
  return value;
}

template <>
__device__ __forceinline__ float to_float<__half>(__half value) {
  return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ float from_float<float>(float value) {
  return value;
}

template <>
__device__ __forceinline__ __half from_float<__half>(float value) {
  return __float2half_rn(value);
}

template <typename T>
__global__ void rmsnorm_scalar_safe_kernel(const T* __restrict__ input,
                                           const T* __restrict__ weight,
                                           T* __restrict__ output,
                                           int hidden,
                                           float epsilon) {
  extern __shared__ float warp_square_sum[];
  const int row = static_cast<int>(blockIdx.x);
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int num_warps = blockDim.x >> 5;
  const size_t row_offset = static_cast<size_t>(row) * hidden;

  float local_square_sum = 0.0F;
  for (int column = thread; column < hidden; column += blockDim.x) {
    const float value = to_float<T>(input[row_offset + column]);
    local_square_sum += value * value;
  }
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_square_sum +=
        __shfl_down_sync(0xFFFFFFFFu, local_square_sum, offset);
  }
  if (lane == 0) {
    warp_square_sum[warp] = local_square_sum;
  }
  __syncthreads();

  float total = 0.0F;
  if (warp == 0) {
    total = (lane < num_warps) ? warp_square_sum[lane] : 0.0F;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      total += __shfl_down_sync(0xFFFFFFFFu, total, offset);
    }
  }
  if (warp == 0 && lane == 0) {
    warp_square_sum[0] = total;
  }
  __syncthreads();

  const float inverse_rms =
      rsqrtf(warp_square_sum[0] / static_cast<float>(hidden) + epsilon);
  for (int column = thread; column < hidden; column += blockDim.x) {
    const size_t index = row_offset + column;
    const float value = to_float<T>(input[index]);
    const float scale = to_float<T>(weight[column]);
    output[index] = from_float<T>(value * inverse_rms * scale);
  }
}

}  // namespace

cudaError_t rmsnorm_v2(const void* input,
                       const void* weight,
                       void* output,
                       int64_t rows,
                       int64_t hidden,
                       float epsilon,
                       DType dtype,
                       int block_size,
                       cudaStream_t stream) {
  if (rows < 1 || hidden < 1) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid(static_cast<unsigned int>(rows));
  const dim3 block(static_cast<unsigned int>(block_size));
  const int num_warps = block_size / 32;
  const size_t shared_bytes = static_cast<size_t>(num_warps) * sizeof(float);

  const int alignment = dtype == DType::kFloat16 ? 4 : 16;
  const int vector_width = dtype == DType::kFloat16 ? 2 : 4;
  const bool aligned =
      (reinterpret_cast<uintptr_t>(input) % alignment) == 0 &&
      (reinterpret_cast<uintptr_t>(weight) % alignment) == 0 &&
      (reinterpret_cast<uintptr_t>(output) % alignment) == 0;
  if (!aligned || (hidden % vector_width) != 0) {
    return rmsnorm_scalar_safe(input, weight, output, rows, hidden, epsilon,
                               dtype, block_size, stream);
  }

  if (dtype == DType::kFloat16) {
    rmsnorm_v2_f16_kernel<<<grid, block, shared_bytes, stream>>>(
        static_cast<const __half*>(input),
        static_cast<const __half*>(weight),
        static_cast<__half*>(output),
        static_cast<int>(hidden),
        epsilon);
  } else {
    rmsnorm_v2_f32_kernel<<<grid, block, shared_bytes, stream>>>(
        static_cast<const float*>(input),
        static_cast<const float*>(weight),
        static_cast<float*>(output),
        static_cast<int>(hidden),
        epsilon);
  }
  return cudaGetLastError();
}

cudaError_t rmsnorm_scalar_safe(const void* input,
                                const void* weight,
                                void* output,
                                int64_t rows,
                                int64_t hidden,
                                float epsilon,
                                DType dtype,
                                int block_size,
                                cudaStream_t stream) {
  if (rows < 1 || hidden < 1) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid(static_cast<unsigned int>(rows));
  const dim3 block(static_cast<unsigned int>(block_size));
  const size_t shared_bytes =
      static_cast<size_t>(block_size / 32) * sizeof(float);
  if (dtype == DType::kFloat16) {
    rmsnorm_scalar_safe_kernel<<<grid, block, shared_bytes, stream>>>(
        static_cast<const __half*>(input),
        static_cast<const __half*>(weight),
        static_cast<__half*>(output), static_cast<int>(hidden), epsilon);
  } else if (dtype == DType::kFloat32) {
    rmsnorm_scalar_safe_kernel<<<grid, block, shared_bytes, stream>>>(
        static_cast<const float*>(input),
        static_cast<const float*>(weight),
        static_cast<float*>(output), static_cast<int>(hidden), epsilon);
  } else {
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

int rmsnorm_v2_occupancy(DType dtype, int block_size) {
  int max_blocks = 0;
  const size_t shared_bytes =
      static_cast<size_t>(block_size / 32) * sizeof(float);
  if (dtype == DType::kFloat16) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks, rmsnorm_v2_f16_kernel, block_size, shared_bytes);
  } else {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks, rmsnorm_v2_f32_kernel, block_size, shared_bytes);
  }
  return max_blocks;
}

int rmsnorm_scalar_safe_occupancy(DType dtype, int block_size) {
  int max_blocks = 0;
  const size_t shared_bytes =
      static_cast<size_t>(block_size / 32) * sizeof(float);
  if (dtype == DType::kFloat16) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks, rmsnorm_scalar_safe_kernel<__half>, block_size,
        shared_bytes);
  } else if (dtype == DType::kFloat32) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks, rmsnorm_scalar_safe_kernel<float>, block_size,
        shared_bytes);
  }
  return max_blocks;
}

}  // namespace hqsb
