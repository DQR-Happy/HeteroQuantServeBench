#include "hqsb/rmsnorm.h"
#include "rmsnorm_launchers.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

  // float4 requires 16-byte aligned rows; when hidden is not a multiple of
  // 4, successive rows are misaligned, so we fall back to a fully scalar
  // path (vec_count == 0).
  const int vec_count = (hidden % 4 == 0) ? (hidden >> 2) : 0;
  const int tail_start = vec_count << 2;

  float local_square_sum = 0.0F;
  const float4* row_in4 = reinterpret_cast<const float4*>(row_in);

  for (int i = thread; i < vec_count; i += blockDim.x) {
    const float4 v = row_in4[i];
    local_square_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  for (int c = tail_start + thread; c < hidden; c += blockDim.x) {
    const float v = row_in[c];
    local_square_sum += v * v;
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
  for (int c = tail_start + thread; c < hidden; c += blockDim.x) {
    row_out[c] = row_in[c] * inverse_rms * weight[c];
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

  // half2 requires 4-byte aligned rows; odd column counts misalign the next
  // row (each row is hidden*2 bytes), so fall back to a scalar path.
  const int vec_count = (hidden % 2 == 0) ? (hidden >> 1) : 0;
  const int tail_start = vec_count << 1;

  float local_square_sum = 0.0F;
  const __half2* row_in2 = reinterpret_cast<const __half2*>(row_in);

  for (int i = thread; i < vec_count; i += blockDim.x) {
    const float2 v = __half22float2(row_in2[i]);
    local_square_sum += v.x * v.x + v.y * v.y;
  }
  for (int c = tail_start + thread; c < hidden; c += blockDim.x) {
    const float v = __half2float(row_in[c]);
    local_square_sum += v * v;
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
  for (int c = tail_start + thread; c < hidden; c += blockDim.x) {
    const float v = __half2float(row_in[c]);
    const float w = __half2float(weight[c]);
    row_out[c] = __float2half_rn(v * inverse_rms * w);
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

}  // namespace hqsb
