#include "hqsb/fused_residual_rmsnorm.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace hqsb {

namespace {

// ── CPU reference (FP64) ──────────────────────────────────────────────

void reference_fp64(const float* input,
                    const float* residual,
                    const float* weight,
                    float* output,
                    int64_t rows,
                    int64_t hidden,
                    float epsilon) {
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t offset = row * hidden;
    double square_sum = 0.0;
    for (int64_t col = 0; col < hidden; ++col) {
      const double x = static_cast<double>(input[offset + col]) +
                       static_cast<double>(residual[offset + col]);
      square_sum += x * x;
    }
    const double inverse_rms =
        1.0 / std::sqrt(square_sum / static_cast<double>(hidden) +
                        static_cast<double>(epsilon));
    for (int64_t col = 0; col < hidden; ++col) {
      const double x = static_cast<double>(input[offset + col]) +
                       static_cast<double>(residual[offset + col]);
      output[offset + col] = static_cast<float>(
          x * inverse_rms * static_cast<double>(weight[col]));
    }
  }
}

// ── V0: fused + shared-memory reduction ───────────────────────────────

__global__ void fused_residual_v0_kernel(const float* input,
                                         const float* residual,
                                         const float* weight,
                                         float* output,
                                         int hidden,
                                         float epsilon) {
  extern __shared__ float shared_square_sum[];

  const int row = static_cast<int>(blockIdx.x);
  const int thread = static_cast<int>(threadIdx.x);
  const size_t row_offset = static_cast<size_t>(row) * hidden;

  float local_square_sum = 0.0F;
  for (int column = thread; column < hidden; column += blockDim.x) {
    const float x = input[row_offset + column] + residual[row_offset + column];
    output[row_offset + column] = x;  // reuse output as the add intermediate
    local_square_sum += x * x;
  }

  shared_square_sum[thread] = local_square_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread < static_cast<int>(stride)) {
      shared_square_sum[thread] +=
          shared_square_sum[thread + static_cast<int>(stride)];
    }
    __syncthreads();
  }

  const float inverse_rms =
      rsqrtf(shared_square_sum[0] / static_cast<float>(hidden) + epsilon);

  for (int column = thread; column < hidden; column += blockDim.x) {
    output[row_offset + column] =
        output[row_offset + column] * inverse_rms * weight[column];
  }
}

// ── V1: fused + warp shuffle + vectorized ─────────────────────────────

// FP32 float4 variant.
__global__ void fused_residual_v1_f32_kernel(const float* __restrict__ input,
                                             const float* __restrict__ residual,
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
  const float* row_res = residual + row_offset;
  float* row_out = output + row_offset;

  const int vec_count = (hidden % 4 == 0) ? (hidden >> 2) : 0;
  const int tail_start = vec_count << 2;

  float local = 0.0F;
  const float4* in4 = reinterpret_cast<const float4*>(row_in);
  const float4* res4 = reinterpret_cast<const float4*>(row_res);
  float4* out4 = reinterpret_cast<float4*>(row_out);

  for (int i = thread; i < vec_count; i += blockDim.x) {
    const float4 a = in4[i];
    const float4 b = res4[i];
    float4 x;
    x.x = a.x + b.x;
    x.y = a.y + b.y;
    x.z = a.z + b.z;
    x.w = a.w + b.w;
    out4[i] = x;
    local += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
  }
  for (int c = tail_start + thread; c < hidden; c += blockDim.x) {
    const float x = row_in[c] + row_res[c];
    row_out[c] = x;
    local += x * x;
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local += __shfl_down_sync(0xFFFFFFFFu, local, offset);
  }
  if (lane == 0) {
    warp_square_sum[warp] = local;
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

  const float4* w4 = reinterpret_cast<const float4*>(weight);
  for (int i = thread; i < vec_count; i += blockDim.x) {
    const float4 x = out4[i];
    const float4 w = w4[i];
    float4 o;
    o.x = x.x * inverse_rms * w.x;
    o.y = x.y * inverse_rms * w.y;
    o.z = x.z * inverse_rms * w.z;
    o.w = x.w * inverse_rms * w.w;
    out4[i] = o;
  }
  for (int c = tail_start + thread; c < hidden; c += blockDim.x) {
    row_out[c] = row_out[c] * inverse_rms * weight[c];
  }
}

// FP16 half2 variant.
__global__ void fused_residual_v1_f16_kernel(const __half* __restrict__ input,
                                             const __half* __restrict__ residual,
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
  const __half* row_res = residual + row_offset;
  __half* row_out = output + row_offset;

  const int vec_count = (hidden % 2 == 0) ? (hidden >> 1) : 0;
  const int tail_start = vec_count << 1;

  float local = 0.0F;
  const __half2* in2 = reinterpret_cast<const __half2*>(row_in);
  const __half2* res2 = reinterpret_cast<const __half2*>(row_res);
  __half2* out2 = reinterpret_cast<__half2*>(row_out);

  for (int i = thread; i < vec_count; i += blockDim.x) {
    const float2 a = __half22float2(in2[i]);
    const float2 b = __half22float2(res2[i]);
    float2 x;
    x.x = a.x + b.x;
    x.y = a.y + b.y;
    out2[i] = __float22half2_rn(x);
    local += x.x * x.x + x.y * x.y;
  }
  for (int c = tail_start + thread; c < hidden; c += blockDim.x) {
    const float x = __half2float(row_in[c]) + __half2float(row_res[c]);
    row_out[c] = __float2half_rn(x);
    local += x * x;
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local += __shfl_down_sync(0xFFFFFFFFu, local, offset);
  }
  if (lane == 0) {
    warp_square_sum[warp] = local;
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

  const __half2* w2 = reinterpret_cast<const __half2*>(weight);
  for (int i = thread; i < vec_count; i += blockDim.x) {
    const float2 x = __half22float2(out2[i]);
    const float2 w = __half22float2(w2[i]);
    float2 o;
    o.x = x.x * inverse_rms * w.x;
    o.y = x.y * inverse_rms * w.y;
    out2[i] = __float22half2_rn(o);
  }
  for (int c = tail_start + thread; c < hidden; c += blockDim.x) {
    const float x = __half2float(row_out[c]);
    const float w = __half2float(weight[c]);
    row_out[c] = __float2half_rn(x * inverse_rms * w);
  }
}

}  // namespace

void fused_residual_reference_cpu(const float* input,
                                  const float* residual,
                                  const float* weight,
                                  float* output,
                                  int64_t rows,
                                  int64_t hidden,
                                  float epsilon) {
  reference_fp64(input, residual, weight, output, rows, hidden, epsilon);
}

const char* fused_residual_variant_name(FusedResidualVariant variant) {
  switch (variant) {
    case FusedResidualVariant::kAuto:
      return "auto";
    case FusedResidualVariant::kReference:
      return "reference";
    case FusedResidualVariant::kV0Shared:
      return "v0_shared";
    case FusedResidualVariant::kV1Vectorized:
      return "v1_vectorized";
  }
  return "unknown";
}

FusedResidualVariant fused_residual_select_variant(int64_t hidden,
                                                   DType dtype) {
  // kV1Vectorized is the only FP16-capable variant and its half2 kernel
  // handles odd column counts via a scalar tail, so FP16 always routes to
  // V1. kV0Shared (scalar) remains the FP32 fallback for non-multiple-of-4
  // column counts.
  if (dtype == DType::kFloat16) {
    return FusedResidualVariant::kV1Vectorized;
  }
  return (hidden % 4 == 0) ? FusedResidualVariant::kV1Vectorized
                           : FusedResidualVariant::kV0Shared;
}

cudaError_t fused_residual_rmsnorm_forward(const void* input,
                                           const void* residual,
                                           const void* weight,
                                           void* output,
                                           int64_t rows,
                                           int64_t hidden,
                                           float epsilon,
                                           DType dtype,
                                           FusedResidualVariant variant,
                                           cudaStream_t stream) {
  if (input == nullptr || residual == nullptr || weight == nullptr ||
      output == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (rows < 1 || hidden < 1) {
    return cudaErrorInvalidValue;
  }

  FusedResidualVariant chosen = variant;
  if (chosen == FusedResidualVariant::kAuto) {
    chosen = fused_residual_select_variant(hidden, dtype);
  }
  if (chosen == FusedResidualVariant::kReference) {
    return cudaErrorInvalidValue;
  }

  const int block_size = 256;
  const dim3 grid(static_cast<unsigned int>(rows));
  const dim3 block(static_cast<unsigned int>(block_size));

  switch (chosen) {
    case FusedResidualVariant::kV0Shared: {
      if (dtype != DType::kFloat32) {
        return cudaErrorInvalidValue;
      }
      const size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(float);
      fused_residual_v0_kernel<<<grid, block, shared_bytes, stream>>>(
          static_cast<const float*>(input),
          static_cast<const float*>(residual),
          static_cast<const float*>(weight),
          static_cast<float*>(output),
          static_cast<int>(hidden), epsilon);
      return cudaGetLastError();
    }
    case FusedResidualVariant::kV1Vectorized: {
      const size_t shared_bytes =
          static_cast<size_t>(block_size / 32) * sizeof(float);
      if (dtype == DType::kFloat16) {
        fused_residual_v1_f16_kernel<<<grid, block, shared_bytes, stream>>>(
            static_cast<const __half*>(input),
            static_cast<const __half*>(residual),
            static_cast<const __half*>(weight),
            static_cast<__half*>(output),
            static_cast<int>(hidden), epsilon);
      } else {
        fused_residual_v1_f32_kernel<<<grid, block, shared_bytes, stream>>>(
            static_cast<const float*>(input),
            static_cast<const float*>(residual),
            static_cast<const float*>(weight),
            static_cast<float*>(output),
            static_cast<int>(hidden), epsilon);
      }
      return cudaGetLastError();
    }
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace hqsb
