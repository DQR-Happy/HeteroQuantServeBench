// New Qwen/HF semantics. The frozen S03/S06 exports are intentionally untouched.
#include "model_kernels.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

extern "C" int hqsb_l2_abi() { return 1; }

__device__ float sum_block(float x) {
    __shared__ float warps[4];
    for (int d = 16; d; d /= 2) x += __shfl_down_sync(0xffffffff, x, d);
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x / 32] = x;
    __syncthreads();
    x = threadIdx.x < 4 ? warps[threadIdx.x] : 0.f;
    if (threadIdx.x < 32)
        for (int d = 16; d; d /= 2) x += __shfl_down_sync(0xffffffff, x, d);
    if (threadIdx.x == 0) warps[0] = x;
    __syncthreads();
    return warps[0];
}

template<bool Fused, bool Vector>
__global__ void norm(const half* x, const half* residual, const half* weight,
                     half* y, half* updated, int h, float eps) {
    int64_t row = int64_t(blockIdx.x) * h;
    float total = 0.f;
    if constexpr (Vector) {
        for (int j = threadIdx.x * 2; j < h; j += blockDim.x * 2) {
            half2 v = *reinterpret_cast<const half2*>(x + row + j);
            float2 f = __half22float2(v);
            total += f.x * f.x + f.y * f.y;
        }
    } else {
        for (int j = threadIdx.x; j < h; j += blockDim.x) {
            half v = x[row + j];
            if constexpr (Fused) {
                v = __hadd(v, residual[row + j]);
                updated[row + j] = v;
            }
            float f = __half2float(v);
            total += f * f;
        }
    }
    float inv = rsqrtf(sum_block(total) / h + eps);
    // HF casts the normalized value BEFORE the weight multiplication.
    for (int j = threadIdx.x; j < h; j += blockDim.x) {
        half v;
        if constexpr (Fused) v = updated[row + j];
        else v = x[row + j];
        y[row + j] = __hmul(__float2half_rn(__half2float(v) * inv), weight[j]);
    }
}

extern "C" int hqsb_l2_norm(const void* x, const void* r, const void* w,
                            void* y, void* updated, int64_t rows, int h,
                            float eps, int variant, void* stream) {
    if (!x || !w || !y || rows <= 0 || rows > 2147483647 || h <= 0 ||
        !(eps > 0) || !std::isfinite(eps) || variant < 1 || variant > 3) return -1;
    auto s = static_cast<cudaStream_t>(stream);
    if (variant == 3) {
        if (!r || !updated) return -1;
        norm<true,false><<<rows,128,0,s>>>((const half*)x,(const half*)r,(const half*)w,
                                         (half*)y,(half*)updated,h,eps);
    } else if (variant == 2) {
        // The bridge must reject unsupported vector signatures, never silently change version.
        if (h % 2 || (reinterpret_cast<uintptr_t>(x) % 4)) return -2;
        norm<false,true><<<rows,128,0,s>>>((const half*)x,nullptr,(const half*)w,(half*)y,nullptr,h,eps);
    } else {
        norm<false,false><<<rows,128,0,s>>>((const half*)x,nullptr,(const half*)w,(half*)y,nullptr,h,eps);
    }
    return int(cudaGetLastError());
}

__global__ void silu_mul(const half* gate, const half* up, half* y, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __half2float(gate[i]);
        half silu = __float2half_rn(g / (1.f + expf(-g)));
        y[i] = __hmul(silu, up[i]);
    }
}
extern "C" int hqsb_l2_silu(const void* gate, const void* up, void* y, int64_t n, void* stream) {
    if (!gate || !up || !y || n <= 0 || n > int64_t(2147483647)*256) return -1;
    silu_mul<<<(n+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(
        (const half*)gate,(const half*)up,(half*)y,n);
    return int(cudaGetLastError());
}
