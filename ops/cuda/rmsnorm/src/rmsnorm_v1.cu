#include "hqsb/rmsnorm.h"
#include "rmsnorm_launchers.h"

#include <cuda_runtime.h>

namespace hqsb {

namespace {

// V1 — warp-shuffle reduction. Hypothesis: the V0 shared-memory tree
// reduction performs ``log2(blockDim)`` __syncthreads and shared accesses;
// most of that traffic is within a warp, which can be collapsed using
// ``__shfl_down_sync``. V1 reduces shared-memory traffic to one value per
// warp (instead of one per thread) and removes the intra-warp barriers.
//
// Layout: one block per row. Partial sums are reduced within each warp via
// shuffle; each warp writes a single value to shared memory; the first warp
// reduces across warps.
__global__ void rmsnorm_v1_kernel(const float* input,
                                  const float* weight,
                                  float* output,
                                  int hidden,
                                  float epsilon) {
  extern __shared__ float warp_square_sum[];  // one slot per warp

  const int row = static_cast<int>(blockIdx.x);
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int num_warps = blockDim.x >> 5;
  const size_t row_offset = static_cast<size_t>(row) * hidden;

  float local_square_sum = 0.0F;
  for (int column = thread; column < hidden; column += blockDim.x) {
    const float value = input[row_offset + column];
    local_square_sum += value * value;
  }

  // Intra-warp reduction (no shared memory, no barrier).
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_square_sum +=
        __shfl_down_sync(0xFFFFFFFFu, local_square_sum, offset);
  }

  if (lane == 0) {
    warp_square_sum[warp] = local_square_sum;
  }
  __syncthreads();

  // Inter-warp reduction by the first warp.
  float total_square_sum = 0.0F;
  if (warp == 0) {
    total_square_sum = (lane < num_warps) ? warp_square_sum[lane] : 0.0F;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      total_square_sum +=
          __shfl_down_sync(0xFFFFFFFFu, total_square_sum, offset);
    }
  }

  // Broadcast the total to all threads via shared memory.
  if (warp == 0 && lane == 0) {
    warp_square_sum[0] = total_square_sum;
  }
  __syncthreads();

  const float inverse_rms =
      rsqrtf(warp_square_sum[0] / static_cast<float>(hidden) + epsilon);

  for (int column = thread; column < hidden; column += blockDim.x) {
    output[row_offset + column] =
        input[row_offset + column] * inverse_rms * weight[column];
  }
}

}  // namespace

cudaError_t rmsnorm_v1(const float* input,
                       const float* weight,
                       float* output,
                       int64_t rows,
                       int64_t hidden,
                       float epsilon,
                       int block_size,
                       cudaStream_t stream) {
  if (rows < 1 || hidden < 1) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid(static_cast<unsigned int>(rows));
  const dim3 block(static_cast<unsigned int>(block_size));
  const int num_warps = block_size / 32;
  const size_t shared_bytes = static_cast<size_t>(num_warps) * sizeof(float);

  rmsnorm_v1_kernel<<<grid, block, shared_bytes, stream>>>(
      input, weight, output, static_cast<int>(hidden), epsilon);
  return cudaGetLastError();
}

int rmsnorm_v1_occupancy(int block_size) {
  int max_blocks = 0;
  const size_t shared_bytes =
      static_cast<size_t>(block_size / 32) * sizeof(float);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks, rmsnorm_v1_kernel, block_size, shared_bytes);
  return max_blocks;
}

}  // namespace hqsb
