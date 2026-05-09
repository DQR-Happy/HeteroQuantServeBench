#include "hqsb/rmsnorm.h"
#include "rmsnorm_launchers.h"

#include <cuda_runtime.h>

namespace hqsb {

namespace {

// V0 — shared-memory tree reduction (S00 baseline, preserved verbatim in
// behavior). Hypothesis: a simple shared-memory reduction is correctness-
// first and easy to reason about; it is the tracked baseline all later
// variants are compared against.
//
// Layout: one block per row. Each thread accumulates a strided partial sum
// of squares, writes it to shared memory, then the block performs a classic
// tree reduction. The block size bounds shared-memory usage to
// ``blockDim * sizeof(float)``.
__global__ void rmsnorm_v0_kernel(const float* input,
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
    const float value = input[row_offset + column];
    local_square_sum += value * value;
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
        input[row_offset + column] * inverse_rms * weight[column];
  }
}

}  // namespace

// Launcher for V0. ``block_size`` must be a power of two in [32, 1024].
cudaError_t rmsnorm_v0(const float* input,
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
  const size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(float);

  rmsnorm_v0_kernel<<<grid, block, shared_bytes, stream>>>(
      input, weight, output, static_cast<int>(hidden), epsilon);
  return cudaGetLastError();
}

int rmsnorm_v0_occupancy(int block_size) {
  int max_blocks = 0;
  const size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(float);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks, rmsnorm_v0_kernel, block_size, shared_bytes);
  return max_blocks;
}

}  // namespace hqsb
