#include "hqsb/fused_residual_rmsnorm.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
  int rows = 1, hidden = 2048, iterations = 100;
  std::string mode = "fused";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--rows" && i + 1 < argc) rows = std::atoi(argv[++i]);
    else if (arg == "--hidden" && i + 1 < argc) hidden = std::atoi(argv[++i]);
    else if (arg == "--iterations" && i + 1 < argc) iterations = std::atoi(argv[++i]);
    else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
  }
  const size_t n = static_cast<size_t>(rows) * hidden;
  float *x = nullptr, *r = nullptr, *w = nullptr, *ro = nullptr, *y = nullptr;
  cudaMalloc(&x, n * sizeof(float)); cudaMalloc(&r, n * sizeof(float));
  cudaMalloc(&w, hidden * sizeof(float)); cudaMalloc(&ro, n * sizeof(float));
  cudaMalloc(&y, n * sizeof(float));
  cudaMemset(x, 0, n * sizeof(float)); cudaMemset(r, 0, n * sizeof(float));
  cudaMemset(w, 0, hidden * sizeof(float));
  cudaStream_t stream; cudaStreamCreate(&stream);
  auto launch = [&]() {
    if (mode == "separate") {
      return hqsb::separate_residual_rmsnorm_forward(
          x, r, w, ro, y, rows, hidden, 1e-6F, hqsb::DType::kFloat32,
          hqsb::FusedResidualSemantic::kStrictRoundedResidual, stream);
    }
    return hqsb::fused_residual_rmsnorm_forward(
        x, r, w, ro, y, rows, hidden, 1e-6F, hqsb::DType::kFloat32,
        hqsb::FusedResidualVariant::kV1WarpShuffle,
        hqsb::FusedResidualSemantic::kStrictRoundedResidual, stream);
  };
  for (int i = 0; i < 10; ++i) launch();
  cudaStreamSynchronize(stream);
  cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) launch();
  cudaEventRecord(stop, stream); cudaEventSynchronize(stop);
  float ms = 0; cudaEventElapsedTime(&ms, start, stop);
  std::printf("mode,rows,hidden,latency_ms\n%s,%d,%d,%.9f\n",
              mode.c_str(), rows, hidden, ms / iterations);
  cudaEventDestroy(start); cudaEventDestroy(stop); cudaStreamDestroy(stream);
  cudaFree(x); cudaFree(r); cudaFree(w); cudaFree(ro); cudaFree(y);
  return 0;
}
