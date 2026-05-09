#include "hqsb/fused_residual_rmsnorm.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using hqsb::DType;
using hqsb::FusedResidualVariant;

size_t element_bytes(DType dtype) {
  return (dtype == DType::kFloat16) ? sizeof(__half) : sizeof(float);
}

std::vector<float> gen(size_t n) {
  std::vector<float> v(n);
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> d(-1.0F, 1.0F);
  for (float& x : v) x = d(rng);
  return v;
}

std::vector<__half> to_half(const std::vector<float>& v) {
  std::vector<__half> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = __float2half_rn(v[i]);
  return o;
}

const char* dtype_name(DType d) {
  return d == DType::kFloat16 ? "fp16" : "fp32";
}

void run_case(DType dtype, int64_t rows, int64_t hidden,
              FusedResidualVariant variant, int block, int warmup,
              int iterations) {
  const size_t n = static_cast<size_t>(rows) * hidden;
  const size_t in_bytes = n * element_bytes(dtype);
  const size_t w_bytes = static_cast<size_t>(hidden) * element_bytes(dtype);

  auto h_in = gen(n), h_res = gen(n), h_w = gen(hidden);
  void *d_in = nullptr, *d_res = nullptr, *d_w = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, in_bytes);
  cudaMalloc(&d_res, in_bytes);
  cudaMalloc(&d_w, w_bytes);
  cudaMalloc(&d_out, in_bytes);

  if (dtype == DType::kFloat16) {
    auto hi = to_half(h_in), hr = to_half(h_res), hw = to_half(h_w);
    cudaMemcpy(d_in, hi.data(), in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, hr.data(), in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, hw.data(), w_bytes, cudaMemcpyHostToDevice);
  } else {
    cudaMemcpy(d_in, h_in.data(), in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, h_res.data(), in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), w_bytes, cudaMemcpyHostToDevice);
  }

  auto launch = [&](cudaStream_t s) {
    return hqsb::fused_residual_rmsnorm_forward(
        d_in, d_res, d_w, d_out, rows, hidden, 1e-5F, dtype, variant, s);
  };

  for (int i = 0; i < warmup; ++i) launch(0);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < iterations; ++i) launch(0);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float device_ms = 0.0F;
  cudaEventElapsedTime(&device_ms, start, stop);
  device_ms /= static_cast<float>(iterations);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Fused traffic: read input + read residual + write temp (first pass),
  // read temp + write out + read weight (second pass) = 5n + hidden.
  const double traffic_bytes =
      static_cast<double>(5 * n + hidden) * element_bytes(dtype);
  const double bw = traffic_bytes / (device_ms / 1e3) / 1e9;

  std::printf("%s,%lld,%lld,%s,%d,%.4f,%.2f\n", dtype_name(dtype),
              (long long)rows, (long long)hidden,
              hqsb::fused_residual_variant_name(variant), block, device_ms, bw);

  cudaFree(d_in);
  cudaFree(d_res);
  cudaFree(d_w);
  cudaFree(d_out);
}

}  // namespace

int main(int argc, char** argv) {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    std::fprintf(stderr, "no CUDA device\n");
    return EXIT_FAILURE;
  }

  int64_t rows = 512, hidden = 2048;
  int block = 256, warmup = 20, iterations = 200;
  std::string dtype = "fp32";
  std::string variant = "all";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--rows") rows = std::atoll(next());
    else if (a == "--hidden") hidden = std::atoll(next());
    else if (a == "--block") block = std::atoi(next());
    else if (a == "--warmup") warmup = std::atoi(next());
    else if (a == "--iterations") iterations = std::atoi(next());
    else if (a == "--dtype") dtype = next();
    else if (a == "--variant") variant = next();
  }

  DType dt = (dtype == "fp16") ? DType::kFloat16 : DType::kFloat32;
  std::vector<FusedResidualVariant> variants;
  if (variant == "all") {
    // V0 (shared reduction) is FP32-only; FP16 runs V1 only.
    if (dt == DType::kFloat16) {
      variants = {FusedResidualVariant::kV1Vectorized};
    } else {
      variants = {FusedResidualVariant::kV0Shared,
                  FusedResidualVariant::kV1Vectorized};
    }
  } else if (variant == "v0") {
    variants = {FusedResidualVariant::kV0Shared};
  } else if (variant == "v1") {
    variants = {FusedResidualVariant::kV1Vectorized};
  } else {
    std::fprintf(stderr, "unknown --variant %s\n", variant.c_str());
    return EXIT_FAILURE;
  }

  std::printf("dtype,rows,hidden,variant,block_size,device_ms,device_bw_gbps\n");
  for (auto v : variants) {
    run_case(dt, rows, hidden, v, block, warmup, iterations);
  }
  return EXIT_SUCCESS;
}
