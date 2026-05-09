#include "hqsb/fused_residual_rmsnorm.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <random>
#include <vector>

#include "test_metrics.h"
#include "test_util.h"

namespace {

using hqsb::DType;
using hqsb::FusedResidualVariant;

std::vector<__half> to_half(const std::vector<float>& v) {
  std::vector<__half> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = __float2half_rn(v[i]);
  return out;
}

std::vector<float> from_half(const std::vector<__half>& v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = __half2float(v[i]);
  return out;
}

size_t element_bytes(DType dtype) {
  return (dtype == DType::kFloat16) ? sizeof(__half) : sizeof(float);
}

std::vector<float> generate(size_t n, int mode) {
  std::vector<float> v(n);
  std::mt19937 rng(99);
  switch (mode) {
    case 1: {
      std::fill(v.begin(), v.end(), 0.0F);
      break;
    }
    case 2: {
      std::uniform_real_distribution<float> d(-1.0e4F, 1.0e4F);
      for (float& x : v) x = d(rng);
      break;
    }
    default: {
      std::uniform_real_distribution<float> d(-1.0F, 1.0F);
      for (float& x : v) x = d(rng);
      break;
    }
  }
  return v;
}

std::vector<float> generate_weight(size_t n) {
  std::vector<float> v(n);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> d(0.5F, 1.5F);
  for (float& x : v) x = d(rng);
  return v;
}

hqsb::test::Metrics run_and_compare(DType dtype,
                                    const std::vector<float>& input,
                                    const std::vector<float>& residual,
                                    const std::vector<float>& weight,
                                    int64_t rows,
                                    int64_t hidden,
                                    FusedResidualVariant variant) {
  const size_t n = static_cast<size_t>(rows) * hidden;
  std::vector<float> reference(n);
  hqsb::fused_residual_reference_cpu(input.data(), residual.data(),
                                     weight.data(), reference.data(),
                                     rows, hidden, 1e-5F);

  void *d_in = nullptr, *d_res = nullptr, *d_w = nullptr, *d_out = nullptr;
  const size_t in_bytes = n * element_bytes(dtype);
  const size_t w_bytes = static_cast<size_t>(hidden) * element_bytes(dtype);
  cudaMalloc(&d_in, in_bytes);
  cudaMalloc(&d_res, in_bytes);
  cudaMalloc(&d_w, w_bytes);
  cudaMalloc(&d_out, in_bytes);

  if (dtype == DType::kFloat16) {
    auto hi = to_half(input), hr = to_half(residual), hw = to_half(weight);
    cudaMemcpy(d_in, hi.data(), in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, hr.data(), in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, hw.data(), w_bytes, cudaMemcpyHostToDevice);
  } else {
    cudaMemcpy(d_in, input.data(), in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, residual.data(), in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, weight.data(), w_bytes, cudaMemcpyHostToDevice);
  }

  cudaError_t err = hqsb::fused_residual_rmsnorm_forward(
      d_in, d_res, d_w, d_out, rows, hidden, 1e-5F, dtype, variant, 0);
  if (err != cudaSuccess) {
    cudaFree(d_in); cudaFree(d_res); cudaFree(d_w); cudaFree(d_out);
    hqsb::test::Metrics m;
    m.max_abs_error = 1e9;
    return m;
  }
  cudaDeviceSynchronize();

  std::vector<float> actual(n);
  if (dtype == DType::kFloat16) {
    std::vector<__half> ho(n);
    cudaMemcpy(ho.data(), d_out, in_bytes, cudaMemcpyDeviceToHost);
    actual = from_half(ho);
  } else {
    cudaMemcpy(actual.data(), d_out, in_bytes, cudaMemcpyDeviceToHost);
  }

  cudaFree(d_in); cudaFree(d_res); cudaFree(d_w); cudaFree(d_out);
  return hqsb::test::compute_metrics(actual.data(), reference.data(), n);
}

void test_fp32_baseline() {
  auto input = generate(512 * 1024, 0);
  auto residual = generate(512 * 1024, 0);
  auto weight = generate_weight(1024);
  for (auto variant :
       {FusedResidualVariant::kV0Shared, FusedResidualVariant::kV1Vectorized}) {
    auto m = run_and_compare(DType::kFloat32, input, residual, weight, 512,
                             1024, variant);
    CHECK_NEAR(m.max_abs_error, 0.0, 5e-4);
  }
}

void test_fp32_non_power_of_two() {
  for (int64_t hidden : {100, 500}) {
    auto input = generate(32 * hidden, 0);
    auto residual = generate(32 * hidden, 0);
    auto weight = generate_weight(hidden);
    for (auto variant :
         {FusedResidualVariant::kV0Shared, FusedResidualVariant::kV1Vectorized}) {
      auto m = run_and_compare(DType::kFloat32, input, residual, weight, 32,
                               hidden, variant);
      CHECK_NEAR(m.max_abs_error, 0.0, 5e-4);
    }
  }
}

void test_fp16() {
  auto input = generate(32 * 2048, 0);
  auto residual = generate(32 * 2048, 0);
  auto weight = generate_weight(2048);
  auto m = run_and_compare(DType::kFloat16, input, residual, weight, 32, 2048,
                           FusedResidualVariant::kV1Vectorized);
  CHECK_NEAR(m.max_abs_error, 0.0, 2e-2);
}

void test_extreme() {
  auto input = generate(64 * 256, 2);
  auto residual = generate(64 * 256, 1);  // zero residual
  auto weight = generate_weight(256);
  auto m = run_and_compare(DType::kFloat32, input, residual, weight, 64, 256,
                           FusedResidualVariant::kV1Vectorized);
  CHECK_NEAR(m.max_abs_error, 0.0, 1e-3);
}

void test_dispatcher() {
  CHECK(hqsb::fused_residual_select_variant(2048, DType::kFloat32) ==
        FusedResidualVariant::kV1Vectorized);
  // 101 % 4 == 1 -> float4 vectorization impossible -> V0 scalar fallback.
  CHECK(hqsb::fused_residual_select_variant(101, DType::kFloat32) ==
        FusedResidualVariant::kV0Shared);
  CHECK(hqsb::fused_residual_select_variant(2048, DType::kFloat16) ==
        FusedResidualVariant::kV1Vectorized);
  // FP16 always routes to V1 (half2 kernel handles odd tails).
  CHECK(hqsb::fused_residual_select_variant(3, DType::kFloat16) ==
        FusedResidualVariant::kV1Vectorized);
}

void test_invalid_arguments() {
  float a = 1.0F, r = 0.0F, w = 1.0F, o = 0.0F;
  CHECK(hqsb::fused_residual_rmsnorm_forward(
            nullptr, &r, &w, &o, 1, 1, 1e-5F, DType::kFloat32,
            FusedResidualVariant::kAuto, 0) == cudaErrorInvalidValue);
  CHECK(hqsb::fused_residual_rmsnorm_forward(
            &a, &r, &w, &o, 0, 1, 1e-5F, DType::kFloat32,
            FusedResidualVariant::kAuto, 0) == cudaErrorInvalidValue);
  CHECK(hqsb::fused_residual_rmsnorm_forward(
            &a, &r, &w, &o, 1, 2, 1e-5F, DType::kFloat16,
            FusedResidualVariant::kV0Shared, 0) == cudaErrorInvalidValue);
}

}  // namespace

int main() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return EXIT_FAILURE;
  }
  test_fp32_baseline();
  test_fp32_non_power_of_two();
  test_fp16();
  test_extreme();
  test_dispatcher();
  test_invalid_arguments();
  return hqsb::test::finish("test_fused_residual_rmsnorm");
}
