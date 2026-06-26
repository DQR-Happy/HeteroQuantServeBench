#include "hqsb/fused_residual_rmsnorm.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <random>
#include <type_traits>
#include <vector>

#include "test_util.h"

namespace {
using hqsb::DType;
using hqsb::FusedResidualSemantic;
using hqsb::FusedResidualVariant;

template <typename T> T cast(float value);
template <> float cast<float>(float value) { return value; }
template <> __half cast<__half>(float value) { return __float2half_rn(value); }
template <typename T> float as_float(T value);
template <> float as_float<float>(float value) { return value; }
template <> float as_float<__half>(__half value) { return __half2float(value); }

template <typename T>
void run_case(int rows, int hidden, FusedResidualVariant variant, float atol) {
  const size_t n = static_cast<size_t>(rows) * hidden;
  std::mt19937 rng(20260918 + hidden);
  std::uniform_real_distribution<float> input_dist(-0.75F, 0.75F);
  std::uniform_real_distribution<float> weight_dist(0.5F, 1.5F);
  std::vector<T> input(n), residual(n), weight(hidden);
  for (size_t i = 0; i < n; ++i) {
    input[i] = cast<T>(input_dist(rng));
    residual[i] = cast<T>(input_dist(rng));
  }
  for (int i = 0; i < hidden; ++i) weight[i] = cast<T>(weight_dist(rng));

  std::vector<T> ref_residual(n), ref_y(n);
  for (int row = 0; row < rows; ++row) {
    double sum = 0.0;
    for (int col = 0; col < hidden; ++col) {
      const size_t i = static_cast<size_t>(row) * hidden + col;
      ref_residual[i] = cast<T>(as_float(input[i]) + as_float(residual[i]));
      const double value = as_float(ref_residual[i]);
      sum += value * value;
    }
    const double inv = 1.0 / std::sqrt(sum / hidden + 1e-6);
    for (int col = 0; col < hidden; ++col) {
      const size_t i = static_cast<size_t>(row) * hidden + col;
      ref_y[i] = cast<T>(as_float(ref_residual[i]) * as_float(weight[col]) * inv);
    }
  }

  T *d_input = nullptr, *d_residual = nullptr, *d_weight = nullptr;
  T *d_residual_out = nullptr, *d_y = nullptr;
  cudaMalloc(&d_input, n * sizeof(T));
  cudaMalloc(&d_residual, n * sizeof(T));
  cudaMalloc(&d_weight, hidden * sizeof(T));
  cudaMalloc(&d_residual_out, n * sizeof(T));
  cudaMalloc(&d_y, n * sizeof(T));
  cudaMemcpy(d_input, input.data(), n * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_residual, residual.data(), n * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, weight.data(), hidden * sizeof(T), cudaMemcpyHostToDevice);
  const DType dtype = std::is_same<T, __half>::value ? DType::kFloat16
                                                     : DType::kFloat32;
  CHECK(hqsb::fused_residual_rmsnorm_forward(
            d_input, d_residual, d_weight, d_residual_out, d_y, rows, hidden,
            1e-6F, dtype, variant,
            FusedResidualSemantic::kStrictRoundedResidual, nullptr) ==
        cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  std::vector<T> got_residual(n), got_y(n);
  cudaMemcpy(got_residual.data(), d_residual_out, n * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(got_y.data(), d_y, n * sizeof(T), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < n; ++i) {
    CHECK_NEAR(as_float(got_residual[i]), as_float(ref_residual[i]), 0.0);
    CHECK_NEAR(as_float(got_y[i]), as_float(ref_y[i]), atol);
  }
  cudaFree(d_input); cudaFree(d_residual); cudaFree(d_weight);
  cudaFree(d_residual_out); cudaFree(d_y);
}

void invalid_alias_is_rejected() {
  float *a = nullptr, *r = nullptr, *w = nullptr, *y = nullptr;
  cudaMalloc(&a, 128 * sizeof(float));
  cudaMalloc(&r, 128 * sizeof(float));
  cudaMalloc(&w, 128 * sizeof(float));
  cudaMalloc(&y, 128 * sizeof(float));
  CHECK(hqsb::fused_residual_rmsnorm_forward(
            a, r, w, a, y, 1, 128, 1e-6F, DType::kFloat32,
            FusedResidualVariant::kV1WarpShuffle,
            FusedResidualSemantic::kStrictRoundedResidual, nullptr) ==
        cudaErrorInvalidValue);
  cudaFree(a); cudaFree(r); cudaFree(w); cudaFree(y);
}
}  // namespace

int main() {
  run_case<float>(1, 128, FusedResidualVariant::kV0SharedTree, 2e-5F);
  run_case<float>(17, 101, FusedResidualVariant::kV1WarpShuffle, 2e-5F);
  run_case<float>(2, 8192, FusedResidualVariant::kV1WarpShuffle, 3e-5F);
  run_case<__half>(1, 2048, FusedResidualVariant::kV1WarpShuffle, 4e-3F);
  run_case<__half>(8, 101, FusedResidualVariant::kV1WarpShuffle, 4e-3F);
  invalid_alias_is_rejected();
  return hqsb::test::finish("test_fused_residual_rmsnorm");
}
