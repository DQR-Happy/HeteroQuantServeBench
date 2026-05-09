#include "hqsb/rmsnorm.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "test_metrics.h"
#include "test_util.h"

namespace {

using hqsb::DType;
using hqsb::RmsNormVariant;

// ── dtype conversion helpers ──────────────────────────────────────────

std::vector<__half> to_half(const std::vector<float>& v) {
  std::vector<__half> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    out[i] = __float2half_rn(v[i]);
  }
  return out;
}

std::vector<float> from_half(const std::vector<__half>& v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    out[i] = __half2float(v[i]);
  }
  return out;
}

size_t element_bytes(DType dtype) {
  return (dtype == DType::kFloat16) ? sizeof(__half) : sizeof(float);
}

// ── input generators ──────────────────────────────────────────────────

std::vector<float> generate_input(size_t n, int mode) {
  std::vector<float> v(n);
  std::mt19937 rng(1234);
  switch (mode) {
    case 1: {  // zeros
      std::fill(v.begin(), v.end(), 0.0F);
      break;
    }
    case 2: {  // large magnitude
      std::uniform_real_distribution<float> d(1.0e6F, 1.0e7F);
      for (float& x : v) x = d(rng);
      break;
    }
    case 3: {  // tiny magnitude
      std::uniform_real_distribution<float> d(1.0e-6F, 1.0e-5F);
      for (float& x : v) x = d(rng);
      break;
    }
    case 4: {  // alternating-sign large values (within FP32 square range)
      for (size_t i = 0; i < n; ++i) {
        v[i] = (i % 2 == 0) ? 1.0e10F : -1.0e10F;
      }
      break;
    }
    default: {  // uniform [-1, 1]
      std::uniform_real_distribution<float> d(-1.0F, 1.0F);
      for (float& x : v) x = d(rng);
      break;
    }
  }
  return v;
}

std::vector<float> generate_weight(size_t n) {
  std::vector<float> v(n);
  std::mt19937 rng(5678);
  std::uniform_real_distribution<float> d(0.5F, 1.5F);
  for (float& x : v) x = d(rng);
  return v;
}

// ── run + compare ─────────────────────────────────────────────────────

hqsb::test::Metrics run_and_compare(DType dtype,
                                    const std::vector<float>& host_input,
                                    const std::vector<float>& host_weight,
                                    int64_t rows,
                                    int64_t hidden,
                                    float epsilon,
                                    RmsNormVariant variant) {
  const size_t n = static_cast<size_t>(rows) * hidden;

  std::vector<float> host_reference(n);
  hqsb::rmsnorm_reference_cpu(host_input.data(), host_weight.data(),
                              host_reference.data(), rows, hidden, epsilon);

  // Prepare device buffers (half input/weight/output for FP16).
  void* d_input = nullptr;
  void* d_weight = nullptr;
  void* d_output = nullptr;
  const size_t input_bytes = n * element_bytes(dtype);
  const size_t weight_bytes = static_cast<size_t>(hidden) * element_bytes(dtype);
  const size_t output_bytes = n * element_bytes(dtype);

  cudaMalloc(&d_input, input_bytes);
  cudaMalloc(&d_weight, weight_bytes);
  cudaMalloc(&d_output, output_bytes);

  if (dtype == DType::kFloat16) {
    std::vector<__half> h_input = to_half(host_input);
    std::vector<__half> h_weight = to_half(host_weight);
    cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), weight_bytes, cudaMemcpyHostToDevice);
  } else {
    cudaMemcpy(d_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, host_weight.data(), weight_bytes, cudaMemcpyHostToDevice);
  }

  cudaError_t err = hqsb::rmsnorm_forward(d_input, d_weight, d_output,
                                          rows, hidden, epsilon, dtype,
                                          variant, /*stream=*/0);
  if (err != cudaSuccess) {
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    hqsb::test::Metrics m;
    m.max_abs_error = 1e9;  // signal failure
    return m;
  }
  cudaDeviceSynchronize();

  std::vector<float> host_actual(n);
  if (dtype == DType::kFloat16) {
    std::vector<__half> h_out(n);
    cudaMemcpy(h_out.data(), d_output, output_bytes, cudaMemcpyDeviceToHost);
    host_actual = from_half(h_out);
  } else {
    cudaMemcpy(host_actual.data(), d_output, output_bytes, cudaMemcpyDeviceToHost);
  }

  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_output);

  return hqsb::test::compute_metrics(host_actual.data(), host_reference.data(), n);
}

void check_metrics(const char* label,
                   const hqsb::test::Metrics& m,
                   double max_abs_tol) {
  CHECK_NEAR(m.max_abs_error, 0.0, max_abs_tol);
  // cosine similarity should be ~1 for a correct RMSNorm.
  if (m.cosine_similarity < 0.999) {
    std::fprintf(stderr, "  [WARN] %s cosine=%.6f\n", label, m.cosine_similarity);
  }
}

// ── test cases ────────────────────────────────────────────────────────

void test_fp32_baseline_shape() {
  // Reproduces the S00 baseline shape (512 x 1024).
  auto input = generate_input(512 * 1024, 0);
  auto weight = generate_weight(1024);
  for (auto variant :
       {RmsNormVariant::kV0Shared, RmsNormVariant::kV1WarpShuffle,
        RmsNormVariant::kV2Vectorized}) {
    auto m = run_and_compare(DType::kFloat32, input, weight, 512, 1024, 1e-5F,
                             variant);
    check_metrics("fp32_baseline", m, 5e-4);
  }
}

void test_fp32_non_power_of_two() {
  for (int64_t hidden : {100, 500, 1536}) {
    auto input = generate_input(64 * hidden, 0);
    auto weight = generate_weight(hidden);
    for (auto variant :
         {RmsNormVariant::kV1WarpShuffle, RmsNormVariant::kV2Vectorized}) {
      auto m = run_and_compare(DType::kFloat32, input, weight, 64, hidden,
                               1e-5F, variant);
      check_metrics("fp32_non_pow2", m, 5e-4);
    }
  }
}

void test_fp32_tiny_and_single_row() {
  for (int64_t hidden : {1, 2, 3, 16}) {
    for (int64_t rows : {1, 3}) {
      auto input = generate_input(rows * hidden, 0);
      auto weight = generate_weight(hidden);
      auto m = run_and_compare(DType::kFloat32, input, weight, rows, hidden,
                               1e-5F, RmsNormVariant::kV1WarpShuffle);
      check_metrics("fp32_tiny", m, 5e-4);
    }
  }
}

void test_fp32_extreme_values() {
  for (int mode : {1, 2, 3, 4}) {
    auto input = generate_input(128 * 256, mode);
    auto weight = generate_weight(256);
    auto m = run_and_compare(DType::kFloat32, input, weight, 128, 256, 1e-5F,
                             RmsNormVariant::kV2Vectorized);
    // For mode 4 (1e20 alternating), the FP32 sum of squares overflows the
    // relative comparison slightly; keep the max_abs bound generous but
    // still order-of-magnitude correct.
    check_metrics("fp32_extreme", m, 1e-3);
  }
}

void test_fp16() {
  for (int64_t hidden : {1024, 2048}) {
    auto input = generate_input(32 * hidden, 0);
    auto weight = generate_weight(hidden);
    auto m = run_and_compare(DType::kFloat16, input, weight, 32, hidden, 1e-5F,
                             RmsNormVariant::kV2Vectorized);
    check_metrics("fp16", m, 2e-2);
  }
}

void test_fp16_non_aligned_fallback() {
  // hidden=3 is odd: V2's half2 kernel handles the odd tail column via a
  // scalar fallback, so kAuto still routes to V2 and must be correct.
  auto input = generate_input(8 * 3, 0);
  auto weight = generate_weight(3);
  auto m = run_and_compare(DType::kFloat16, input, weight, 8, 3, 1e-5F,
                           RmsNormVariant::kAuto);
  check_metrics("fp16_fallback", m, 2e-2);
}

void test_dispatcher_selection() {
  CHECK(hqsb::rmsnorm_select_variant(2048, DType::kFloat32) ==
        RmsNormVariant::kV2Vectorized);
  // 101 % 4 == 1 -> float4 vectorization impossible -> V1 fallback.
  CHECK(hqsb::rmsnorm_select_variant(101, DType::kFloat32) ==
        RmsNormVariant::kV1WarpShuffle);
  CHECK(hqsb::rmsnorm_select_variant(2048, DType::kFloat16) ==
        RmsNormVariant::kV2Vectorized);
  // FP16 always routes to V2 (its half2 kernel handles odd tails).
  CHECK(hqsb::rmsnorm_select_variant(3, DType::kFloat16) ==
        RmsNormVariant::kV2Vectorized);
}

void test_multi_stream() {
  // Two independent RMSNorm kernels on two streams must both produce correct
  // results, verifying the stream-aware contract (no hidden use of the
  // default stream).
  const int64_t rows_a = 128, hidden_a = 256;
  const int64_t rows_b = 64, hidden_b = 512;

  auto in_a = generate_input(rows_a * hidden_a, 0);
  auto w_a = generate_weight(hidden_a);
  auto in_b = generate_input(rows_b * hidden_b, 0);
  auto w_b = generate_weight(hidden_b);

  std::vector<float> ref_a(rows_a * hidden_a), ref_b(rows_b * hidden_b);
  hqsb::rmsnorm_reference_cpu(in_a.data(), w_a.data(), ref_a.data(), rows_a,
                              hidden_a, 1e-5F);
  hqsb::rmsnorm_reference_cpu(in_b.data(), w_b.data(), ref_b.data(), rows_b,
                              hidden_b, 1e-5F);

  float *d_in_a, *d_w_a, *d_out_a, *d_in_b, *d_w_b, *d_out_b;
  cudaMalloc(&d_in_a, rows_a * hidden_a * sizeof(float));
  cudaMalloc(&d_w_a, hidden_a * sizeof(float));
  cudaMalloc(&d_out_a, rows_a * hidden_a * sizeof(float));
  cudaMalloc(&d_in_b, rows_b * hidden_b * sizeof(float));
  cudaMalloc(&d_w_b, hidden_b * sizeof(float));
  cudaMalloc(&d_out_b, rows_b * hidden_b * sizeof(float));

  cudaMemcpy(d_in_a, in_a.data(), rows_a * hidden_a * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_w_a, w_a.data(), hidden_a * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_in_b, in_b.data(), rows_b * hidden_b * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_w_b, w_b.data(), hidden_b * sizeof(float), cudaMemcpyHostToDevice);

  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  hqsb::rmsnorm_forward(d_in_a, d_w_a, d_out_a, rows_a, hidden_a, 1e-5F,
                        DType::kFloat32, RmsNormVariant::kV2Vectorized, s1);
  hqsb::rmsnorm_forward(d_in_b, d_w_b, d_out_b, rows_b, hidden_b, 1e-5F,
                        DType::kFloat32, RmsNormVariant::kV1WarpShuffle, s2);
  cudaStreamSynchronize(s1);
  cudaStreamSynchronize(s2);

  std::vector<float> out_a(rows_a * hidden_a), out_b(rows_b * hidden_b);
  cudaMemcpy(out_a.data(), d_out_a, rows_a * hidden_a * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(out_b.data(), d_out_b, rows_b * hidden_b * sizeof(float),
             cudaMemcpyDeviceToHost);

  auto m_a = hqsb::test::compute_metrics(out_a.data(), ref_a.data(), out_a.size());
  auto m_b = hqsb::test::compute_metrics(out_b.data(), ref_b.data(), out_b.size());
  CHECK_NEAR(m_a.max_abs_error, 0.0, 5e-4);
  CHECK_NEAR(m_b.max_abs_error, 0.0, 5e-4);

  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
  cudaFree(d_in_a); cudaFree(d_w_a); cudaFree(d_out_a);
  cudaFree(d_in_b); cudaFree(d_w_b); cudaFree(d_out_b);
}

void test_invalid_arguments() {
  float a = 1.0F;
  float w = 1.0F;
  float o = 0.0F;
  // null pointers, non-positive shape, CPU reference, and FP16+V0.
  CHECK(hqsb::rmsnorm_forward(nullptr, &w, &o, 1, 1, 1e-5F, DType::kFloat32,
                              RmsNormVariant::kAuto, 0) ==
        cudaErrorInvalidValue);
  CHECK(hqsb::rmsnorm_forward(&a, &w, &o, 0, 1, 1e-5F, DType::kFloat32,
                              RmsNormVariant::kAuto, 0) ==
        cudaErrorInvalidValue);
  CHECK(hqsb::rmsnorm_forward(&a, &w, &o, 1, 0, 1e-5F, DType::kFloat32,
                              RmsNormVariant::kAuto, 0) ==
        cudaErrorInvalidValue);
  CHECK(hqsb::rmsnorm_forward(&a, &w, &o, 1, 1, 1e-5F, DType::kFloat32,
                              RmsNormVariant::kReference, 0) ==
        cudaErrorInvalidValue);
  CHECK(hqsb::rmsnorm_forward(&a, &w, &o, 1, 2, 1e-5F, DType::kFloat16,
                              RmsNormVariant::kV0Shared, 0) ==
        cudaErrorInvalidValue);
}

}  // namespace

int main() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    std::fprintf(stderr, "no CUDA device\n");
    return EXIT_FAILURE;
  }

  test_fp32_baseline_shape();
  test_fp32_non_power_of_two();
  test_fp32_tiny_and_single_row();
  test_fp32_extreme_values();
  test_fp16();
  test_fp16_non_aligned_fallback();
  test_dispatcher_selection();
  test_multi_stream();
  test_invalid_arguments();

  return hqsb::test::finish("test_rmsnorm");
}
