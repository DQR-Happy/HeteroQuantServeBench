// HQSB CUTLASS GEMM comparison benchmark (S04).
//
// A minimal, self-contained FP16 CUTLASS GEMM driver that:
//   * instantiates cutlass::gemm::device::Gemm with the DEFAULT
//     tensor-op configuration (no hand-tuned tile), matching the
//     "out-of-the-box" vendor-library experience of cuBLAS and Triton
//     autotune;
//   * verifies against a naive FP32-accumulated reference;
//   * reports median device time via cudaEvent.
//
// This is the "template library peak" data point in the S04 comparison
// (cuBLAS vs Triton vs CUTLASS). It depends on the external CUTLASS headers.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

namespace {

using ElementInput = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;
using Layout = cutlass::layout::RowMajor;

// Explicit tensor-op + Sm80 so the FP16 tensor-core MMA path is selected
// (the bare 8-argument form defaults to SIMT Sm70, which never uses the
// tensor cores and is not a meaningful vendor-library comparison).
using CutlassGemm = cutlass::gemm::device::Gemm<
    ElementInput, Layout,
    ElementInput, Layout,
    ElementOutput, Layout,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80>;

// Naive FP32-accumulated reference GEMM (host-side, correctness only).
void reference_gemm_host(const std::vector<ElementInput>& a,
                         const std::vector<ElementInput>& b,
                         std::vector<float>& c,
                         int m, int n, int k) {
  c.assign(static_cast<size_t>(m) * n, 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        acc += float(a[static_cast<size_t>(i) * k + kk]) *
               float(b[static_cast<size_t>(kk) * n + j]);
      }
      c[static_cast<size_t>(i) * n + j] = acc;
    }
  }
}

void randomize(std::vector<ElementInput>& v, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  for (auto& x : v) {
    x = ElementInput::convert(d(rng));
  }
}

}  // namespace

int main(int argc, char** argv) {
  int m = 512, n = 2048, k = 2048;
  int warmup = 10, iterations = 50;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--m") m = std::atoi(next());
    else if (a == "--n") n = std::atoi(next());
    else if (a == "--k") k = std::atoi(next());
    else if (a == "--warmup") warmup = std::atoi(next());
    else if (a == "--iterations") iterations = std::atoi(next());
    else if (a == "--help") {
      std::printf("usage: bench_cutlass_gemm [--m M --n N --k K]\n");
      return 0;
    }
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  const size_t size_a = static_cast<size_t>(m) * k;
  const size_t size_b = static_cast<size_t>(k) * n;
  const size_t size_c = static_cast<size_t>(m) * n;

  std::vector<ElementInput> h_a(size_a), h_b(size_b);
  randomize(h_a, 0);
  randomize(h_b, 17);

  ElementInput *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size_a * sizeof(ElementInput));
  cudaMalloc(&d_b, size_b * sizeof(ElementInput));
  cudaMalloc(&d_c, size_c * sizeof(ElementInput));
  cudaMemcpy(d_a, h_a.data(), size_a * sizeof(ElementInput), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), size_b * sizeof(ElementInput), cudaMemcpyHostToDevice);

  // CUTLASS GEMM operator + arguments.
  CutlassGemm gemm_op;
  cutlass::gemm::GemmCoord problem_size(m, n, k);
  cutlass::TensorRef<ElementInput const, Layout> ref_a(d_a, Layout(k));
  cutlass::TensorRef<ElementInput const, Layout> ref_b(d_b, Layout(n));
  cutlass::TensorRef<ElementOutput const, Layout> ref_c(d_c, Layout(n));
  cutlass::TensorRef<ElementOutput, Layout> ref_d(d_c, Layout(n));
  typename CutlassGemm::EpilogueOutputOp::Params epilogue(
      ElementAccumulator(1), ElementAccumulator(0));
  CutlassGemm::Arguments args(problem_size, ref_a, ref_b, ref_c, ref_d, epilogue);

  // Warmup.
  for (int i = 0; i < warmup; ++i) {
    gemm_op(args);
  }
  cudaDeviceSynchronize();

  // Timed region (median).
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  std::vector<float> times;
  times.reserve(iterations);
  for (int i = 0; i < iterations; ++i) {
    cudaEventRecord(start);
    gemm_op(args);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    times.push_back(ms);
  }
  std::sort(times.begin(), times.end());
  const float median_ms = times[times.size() / 2];
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Correctness vs host reference.
  std::vector<ElementInput> h_c(size_c);
  cudaMemcpy(h_c.data(), d_c, size_c * sizeof(ElementInput), cudaMemcpyDeviceToHost);
  std::vector<float> ref;
  reference_gemm_host(h_a, h_b, ref, m, n, k);

  float max_err = 0.0f;
  for (size_t i = 0; i < size_c; ++i) {
    max_err = std::max(max_err, std::fabs(float(h_c[i]) - ref[i]));
  }

  std::printf("fp16,%d,%d,%d,cutlass_default,%.4f,%.4f\n",
              m, n, k, median_ms, max_err);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}
