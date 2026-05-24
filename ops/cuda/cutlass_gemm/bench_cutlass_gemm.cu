// HQSB CUTLASS GEMM comparison / correctness driver (S04).
//
// A minimal, self-contained FP16 CUTLASS GEMM driver that:
//   * instantiates cutlass::gemm::device::Gemm with an explicit
//     tensor-op configuration (no hand-tuned tile), matching the
//     "out-of-the-box" vendor-library experience of cuBLAS and Triton
//     autotune;
//   * **selects a tile whose shared-memory footprint fits the device**, so
//     the same binary is usable on sm_80 / sm_86 / sm_87 instead of only on
//     architectures with a 163 KiB per-block shared-memory budget;
//   * verifies against an FP64 host reference and reports the five standard
//     hqsb::test metrics (see ops/cuda/common/test_metrics.h);
//   * reports median device time via cudaEvent.
//
// Cross-architecture background (S04 补齐)
// ----------------------------------------
// The CUTLASS default for Sm80 tensor-op FP16 resolves to
// ThreadblockShape<128,256,64> x 3 stages = 147456 B (144 KiB) of *dynamic*
// shared memory. CUTLASS programs the kernel with
// ``cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, 144 KiB)``
// (see third_party/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h)
// and turns a failure into ``cutlass::Status::kErrorInternal``.
//
//   sm_80 / sm_87 : opt-in smem 163 KiB -> 144 KiB fits;
//   sm_86 (RTX 30): opt-in smem  99 KiB -> the attribute call fails with
//                   cudaErrorInvalidValue and the kernel never launches.
//
// Rather than ignore that (which used to emit a timing for a kernel that
// never executed), this driver picks the largest configuration the device can
// actually hold and fails loudly when correctness is not met.
//
// Exit codes (CLI contract)
// -------------------------
//   0  GEMM ran and passed the correctness gate
//   2  the kernel did not run (launch/attribute failure)
//   3  the kernel ran but produced numerically wrong output
//   4  no supported configuration fits this device's shared-memory budget
//
// Usage:
//   hqsb_cutlass_gemm_bench [--m M --n N --k K] [--warmup W] [--iterations I]
//                           [--config auto|large|compact]

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "test_metrics.h"

namespace {

using ElementInput = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;
using Layout = cutlass::layout::RowMajor;

using Epilogue = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 8, ElementAccumulator, ElementAccumulator>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;

// ── Compile-time configurations ─────────────────────────────────────────
//
// Explicit tensor-op + Sm80 so the FP16 tensor-core MMA path is selected
// (the bare 8-argument form defaults to SIMT Sm70, which never uses the
// tensor cores and is not a meaningful vendor-library comparison).
//
// "large": 128x256x64 x 3 stages -> 147456 B dynamic smem (sm_80 / sm_87).
// "compact": 128x128x64 x 3 stages -> 98304 B dynamic smem (also sm_86).
using GemmLargeSmem = cutlass::gemm::device::Gemm<
    ElementInput, Layout,
    ElementInput, Layout,
    ElementOutput, Layout,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue,
    Swizzle,
    3>;

using GemmSmallSmem = cutlass::gemm::device::Gemm<
    ElementInput, Layout,
    ElementInput, Layout,
    ElementOutput, Layout,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue,
    Swizzle,
    3>;

/// Dynamic shared memory this configuration asks the device for.
template <typename GemmT>
constexpr size_t shared_storage_size() {
  return sizeof(typename GemmT::GemmKernel::SharedStorage);
}

/// FP64-accumulated reference GEMM (host-side, correctness only).
///
/// Accumulating in double matters: the kernels under test accumulate in
/// FP32, so an FP32 host reference would carry an error of the same order as
/// the thing being measured and the comparison would be meaningless.
void reference_gemm_host(const std::vector<ElementInput>& a,
                         const std::vector<ElementInput>& b,
                         std::vector<double>& c,
                         int m, int n, int k) {
  c.assign(static_cast<size_t>(m) * n, 0.0);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int kk = 0; kk < k; ++kk) {
        acc += static_cast<double>(float(a[static_cast<size_t>(i) * k + kk])) *
               static_cast<double>(float(b[static_cast<size_t>(kk) * n + j]));
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

/// Correctness gate for an FP16-output GEMM.
///
/// Both gates are scale-free on purpose. A per-element
/// ``atol + rtol * |expected|`` bound is unsound here: the reduction length
/// ``K`` (not the output magnitude) sets the achievable agreement, so an
/// output that happens to be near zero can carry an error far larger than its
/// own value. The relative L2 error and the cosine similarity normalize by
/// the reference's norm and therefore stay meaningful at any magnitude, while
/// a genuinely broken kernel collapses the cosine towards zero.
///
/// Thresholds follow the FP16 output quantization floor (~2^-11):
///   * l2_relative_error <= 1e-2   (about 20x the quantization floor)
///   * cosine_similarity >= 0.9999
bool correctness_ok(const hqsb::test::Metrics& m) {
  constexpr double kL2RelativeTolerance = 1e-2;
  constexpr double kMinCosineSimilarity = 0.9999;
  return m.l2_relative_error <= kL2RelativeTolerance &&
         m.cosine_similarity >= kMinCosineSimilarity;
}

/// Run one configuration end to end; returns a process exit code.
template <typename GemmT>
int run_config(const char* config_name,
               int m, int n, int k, int warmup, int iterations,
               size_t device_optin_smem) {
  const size_t smem = shared_storage_size<GemmT>();
  std::printf("config=%s shared_storage_bytes=%zu device_optin_smem_bytes=%zu\n",
              config_name, smem, device_optin_smem);
  if (smem > device_optin_smem) {
    std::fprintf(stderr,
                 "hqsb_cutlass_gemm_bench: config '%s' needs %zu B of dynamic "
                 "shared memory but this device allows at most %zu B per "
                 "block.\n",
                 config_name, smem, device_optin_smem);
    return 4;
  }

  const size_t size_a = static_cast<size_t>(m) * k;
  const size_t size_b = static_cast<size_t>(k) * n;
  const size_t size_c = static_cast<size_t>(m) * n;

  std::vector<ElementInput> h_a(size_a), h_b(size_b);
  randomize(h_a, 0);
  randomize(h_b, 17);

  ElementInput *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  cudaMalloc(&d_a, size_a * sizeof(ElementInput));
  cudaMalloc(&d_b, size_b * sizeof(ElementInput));
  cudaMalloc(&d_c, size_c * sizeof(ElementInput));
  if (d_a == nullptr || d_b == nullptr || d_c == nullptr) {
    std::fprintf(stderr, "hqsb_cutlass_gemm_bench: cudaMalloc failed\n");
    return 2;
  }
  cudaMemcpy(d_a, h_a.data(), size_a * sizeof(ElementInput), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), size_b * sizeof(ElementInput), cudaMemcpyHostToDevice);

  GemmT gemm_op;
  cutlass::gemm::GemmCoord problem_size(m, n, k);
  cutlass::TensorRef<ElementInput const, Layout> ref_a(d_a, Layout(k));
  cutlass::TensorRef<ElementInput const, Layout> ref_b(d_b, Layout(n));
  cutlass::TensorRef<ElementOutput const, Layout> ref_c(d_c, Layout(n));
  cutlass::TensorRef<ElementOutput, Layout> ref_d(d_c, Layout(n));
  typename GemmT::EpilogueOutputOp::Params epilogue(
      ElementAccumulator(1), ElementAccumulator(0));
  typename GemmT::Arguments args(problem_size, ref_a, ref_b, ref_c, ref_d, epilogue);

  // Fail loudly if the kernel cannot run -- never report a timing or a
  // correctness number for a kernel that never executed.
  cutlass::Status status = gemm_op(args);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
                 "hqsb_cutlass_gemm_bench: CUTLASS GEMM did not run: %s "
                 "(cudaError=%s). Refusing to report a timing/correctness "
                 "pair from a kernel that never executed.\n",
                 cutlassGetStatusString(status),
                 cudaGetErrorString(cudaGetLastError()));
    return 2;
  }

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

  // ── Correctness vs the FP64 host reference ────────────────────────────
  std::vector<ElementInput> h_c(size_c);
  cudaMemcpy(h_c.data(), d_c, size_c * sizeof(ElementInput), cudaMemcpyDeviceToHost);
  std::vector<double> ref;
  reference_gemm_host(h_a, h_b, ref, m, n, k);

  std::vector<float> actual(size_c), expected(size_c);
  for (size_t i = 0; i < size_c; ++i) {
    actual[i] = float(h_c[i]);
    expected[i] = static_cast<float>(ref[i]);
  }
  const hqsb::test::Metrics metrics =
      hqsb::test::compute_metrics(actual.data(), expected.data(), size_c);

  std::printf(
      "  max_abs_error=%.6f mean_abs_error=%.6f rmse=%.6f "
      "cosine=%.8f l2_relative_error=%.6f\n",
      metrics.max_abs_error, metrics.mean_abs_error, metrics.rmse,
      metrics.cosine_similarity, metrics.l2_relative_error);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  std::printf("  device=%s sm_%d%d\n", prop.name, prop.major, prop.minor);

  // Machine-readable line, kept as the LAST stdout line:
  //   dtype,m,n,k,config,median_ms,max_err
  std::printf("fp16,%d,%d,%d,cutlass_%s,%.4f,%.4f\n",
              m, n, k, config_name, median_ms, metrics.max_abs_error);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  if (!correctness_ok(metrics)) {
    std::fprintf(stderr,
                 "hqsb_cutlass_gemm_bench: CORRECTNESS FAILED for config '%s' "
                 "(m=%d n=%d k=%d): l2_relative_error=%.6g (limit 1e-2), "
                 "cosine=%.8f (min 0.9999)\n",
                 config_name, m, n, k, metrics.l2_relative_error,
                 metrics.cosine_similarity);
    return 3;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  int m = 512, n = 2048, k = 2048;
  int warmup = 10, iterations = 50;
  std::string config = "auto";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--m") m = std::atoi(next());
    else if (a == "--n") n = std::atoi(next());
    else if (a == "--k") k = std::atoi(next());
    else if (a == "--warmup") warmup = std::atoi(next());
    else if (a == "--iterations") iterations = std::atoi(next());
    else if (a == "--config") config = next();
    else if (a == "--help") {
      std::printf(
          "usage: hqsb_cutlass_gemm_bench [--m M --n N --k K]\n"
          "       [--warmup W] [--iterations I]\n"
          "       [--config auto|large|compact]\n");
      return 0;
    }
  }

  // Per-block dynamic shared-memory budget actually granted by this device.
  int optin = 0;
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  const size_t device_optin_smem = static_cast<size_t>(optin);

  const size_t large_smem = shared_storage_size<GemmLargeSmem>();
  const size_t small_smem = shared_storage_size<GemmSmallSmem>();

  if (config == "large") {
    return run_config<GemmLargeSmem>("large", m, n, k, warmup, iterations,
                                     device_optin_smem);
  }
  if (config == "compact") {
    return run_config<GemmSmallSmem>("compact", m, n, k, warmup, iterations,
                                     device_optin_smem);
  }
  if (config != "auto") {
    std::fprintf(stderr, "hqsb_cutlass_gemm_bench: unknown --config '%s'\n",
                 config.c_str());
    return 4;
  }

  // auto: prefer the wider tile when the device can hold it, otherwise take
  // the compact one; report 4 only when even the compact one does not fit.
  if (large_smem <= device_optin_smem) {
    return run_config<GemmLargeSmem>("large", m, n, k, warmup, iterations,
                                     device_optin_smem);
  }
  if (small_smem <= device_optin_smem) {
    // Record why the default configuration was skipped -- an arch mismatch
    // must be visible in the evidence, not silently absorbed.
    std::fprintf(stderr,
                 "hqsb_cutlass_gemm_bench: default 'large' config needs %zu B "
                 "> device opt-in %zu B; falling back to 'compact'\n",
                 large_smem, device_optin_smem);
    return run_config<GemmSmallSmem>("compact", m, n, k, warmup, iterations,
                                     device_optin_smem);
  }

  std::fprintf(stderr,
               "hqsb_cutlass_gemm_bench: no supported configuration fits this "
               "device (large=%zu B, compact=%zu B, device opt-in=%zu B)\n",
               large_smem, small_smem, device_optin_smem);
  return 4;
}
