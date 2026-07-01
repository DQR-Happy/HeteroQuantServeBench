// HQSB CUTLASS GEMM comparison / correctness driver (S04).
//
// This unified driver preserves both the cross-architecture safeguards from
// main and the later Jetson experiment instrumentation:
//   * selects a tensor-op tile that fits the device's opt-in shared memory;
//   * uses explicit can_implement / initialize / run checks;
//   * guards both sides of D so tail writes are observable;
//   * verifies bounded cases against an FP64 host reference with the common
//     HQSB metrics, while --no-verify avoids large pageable host allocations;
//   * emits a JSON record for E04 audit drivers followed by the legacy CSV
//     record consumed by scripts/bench/bench_s04.py.
//
// The logical operation is A[M,K] row-major times W[N,K], exposed to CUTLASS
// as a column-major B[K,N]. This matches the weight_NK convention used by the
// recovered E04 experiment drivers.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
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
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;

using Epilogue = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 8, ElementAccumulator, ElementAccumulator>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;

// The default-like wide tile needs about 144 KiB dynamic shared memory. The
// compact tile needs about 96 KiB and therefore also fits sm_86-class devices.
using GemmLargeSmem = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA,
    ElementInput, LayoutB,
    ElementOutput, LayoutD,
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
    ElementInput, LayoutA,
    ElementInput, LayoutB,
    ElementOutput, LayoutD,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue,
    Swizzle,
    3>;

template <typename GemmT>
constexpr size_t shared_storage_size() {
  return sizeof(typename GemmT::GemmKernel::SharedStorage);
}

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
               static_cast<double>(float(b[static_cast<size_t>(j) * k + kk]));
      }
      c[static_cast<size_t>(i) * n + j] = acc;
    }
  }
}

void randomize(std::vector<ElementInput>& values, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  for (auto& value : values) {
    value = ElementInput::convert(distribution(rng));
  }
}

bool correctness_ok(const hqsb::test::Metrics& metrics) {
  constexpr double kL2RelativeTolerance = 1e-2;
  constexpr double kMinCosineSimilarity = 0.9999;
  return metrics.l2_relative_error <= kL2RelativeTolerance &&
         metrics.cosine_similarity >= kMinCosineSimilarity;
}

void free_buffers(ElementInput* d_a,
                  ElementInput* d_b,
                  std::uint8_t* d_c_guarded) {
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c_guarded);
}

template <typename GemmT>
int run_config(const char* config_name,
               int m,
               int n,
               int k,
               int warmup,
               int iterations,
               bool verify,
               size_t device_optin_smem,
               const cudaDeviceProp& prop) {
  constexpr size_t kGuardBytes = 4096;
  const size_t smem = shared_storage_size<GemmT>();
  if (smem > device_optin_smem) {
    std::fprintf(stderr,
                 "hqsb_cutlass_gemm_bench: config '%s' needs %zu B dynamic "
                 "shared memory but the device allows %zu B\n",
                 config_name, smem, device_optin_smem);
    return 4;
  }

  const size_t size_a = static_cast<size_t>(m) * k;
  const size_t size_b = static_cast<size_t>(k) * n;
  const size_t size_c = static_cast<size_t>(m) * n;

  std::vector<ElementInput> h_a;
  std::vector<ElementInput> h_b;
  if (verify) {
    h_a.resize(size_a);
    h_b.resize(size_b);
    randomize(h_a, 0);
    randomize(h_b, 17);
  }

  ElementInput* d_a = nullptr;
  ElementInput* d_b = nullptr;
  std::uint8_t* d_c_guarded = nullptr;
  if (cudaMalloc(&d_a, size_a * sizeof(ElementInput)) != cudaSuccess ||
      cudaMalloc(&d_b, size_b * sizeof(ElementInput)) != cudaSuccess ||
      cudaMalloc(&d_c_guarded,
                 size_c * sizeof(ElementOutput) + 2 * kGuardBytes) !=
          cudaSuccess) {
    std::fprintf(stderr, "hqsb_cutlass_gemm_bench: cudaMalloc failed\n");
    free_buffers(d_a, d_b, d_c_guarded);
    return 2;
  }
  ElementOutput* d_c = reinterpret_cast<ElementOutput*>(
      d_c_guarded + kGuardBytes);
  cudaMemset(d_c_guarded, 0xA5,
             size_c * sizeof(ElementOutput) + 2 * kGuardBytes);
  if (verify) {
    cudaMemcpy(d_a, h_a.data(), size_a * sizeof(ElementInput),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size_b * sizeof(ElementInput),
               cudaMemcpyHostToDevice);
  } else {
    cudaMemset(d_a, 0x3C, size_a * sizeof(ElementInput));
    cudaMemset(d_b, 0x3C, size_b * sizeof(ElementInput));
  }

  GemmT gemm_op;
  cutlass::gemm::GemmCoord problem_size(m, n, k);
  cutlass::TensorRef<ElementInput const, LayoutA> ref_a(d_a, LayoutA(k));
  cutlass::TensorRef<ElementInput const, LayoutB> ref_b(d_b, LayoutB(k));
  cutlass::TensorRef<ElementOutput const, LayoutD> ref_c(d_c, LayoutD(n));
  cutlass::TensorRef<ElementOutput, LayoutD> ref_d(d_c, LayoutD(n));
  typename GemmT::EpilogueOutputOp::Params epilogue(
      ElementAccumulator(1), ElementAccumulator(0));
  typename GemmT::Arguments args(problem_size, ref_a, ref_b, ref_c, ref_d,
                                 epilogue);

  const cutlass::Status can_status = gemm_op.can_implement(args);
  if (can_status != cutlass::Status::kSuccess) {
    std::printf(
        "{\"backend\":\"cutlass_tensorop_sm80\",\"config\":\"%s\","
        "\"m\":%d,\"n\":%d,\"k\":%d,\"can_implement\":\"%s\","
        "\"status\":\"EXPECTED_UNSUPPORTED\"}\n",
        config_name, m, n, k, cutlassGetStatusString(can_status));
    free_buffers(d_a, d_b, d_c_guarded);
    return 4;
  }
  const cutlass::Status init_status = gemm_op.initialize(args);
  if (init_status != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "CUTLASS initialize failed: %s (cudaError=%s)\n",
                 cutlassGetStatusString(init_status),
                 cudaGetErrorString(cudaGetLastError()));
    free_buffers(d_a, d_b, d_c_guarded);
    return 5;
  }

  for (int i = 0; i < warmup; ++i) {
    if (gemm_op.run() != cutlass::Status::kSuccess) {
      free_buffers(d_a, d_b, d_c_guarded);
      return 6;
    }
  }
  cudaDeviceSynchronize();

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  std::vector<float> device_ms;
  std::vector<double> host_completion_us;
  device_ms.reserve(iterations);
  host_completion_us.reserve(iterations);
  for (int i = 0; i < iterations; ++i) {
    const auto host_begin = std::chrono::steady_clock::now();
    cudaEventRecord(start);
    const cutlass::Status run_status = gemm_op.run();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    const auto host_end = std::chrono::steady_clock::now();
    if (run_status != cutlass::Status::kSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      free_buffers(d_a, d_b, d_c_guarded);
      return 7;
    }
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    device_ms.push_back(elapsed_ms);
    host_completion_us.push_back(
        std::chrono::duration<double, std::micro>(host_end - host_begin).count());
  }
  std::sort(device_ms.begin(), device_ms.end());
  const float median_ms = device_ms[device_ms.size() / 2];
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::vector<std::uint8_t> prefix(kGuardBytes);
  std::vector<std::uint8_t> suffix(kGuardBytes);
  cudaMemcpy(prefix.data(), d_c_guarded, kGuardBytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(suffix.data(),
             d_c_guarded + kGuardBytes + size_c * sizeof(ElementOutput),
             kGuardBytes, cudaMemcpyDeviceToHost);
  const auto guard_clean = [](const std::vector<std::uint8_t>& bytes) {
    return std::all_of(bytes.begin(), bytes.end(),
                       [](std::uint8_t value) { return value == 0xA5; });
  };
  const bool guards_ok = guard_clean(prefix) && guard_clean(suffix);

  hqsb::test::Metrics metrics;
  if (verify) {
    std::vector<ElementOutput> h_c(size_c);
    cudaMemcpy(h_c.data(), d_c, size_c * sizeof(ElementOutput),
               cudaMemcpyDeviceToHost);
    std::vector<double> reference;
    reference_gemm_host(h_a, h_b, reference, m, n, k);
    std::vector<float> actual(size_c);
    std::vector<float> expected(size_c);
    for (size_t i = 0; i < size_c; ++i) {
      actual[i] = float(h_c[i]);
      expected[i] = static_cast<float>(reference[i]);
    }
    metrics = hqsb::test::compute_metrics(actual.data(), expected.data(),
                                           size_c);
  }

  const bool passed = guards_ok && (!verify || correctness_ok(metrics));
  const double tflops =
      (2.0 * static_cast<double>(m) * n * k) /
      (static_cast<double>(median_ms) * 1.0e9);
  std::printf(
      "{\"backend\":\"cutlass_tensorop_sm80\","
      "\"actual_backend\":\"cutlass_tensorop_sm80\",\"config\":\"%s\","
      "\"device\":\"%s\",\"sm\":%d,\"m\":%d,\"n\":%d,\"k\":%d,"
      "\"layout_a\":\"row_major\","
      "\"layout_b\":\"column_major_logical_from_weight_NK\","
      "\"layout_d\":\"row_major\",\"dtype\":\"fp16\","
      "\"accumulation\":\"fp32\",\"shared_storage_bytes\":%zu,"
      "\"device_optin_smem_bytes\":%zu,\"workspace_bytes\":0,"
      "\"guard_bytes_each_side\":%zu,\"guards_ok\":%s,\"verify\":%s,"
      "\"max_abs\":%.9g,\"mean_abs\":%.9g,\"rmse\":%.9g,"
      "\"cosine\":%.9g,\"l2_relative_error\":%.9g,"
      "\"median_ms\":%.9g,\"tflops\":%.9g,\"device_ms_raw\":[",
      config_name, prop.name, prop.major * 10 + prop.minor, m, n, k, smem,
      device_optin_smem, kGuardBytes, guards_ok ? "true" : "false",
      verify ? "true" : "false", metrics.max_abs_error,
      metrics.mean_abs_error, metrics.rmse, metrics.cosine_similarity,
      metrics.l2_relative_error, median_ms, tflops);
  for (size_t i = 0; i < device_ms.size(); ++i) {
    std::printf("%s%.9g", i ? "," : "", device_ms[i]);
  }
  std::printf("],\"host_completion_us_raw\":[");
  for (size_t i = 0; i < host_completion_us.size(); ++i) {
    std::printf("%s%.9g", i ? "," : "", host_completion_us[i]);
  }
  std::printf("],\"status\":\"%s\"}\n", passed ? "PASS" : "FAIL");

  // Keep this CSV as the final line for the original S04 benchmark parser.
  std::printf("fp16,%d,%d,%d,cutlass_%s,%.4f,%.6f\n", m, n, k,
              config_name, median_ms, metrics.max_abs_error);

  free_buffers(d_a, d_b, d_c_guarded);
  return passed ? 0 : 8;
}

}  // namespace

int main(int argc, char** argv) {
  int m = 512;
  int n = 2048;
  int k = 2048;
  int warmup = 10;
  int iterations = 50;
  bool verify = true;
  std::string config = "auto";

  for (int i = 1; i < argc; ++i) {
    std::string argument = argv[i];
    auto next = [&]() -> const char* {
      return (i + 1 < argc) ? argv[++i] : "";
    };
    if (argument == "--m") m = std::atoi(next());
    else if (argument == "--n") n = std::atoi(next());
    else if (argument == "--k") k = std::atoi(next());
    else if (argument == "--warmup") warmup = std::atoi(next());
    else if (argument == "--iterations") iterations = std::atoi(next());
    else if (argument == "--config") config = next();
    else if (argument == "--no-verify") verify = false;
    else if (argument == "--help") {
      std::printf(
          "usage: hqsb_cutlass_gemm_bench [--m M --n N --k K]\n"
          "       [--warmup W] [--iterations I] [--no-verify]\n"
          "       [--config auto|large|compact]\n");
      return 0;
    }
  }
  if (m <= 0 || n <= 0 || k <= 0 || warmup < 0 || iterations <= 0) {
    std::fprintf(stderr, "invalid dimensions or iteration counts\n");
    return 2;
  }

  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    return 3;
  }
  int optin = 0;
  if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                             0) != cudaSuccess ||
      optin <= 0) {
    std::fprintf(stderr, "could not query opt-in shared-memory budget\n");
    return 3;
  }
  const size_t device_optin_smem = static_cast<size_t>(optin);
  const size_t large_smem = shared_storage_size<GemmLargeSmem>();
  const size_t small_smem = shared_storage_size<GemmSmallSmem>();

  if (config == "large") {
    return run_config<GemmLargeSmem>("large", m, n, k, warmup, iterations,
                                     verify, device_optin_smem, prop);
  }
  if (config == "compact") {
    return run_config<GemmSmallSmem>("compact", m, n, k, warmup, iterations,
                                     verify, device_optin_smem, prop);
  }
  if (config != "auto") {
    std::fprintf(stderr, "unknown --config '%s'\n", config.c_str());
    return 4;
  }
  if (large_smem <= device_optin_smem) {
    return run_config<GemmLargeSmem>("large", m, n, k, warmup, iterations,
                                     verify, device_optin_smem, prop);
  }
  if (small_smem <= device_optin_smem) {
    std::fprintf(stderr,
                 "large CUTLASS tile needs %zu B > device opt-in %zu B; "
                 "falling back to compact\n",
                 large_smem, device_optin_smem);
    return run_config<GemmSmallSmem>("compact", m, n, k, warmup, iterations,
                                     verify, device_optin_smem, prop);
  }
  std::fprintf(stderr,
               "no supported CUTLASS tile fits (large=%zu, compact=%zu, "
               "device=%zu)\n",
               large_smem, small_smem, device_optin_smem);
  return 4;
}
