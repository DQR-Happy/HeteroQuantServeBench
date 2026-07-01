// E04-05: frozen CUTLASS 2.x-style Gemm configuration-space driver.
//
// The repository pins CUTLASS 4.7, but this experiment deliberately uses the
// stable cutlass::gemm::device::Gemm API.  Each explicit template instance has
// a stable config id and exposes the complete can_implement -> workspace ->
// initialize -> run -> immediate CUDA -> completion CUDA -> numerical chain.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

namespace {

using Element = cutlass::half_t;
using Accumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
using Instruction = cutlass::gemm::GemmShape<16, 8, 16>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using Linear = cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>;
using LinearScalar = cutlass::epilogue::thread::LinearCombination<Element, 1, float, float>;
using Relu = cutlass::epilogue::thread::LinearCombinationRelu<Element, 8, float, float>;

template <typename TB, typename Warp, int Stages, int AlignA = 8,
          int AlignB = 8, bool SplitK = false, typename Epilogue = Linear>
using Gemm = cutlass::gemm::device::Gemm<
    Element, LayoutA, Element, LayoutB, Element, LayoutD, Accumulator,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, Warp,
    Instruction, Epilogue, Swizzle, Stages, AlignA, AlignB, SplitK>;

// Decode/small-M hypotheses: reduce wasted M rows while preserving N CTAs.
using Small32 = Gemm<cutlass::gemm::GemmShape<32, 128, 32>,
                       cutlass::gemm::GemmShape<32, 64, 32>, 2>;
using Small64 = Gemm<cutlass::gemm::GemmShape<64, 128, 32>,
                       cutlass::gemm::GemmShape<32, 64, 32>, 3>;
// Scalar-alignment control: accepts all-dimensional K tails at a throughput cost.
using SmallAlign1 = Gemm<cutlass::gemm::GemmShape<64, 64, 32>,
                          cutlass::gemm::GemmShape<32, 32, 32>, 2, 1, 1,
                          false, LinearScalar>;

// Prefill/large-M hypotheses: increase reuse and pipeline depth.
using Large128 = Gemm<cutlass::gemm::GemmShape<128, 128, 32>,
                        cutlass::gemm::GemmShape<64, 64, 32>, 3>;
using Large256 = Gemm<cutlass::gemm::GemmShape<128, 256, 32>,
                        cutlass::gemm::GemmShape<64, 64, 32>, 3>;
using LargeSplitK = Gemm<cutlass::gemm::GemmShape<128, 128, 32>,
                          cutlass::gemm::GemmShape<64, 64, 32>, 3, 8, 8, true>;
using LargeRelu = Gemm<cutlass::gemm::GemmShape<128, 128, 32>,
                         cutlass::gemm::GemmShape<64, 64, 32>, 3, 8, 8,
                         false, Relu>;

struct Options {
  std::string config = "small_m32n128k32_w32n64_s2_a8_linear";
  int m = 1, n = 2048, k = 2048;
  int warmup = 3, iterations = 9, split_k = 1;
  bool verify = false;
  bool misalign_a = false;
  bool omit_workspace = false;
  bool expect_rejection = false;
  std::string stream_mode = "default";
};

template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;
  size_t count = 0;
  explicit DeviceBuffer(size_t n = 0) : count(n) {
    if (n) cudaMalloc(&ptr, n * sizeof(T));
  }
  ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;
};

const char* cuda_name(cudaError_t status) {
  return status == cudaSuccess ? "success" : cudaGetErrorName(status);
}

void print_configs() {
  std::puts("{\"configs\":["
    "{\"id\":\"small_m32n128k32_w32n64_s2_a8_linear\",\"family\":\"decode_small_m\",\"cta\":[32,128,32],\"warp\":[32,64,32],\"instruction\":[16,8,16],\"stages\":2,\"alignment_a\":8,\"alignment_b\":8,\"epilogue\":\"linear\",\"split_k_serial\":false},"
    "{\"id\":\"small_m64n128k32_w32n64_s3_a8_linear\",\"family\":\"decode_small_m\",\"cta\":[64,128,32],\"warp\":[32,64,32],\"instruction\":[16,8,16],\"stages\":3,\"alignment_a\":8,\"alignment_b\":8,\"epilogue\":\"linear\",\"split_k_serial\":false},"
    "{\"id\":\"small_m64n64k32_w32n32_s2_a1_linear\",\"family\":\"decode_small_m\",\"cta\":[64,64,32],\"warp\":[32,32,32],\"instruction\":[16,8,16],\"stages\":2,\"alignment_a\":1,\"alignment_b\":1,\"epilogue\":\"linear\",\"split_k_serial\":false},"
    "{\"id\":\"large_m128n128k32_w64n64_s3_a8_linear\",\"family\":\"prefill_large_m\",\"cta\":[128,128,32],\"warp\":[64,64,32],\"instruction\":[16,8,16],\"stages\":3,\"alignment_a\":8,\"alignment_b\":8,\"epilogue\":\"linear\",\"split_k_serial\":false},"
    "{\"id\":\"large_m128n256k32_w64n64_s3_a8_linear\",\"family\":\"prefill_large_m\",\"cta\":[128,256,32],\"warp\":[64,64,32],\"instruction\":[16,8,16],\"stages\":3,\"alignment_a\":8,\"alignment_b\":8,\"epilogue\":\"linear\",\"split_k_serial\":false},"
    "{\"id\":\"large_m128n128k32_w64n64_s3_a8_splitk\",\"family\":\"prefill_large_m\",\"cta\":[128,128,32],\"warp\":[64,64,32],\"instruction\":[16,8,16],\"stages\":3,\"alignment_a\":8,\"alignment_b\":8,\"epilogue\":\"linear\",\"split_k_serial\":true},"
    "{\"id\":\"large_m128n128k32_w64n64_s3_a8_relu\",\"family\":\"epilogue_ablation\",\"cta\":[128,128,32],\"warp\":[64,64,32],\"instruction\":[16,8,16],\"stages\":3,\"alignment_a\":8,\"alignment_b\":8,\"epilogue\":\"relu\",\"split_k_serial\":false}"
    "]}");
}

void fill_random(std::vector<Element>& values, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-0.125f, 0.125f);
  for (auto& value : values) value = Element::convert(dist(gen));
}

template <typename GemmOp>
int execute(Options const& opt, bool relu_epilogue) {
  constexpr size_t guard_bytes = 4096;
  size_t size_a = size_t(opt.m) * opt.k;
  size_t size_b = size_t(opt.n) * opt.k;  // physical model weight [N,K]
  size_t size_d = size_t(opt.m) * opt.n;
  DeviceBuffer<Element> a_storage(size_a + 8);
  DeviceBuffer<Element> b(size_b);
  DeviceBuffer<std::uint8_t> d_guarded(size_d * sizeof(Element) + 2 * guard_bytes);
  if (!a_storage.ptr || !b.ptr || !d_guarded.ptr) return 20;
  Element* a = a_storage.ptr + (opt.misalign_a ? 1 : 0);
  Element* d = reinterpret_cast<Element*>(d_guarded.ptr + guard_bytes);
  cudaMemset(d_guarded.ptr, 0xA5, d_guarded.count);

  std::vector<Element> host_a, host_b;
  if (opt.verify) {
    host_a.resize(size_a); host_b.resize(size_b);
    fill_random(host_a, 405u + unsigned(opt.m));
    fill_random(host_b, 504u + unsigned(opt.n + opt.k));
    cudaMemcpy(a, host_a.data(), size_a * sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemcpy(b.ptr, host_b.data(), size_b * sizeof(Element), cudaMemcpyHostToDevice);
  } else {
    cudaMemset(a, 0, size_a * sizeof(Element));
    cudaMemset(b.ptr, 0, size_b * sizeof(Element));
  }

  cudaStream_t stream = nullptr, stream2 = nullptr;
  if (opt.stream_mode != "default") cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (opt.stream_mode == "dual") cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

  typename GemmOp::EpilogueOutputOp::Params epilogue(Accumulator(1), Accumulator(0));
  typename GemmOp::Arguments args(
      {opt.m, opt.n, opt.k}, {a, LayoutA(opt.k)}, {b.ptr, LayoutB(opt.k)},
      {d, LayoutD(opt.n)}, {d, LayoutD(opt.n)}, epilogue, opt.split_k);
  GemmOp gemm;
  cutlass::Status can = gemm.can_implement(args);
  size_t workspace_bytes = GemmOp::get_workspace_size(args);
  DeviceBuffer<std::uint8_t> workspace(opt.omit_workspace ? 0 : workspace_bytes);
  DeviceBuffer<std::uint8_t> d2_guarded(
      opt.stream_mode == "dual" ? size_d * sizeof(Element) + 2 * guard_bytes : 0);
  Element* d2 = d2_guarded.ptr
      ? reinterpret_cast<Element*>(d2_guarded.ptr + guard_bytes) : nullptr;
  if (d2_guarded.ptr) cudaMemset(d2_guarded.ptr, 0xA5, d2_guarded.count);
  typename GemmOp::Arguments args2(
      {opt.m, opt.n, opt.k}, {a, LayoutA(opt.k)}, {b.ptr, LayoutB(opt.k)},
      {d2, LayoutD(opt.n)}, {d2, LayoutD(opt.n)}, epilogue, opt.split_k);
  DeviceBuffer<std::uint8_t> workspace2(
      opt.stream_mode == "dual" && !opt.omit_workspace ? workspace_bytes : 0);
  cutlass::Status init = can == cutlass::Status::kSuccess
      ? gemm.initialize(args, workspace.ptr, stream) : can;
  GemmOp gemm2;
  cutlass::Status dual_init = opt.stream_mode == "dual" && can == cutlass::Status::kSuccess
      ? gemm2.initialize(args2, workspace2.ptr, stream2) : cutlass::Status::kSuccess;
  cutlass::Status run_status = init;
  cutlass::Status dual_run = dual_init;
  cudaError_t immediate = cudaSuccess;
  cudaError_t completion = cudaSuccess;
  std::vector<float> device_ms;
  std::vector<double> host_us;

  if (init == cutlass::Status::kSuccess) {
    for (int i = 0; i < opt.warmup; ++i) {
      run_status = gemm.run(stream);
      if (opt.stream_mode == "dual") dual_run = gemm2.run(stream2);
    }
    completion = cudaStreamSynchronize(stream);
    if (opt.stream_mode == "dual" && cudaStreamSynchronize(stream2) != cudaSuccess)
      completion = cudaErrorUnknown;
    for (int i = 0; i < opt.iterations && run_status == cutlass::Status::kSuccess; ++i) {
      cudaEvent_t begin, end;
      cudaEventCreate(&begin); cudaEventCreate(&end);
      auto host_begin = std::chrono::steady_clock::now();
      cudaEventRecord(begin, stream);
      run_status = gemm.run(stream);
      if (opt.stream_mode == "dual") dual_run = gemm2.run(stream2);
      immediate = cudaPeekAtLastError();
      cudaEventRecord(end, stream);
      completion = cudaEventSynchronize(end);
      if (opt.stream_mode == "dual" && cudaStreamSynchronize(stream2) != cudaSuccess)
        completion = cudaErrorUnknown;
      auto host_end = std::chrono::steady_clock::now();
      float ms = 0.0f; cudaEventElapsedTime(&ms, begin, end);
      device_ms.push_back(ms);
      host_us.push_back(std::chrono::duration<double, std::micro>(host_end - host_begin).count());
      cudaEventDestroy(begin); cudaEventDestroy(end);
    }
  }

  bool guards_ok = false;
  bool numerical_ok = false;
  float max_abs = 0.0f;
  size_t violations = 0;
  if (can == cutlass::Status::kSuccess && init == cutlass::Status::kSuccess &&
      run_status == cutlass::Status::kSuccess && dual_init == cutlass::Status::kSuccess &&
      dual_run == cutlass::Status::kSuccess && completion == cudaSuccess) {
    std::vector<std::uint8_t> prefix(guard_bytes), suffix(guard_bytes);
    cudaMemcpy(prefix.data(), d_guarded.ptr, guard_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(suffix.data(), d_guarded.ptr + guard_bytes + size_d * sizeof(Element),
               guard_bytes, cudaMemcpyDeviceToHost);
    auto clean = [](std::vector<std::uint8_t> const& xs) {
      return std::all_of(xs.begin(), xs.end(), [](std::uint8_t x) { return x == 0xA5; });
    };
    guards_ok = clean(prefix) && clean(suffix);
    if (opt.stream_mode == "dual") {
      cudaMemcpy(prefix.data(), d2_guarded.ptr, guard_bytes, cudaMemcpyDeviceToHost);
      cudaMemcpy(suffix.data(), d2_guarded.ptr + guard_bytes + size_d * sizeof(Element),
                 guard_bytes, cudaMemcpyDeviceToHost);
      guards_ok = guards_ok && clean(prefix) && clean(suffix);
    }
    numerical_ok = guards_ok;
    if (opt.verify) {
      std::vector<Element> host_d(size_d);
      cudaMemcpy(host_d.data(), d, size_d * sizeof(Element), cudaMemcpyDeviceToHost);
      for (int i = 0; i < opt.m; ++i) {
        for (int j = 0; j < opt.n; ++j) {
          float ref = 0.0f;
          for (int kk = 0; kk < opt.k; ++kk)
            ref += float(host_a[size_t(i) * opt.k + kk]) *
                   float(host_b[size_t(j) * opt.k + kk]);
          if (relu_epilogue) ref = std::max(0.0f, ref);
          float err = std::fabs(float(host_d[size_t(i) * opt.n + j]) - ref);
          max_abs = std::max(max_abs, err);
          if (err > 0.01f + 0.02f * std::fabs(ref)) ++violations;
        }
      }
      numerical_ok = guards_ok && violations == 0;
    }
  }

  float median_ms = 0.0f;
  if (!device_ms.empty()) {
    std::sort(device_ms.begin(), device_ms.end());
    median_ms = device_ms[device_ms.size() / 2];
  }
  std::sort(host_us.begin(), host_us.end());
  double host_median = host_us.empty() ? 0.0 : host_us[host_us.size() / 2];
  bool expected_rejection = opt.expect_rejection || opt.misalign_a || opt.omit_workspace ||
                            (!GemmOp::kSplitKSerial && opt.split_k > 1);
  bool stages_ok = can == cutlass::Status::kSuccess && init == cutlass::Status::kSuccess &&
                   dual_init == cutlass::Status::kSuccess &&
                   run_status == cutlass::Status::kSuccess &&
                   dual_run == cutlass::Status::kSuccess && immediate == cudaSuccess &&
                   completion == cudaSuccess && numerical_ok;
  bool passed = expected_rejection ? !stages_ok : stages_ok;
  double tflops = median_ms > 0 ? 2.0 * opt.m * opt.n * opt.k / median_ms / 1.0e9 : 0.0;

  std::printf("{\"config_id\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,"
              "\"split_k_slices\":%d,\"stream_mode\":\"%s\",\"misalign_a\":%s,"
              "\"omit_workspace\":%s,\"can_implement\":\"%s\","
              "\"workspace_bytes\":%zu,\"initialize\":\"%s\",\"run\":\"%s\","
              "\"dual_initialize\":\"%s\",\"dual_run\":\"%s\","
              "\"immediate_cuda\":\"%s\",\"completion_cuda\":\"%s\","
              "\"verify\":%s,\"guards_ok\":%s,\"numerical_ok\":%s,"
              "\"violations\":%zu,\"max_abs\":%.9g,\"median_ms\":%.9g,"
              "\"host_completion_us_median\":%.9g,\"tflops\":%.9g,"
              "\"expected_rejection\":%s,\"status\":\"%s\",\"device_ms_raw\":[",
              opt.config.c_str(), opt.m, opt.n, opt.k, opt.split_k,
              opt.stream_mode.c_str(), opt.misalign_a ? "true" : "false",
              opt.omit_workspace ? "true" : "false", cutlassGetStatusString(can),
              workspace_bytes, cutlassGetStatusString(init), cutlassGetStatusString(run_status),
              cutlassGetStatusString(dual_init), cutlassGetStatusString(dual_run),
              cuda_name(immediate), cuda_name(completion), opt.verify ? "true" : "false",
              guards_ok ? "true" : "false", numerical_ok ? "true" : "false", violations,
              max_abs, median_ms, host_median, tflops,
              expected_rejection ? "true" : "false", passed ? "PASS" : "FAIL");
  for (size_t i = 0; i < device_ms.size(); ++i)
    std::printf("%s%.9g", i ? "," : "", device_ms[i]);
  std::printf("]}\n");
  if (stream) cudaStreamDestroy(stream);
  if (stream2) cudaStreamDestroy(stream2);
  return passed ? 0 : 10;
}

template <typename Op>
int dispatch_execute(Options const& opt, bool relu = false) {
  return execute<Op>(opt, relu);
}

int run_config(Options const& opt) {
  if (opt.config == "small_m32n128k32_w32n64_s2_a8_linear") return dispatch_execute<Small32>(opt);
  if (opt.config == "small_m64n128k32_w32n64_s3_a8_linear") return dispatch_execute<Small64>(opt);
  if (opt.config == "small_m64n64k32_w32n32_s2_a1_linear") return dispatch_execute<SmallAlign1>(opt);
  if (opt.config == "large_m128n128k32_w64n64_s3_a8_linear") return dispatch_execute<Large128>(opt);
  if (opt.config == "large_m128n256k32_w64n64_s3_a8_linear") return dispatch_execute<Large256>(opt);
  if (opt.config == "large_m128n128k32_w64n64_s3_a8_splitk") return dispatch_execute<LargeSplitK>(opt);
  if (opt.config == "large_m128n128k32_w64n64_s3_a8_relu") return dispatch_execute<LargeRelu>(opt, true);
  std::fprintf(stderr, "unknown config: %s\n", opt.config.c_str());
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
    if (arg == "--list-configs") { print_configs(); return 0; }
    if (arg == "--config") opt.config = next();
    else if (arg == "--m") opt.m = std::atoi(next());
    else if (arg == "--n") opt.n = std::atoi(next());
    else if (arg == "--k") opt.k = std::atoi(next());
    else if (arg == "--warmup") opt.warmup = std::atoi(next());
    else if (arg == "--iterations") opt.iterations = std::atoi(next());
    else if (arg == "--split-k") opt.split_k = std::atoi(next());
    else if (arg == "--verify") opt.verify = true;
    else if (arg == "--misalign-a") opt.misalign_a = true;
    else if (arg == "--omit-workspace") opt.omit_workspace = true;
    else if (arg == "--expect-rejection") opt.expect_rejection = true;
    else if (arg == "--stream-mode") opt.stream_mode = next();
    else if (arg == "--help") {
      std::puts("e04_05_cutlass_space --list-configs | --config ID --m M --n N --k K [--verify] [--split-k S] [--misalign-a] [--omit-workspace] [--stream-mode default|nondefault]");
      return 0;
    }
  }
  if (opt.m <= 0 || opt.n <= 0 || opt.k <= 0 || opt.iterations <= 0 ||
      opt.warmup < 0 || opt.split_k <= 0 ||
      (opt.stream_mode != "default" && opt.stream_mode != "nondefault" &&
       opt.stream_mode != "dual")) return 2;
  return run_config(opt);
}
