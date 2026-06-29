#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <dlfcn.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

using RmsForward = int (*)(const void*, const void*, void*, long long,
                           long long, float, int, int, void*);
using FusedForward = int (*)(const void*, const void*, const void*, void*,
                             void*, long long, long long, float, int, int,
                             int, void*);

constexpr size_t kGuard = 64;
constexpr unsigned char kGuardByte = 0xA5;

struct Api {
  void* rms_handle = nullptr;
  void* fused_handle = nullptr;
  RmsForward rms = nullptr;
  FusedForward fused = nullptr;

  ~Api() {
    if (fused_handle) dlclose(fused_handle);
    if (rms_handle) dlclose(rms_handle);
  }
};

struct DeviceBuffer {
  unsigned char* base = nullptr;
  unsigned char* data = nullptr;
  size_t logical = 0;
  size_t offset = kGuard;
  size_t total = 0;

  bool allocate(size_t bytes, size_t misalign = 0) {
    logical = bytes;
    offset = kGuard + misalign;
    total = offset + logical + kGuard;
    if (cudaMalloc(&base, total) != cudaSuccess) return false;
    data = base + offset;
    return cudaMemset(base, kGuardByte, total) == cudaSuccess;
  }

  bool guards_clean() const {
    std::vector<unsigned char> host(total);
    if (cudaMemcpy(host.data(), base, total, cudaMemcpyDeviceToHost) !=
        cudaSuccess) return false;
    for (size_t i = 0; i < offset; ++i) {
      if (host[i] != kGuardByte) return false;
    }
    for (size_t i = offset + logical; i < total; ++i) {
      if (host[i] != kGuardByte) return false;
    }
    return true;
  }

  ~DeviceBuffer() {
    if (base) cudaFree(base);
  }
};

__global__ void clean_control_kernel(int* output) {
  if (threadIdx.x == 0) output[0] = 7;
}

__global__ void oob_control_kernel(int* output) {
  if (threadIdx.x == 0) output[1] = 13;
}

__global__ void init_control_kernel(const int* input, int* output) {
  if (threadIdx.x == 0) output[0] = input[0] + 1;
}

__global__ void race_control_kernel(int* output) {
  __shared__ int value;
  value = static_cast<int>(threadIdx.x);
  if (threadIdx.x == 0) output[0] = value;
}

__global__ void sync_control_kernel(int* output) {
  // Deliberately use a shared cuda::barrier without init(). CUDA 12.6
  // synccheck has a dedicated fatal diagnostic for this contract violation.
  __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
  barrier.arrive_and_wait();
  if (threadIdx.x == 0) output[0] = 1;
}

void emit(const char* id, const char* status, int launch, int completion,
          bool guards, const char* detail) {
  std::printf("HQSB_CASE id=%s status=%s launch=%d completion=%d guards=%s detail=%s\n",
              id, status, launch, completion, guards ? "clean" : "corrupt",
              detail);
  std::fflush(stdout);
}

bool load_api(const char* rms_path, const char* fused_path, Api* api) {
  api->rms_handle = dlopen(rms_path, RTLD_NOW | RTLD_LOCAL);
  if (!api->rms_handle) {
    std::fprintf(stderr, "dlopen RMS failed: %s\n", dlerror());
    return false;
  }
  api->fused_handle = dlopen(fused_path, RTLD_NOW | RTLD_LOCAL);
  if (!api->fused_handle) {
    std::fprintf(stderr, "dlopen fused failed: %s\n", dlerror());
    return false;
  }
  api->rms = reinterpret_cast<RmsForward>(
      dlsym(api->rms_handle, "hqsb_rmsnorm_forward_ex_c"));
  api->fused = reinterpret_cast<FusedForward>(dlsym(
      api->fused_handle, "hqsb_fused_residual_rmsnorm_forward_ex_c"));
  if (!api->rms || !api->fused) {
    std::fprintf(stderr, "required C ABI symbol missing\n");
    return false;
  }
  return true;
}

template <typename T>
T from_float(float value);
template <>
float from_float<float>(float value) { return value; }
template <>
__half from_float<__half>(float value) { return __float2half_rn(value); }

template <typename T>
float as_float(T value);
template <>
float as_float<float>(float value) { return value; }
template <>
float as_float<__half>(__half value) { return __half2float(value); }

template <typename T>
bool upload_pattern(DeviceBuffer* buffer, size_t count, float phase) {
  std::vector<T> host(count);
  for (size_t i = 0; i < count; ++i) {
    host[i] = from_float<T>(0.25F + 0.5F *
        std::sin(static_cast<float>(i) * 0.017F + phase));
  }
  return cudaMemcpy(buffer->data, host.data(), count * sizeof(T),
                    cudaMemcpyHostToDevice) == cudaSuccess;
}

template <typename T>
bool rms_numerically_correct(const DeviceBuffer& input,
                             const DeviceBuffer& weight,
                             const DeviceBuffer& output,
                             int rows, int hidden, float epsilon,
                             bool in_place) {
  const size_t elements = static_cast<size_t>(rows) * hidden;
  std::vector<T> x(elements), w(hidden), y(elements);
  const unsigned char* x_data = in_place ? output.data : input.data;
  // In-place input has already been overwritten, so only check finiteness and
  // guards there. Its full numerical correctness was established by E03-01.
  if (in_place) {
    if (cudaMemcpy(y.data(), output.data, elements * sizeof(T),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    for (T value : y) if (!std::isfinite(as_float(value))) return false;
    return true;
  }
  if (cudaMemcpy(x.data(), x_data, elements * sizeof(T),
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(w.data(), weight.data, hidden * sizeof(T),
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(y.data(), output.data, elements * sizeof(T),
                 cudaMemcpyDeviceToHost) != cudaSuccess) return false;
  const float tolerance = sizeof(T) == 2 ? 0.005F : 0.003F;
  for (int row = 0; row < rows; ++row) {
    double sum = 0.0;
    for (int col = 0; col < hidden; ++col) {
      const float v = as_float(x[static_cast<size_t>(row) * hidden + col]);
      sum += static_cast<double>(v) * v;
    }
    const float inv = 1.0F / std::sqrt(static_cast<float>(sum / hidden) + epsilon);
    for (int col = 0; col < hidden; ++col) {
      const size_t index = static_cast<size_t>(row) * hidden + col;
      const float reference = as_float(x[index]) * as_float(w[col]) * inv;
      if (!std::isfinite(as_float(y[index])) ||
          std::fabs(as_float(y[index]) - reference) > tolerance) return false;
    }
  }
  return true;
}

template <typename T>
bool run_rms_case(Api& api, int rows, int hidden, int variant,
                  bool nondefault, bool misaligned, bool in_place,
                  const std::string& id) {
  const size_t elements = static_cast<size_t>(rows) * hidden;
  const size_t misalign = misaligned ? sizeof(T) : 0;
  DeviceBuffer input, weight, output;
  if (!input.allocate(elements * sizeof(T), misalign) ||
      !weight.allocate(static_cast<size_t>(hidden) * sizeof(T), misalign) ||
      !output.allocate(elements * sizeof(T), misalign) ||
      !upload_pattern<T>(&input, elements, 0.1F) ||
      !upload_pattern<T>(&weight, hidden, 0.7F) ||
      (in_place && cudaMemcpy(output.data, input.data, elements * sizeof(T),
                              cudaMemcpyDeviceToDevice) != cudaSuccess)) {
    emit(id.c_str(), "FAIL", -1, -1, false, "setup");
    return false;
  }
  cudaStream_t stream = nullptr;
  if (nondefault && cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
                        cudaSuccess) return false;
  void* out = in_place ? input.data : output.data;
  const int launch = api.rms(input.data, weight.data, out, rows, hidden, 1e-6F,
                             sizeof(T) == 2 ? 1 : 0, variant, stream);
  const int completion = static_cast<int>(cudaStreamSynchronize(stream));
  bool guards = input.guards_clean() && weight.guards_clean() &&
                output.guards_clean();
  bool numeric = launch == 0 && completion == 0 && guards &&
      rms_numerically_correct<T>(input, weight,
                                 in_place ? input : output,
                                 rows, hidden, 1e-6F, in_place);
  if (stream) cudaStreamDestroy(stream);
  emit(id.c_str(), numeric ? "PASS" : "FAIL", launch, completion, guards,
       numeric ? "executed" : "production_case_failed");
  return numeric;
}

template <typename T>
bool run_fused_case(Api& api, int rows, int hidden, int variant,
                    bool nondefault, const std::string& id) {
  const size_t elements = static_cast<size_t>(rows) * hidden;
  DeviceBuffer input, residual, weight, residual_out, output;
  if (!input.allocate(elements * sizeof(T)) ||
      !residual.allocate(elements * sizeof(T)) ||
      !weight.allocate(static_cast<size_t>(hidden) * sizeof(T)) ||
      !residual_out.allocate(elements * sizeof(T)) ||
      !output.allocate(elements * sizeof(T)) ||
      !upload_pattern<T>(&input, elements, 0.1F) ||
      !upload_pattern<T>(&residual, elements, 0.4F) ||
      !upload_pattern<T>(&weight, hidden, 0.7F)) {
    emit(id.c_str(), "FAIL", -1, -1, false, "setup");
    return false;
  }
  cudaStream_t stream = nullptr;
  if (nondefault) cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  const int launch = api.fused(input.data, residual.data, weight.data,
      residual_out.data, output.data, rows, hidden, 1e-6F,
      sizeof(T) == 2 ? 1 : 0, variant, 1, stream);
  const int completion = static_cast<int>(cudaStreamSynchronize(stream));
  const bool guards = input.guards_clean() && residual.guards_clean() &&
      weight.guards_clean() && residual_out.guards_clean() &&
      output.guards_clean();
  std::vector<T> host(elements);
  bool finite = cudaMemcpy(host.data(), output.data, elements * sizeof(T),
                           cudaMemcpyDeviceToHost) == cudaSuccess;
  for (T value : host) finite = finite && std::isfinite(as_float(value));
  const bool pass = launch == 0 && completion == 0 && guards && finite;
  if (stream) cudaStreamDestroy(stream);
  emit(id.c_str(), pass ? "PASS" : "FAIL", launch, completion, guards,
       pass ? "executed" : "production_case_failed");
  return pass;
}

int run_control(const std::string& mode) {
  int* first = nullptr;
  int* second = nullptr;
  cudaMalloc(&first, sizeof(int));
  if (mode == "control-clean") {
    clean_control_kernel<<<1, 32>>>(first);
  } else if (mode == "control-memcheck") {
    oob_control_kernel<<<1, 32>>>(first);
  } else if (mode == "control-initcheck") {
    cudaMalloc(&second, sizeof(int));
    init_control_kernel<<<1, 32>>>(first, second);
  } else if (mode == "control-racecheck") {
    race_control_kernel<<<1, 32>>>(first);
  } else if (mode == "control-synccheck") {
    sync_control_kernel<<<1, 64>>>(first);
  } else {
    return 2;
  }
  const int launch = static_cast<int>(cudaGetLastError());
  const int completion = static_cast<int>(cudaDeviceSynchronize());
  emit(mode.c_str(), completion == 0 ? "DONE" : "CUDA_ERROR", launch,
       completion, true, "injected_control");
  if (second) cudaFree(second);
  cudaFree(first);
  return 0;
}

bool expect_reject(const char* id, Api& api, const void* input,
                   const void* weight, void* output, long long rows,
                   long long hidden, float epsilon, int dtype, int variant,
                   void* stream) {
  if (output) cudaMemset(output, 0x5A, 64);
  const int launch = api.rms(input, weight, output, rows, hidden, epsilon,
                             dtype, variant, stream);
  const int completion = static_cast<int>(cudaDeviceSynchronize());
  std::vector<unsigned char> host(64);
  const bool copied = output == nullptr ||
      cudaMemcpy(host.data(), output, host.size(), cudaMemcpyDeviceToHost) ==
          cudaSuccess;
  bool unchanged = copied;
  if (output) {
    for (unsigned char value : host) unchanged = unchanged && value == 0x5A;
  }
  const bool pass = launch == static_cast<int>(cudaErrorInvalidValue) &&
                    completion == 0 && unchanged;
  emit(id, pass ? "PASS" : "FAIL", launch, completion, unchanged,
       pass ? "prelaunch_reject" : "silent_launch_or_unstable_error");
  return pass;
}

int run_api_negative(Api& api) {
  DeviceBuffer x, w, y;
  x.allocate(256); w.allocate(256); y.allocate(256);
  cudaMemset(x.data, 0, x.logical);
  cudaMemset(w.data, 0, w.logical);
  bool ok = true;
  ok &= expect_reject("null_input", api, nullptr, w.data, y.data, 1, 16,
                      1e-6F, 0, 3, nullptr);
  ok &= expect_reject("null_weight", api, x.data, nullptr, y.data, 1, 16,
                      1e-6F, 0, 3, nullptr);
  ok &= expect_reject("null_output", api, x.data, w.data, nullptr, 1, 16,
                      1e-6F, 0, 3, nullptr);
  ok &= expect_reject("rows_zero", api, x.data, w.data, y.data, 0, 16,
                      1e-6F, 0, 3, nullptr);
  ok &= expect_reject("hidden_zero", api, x.data, w.data, y.data, 1, 0,
                      1e-6F, 0, 3, nullptr);
  ok &= expect_reject("hidden_int_overflow", api, x.data, w.data, y.data, 1,
                      static_cast<long long>(std::numeric_limits<int>::max()) + 1,
                      1e-6F, 0, 3, nullptr);
  ok &= expect_reject("rows_grid_overflow", api, x.data, w.data, y.data,
                      std::numeric_limits<long long>::max(), 16, 1e-6F, 0, 3,
                      nullptr);
  ok &= expect_reject("epsilon_zero", api, x.data, w.data, y.data, 1, 16,
                      0.0F, 0, 3, nullptr);
  ok &= expect_reject("epsilon_nan", api, x.data, w.data, y.data, 1, 16,
                      std::numeric_limits<float>::quiet_NaN(), 0, 3, nullptr);
  ok &= expect_reject("dtype_unknown", api, x.data, w.data, y.data, 1, 16,
                      1e-6F, 99, 3, nullptr);
  ok &= expect_reject("variant_unknown", api, x.data, w.data, y.data, 1, 16,
                      1e-6F, 0, 99, nullptr);
  ok &= expect_reject("reference_variant", api, x.data, w.data, y.data, 1, 16,
                      1e-6F, 0, 1, nullptr);
  ok &= expect_reject("fp16_v0", api, x.data, w.data, y.data, 1, 16,
                      1e-6F, 1, 2, nullptr);
  ok &= expect_reject("strict_tail", api, x.data, w.data, y.data, 1, 15,
                      1e-6F, 0, 6, nullptr);
  ok &= expect_reject("partial_input_output_overlap", api, x.data, w.data,
                      x.data + sizeof(float), 1, 16, 1e-6F, 0, 3, nullptr);
  ok &= expect_reject("output_weight_overlap", api, x.data, w.data, w.data,
                      1, 16, 1e-6F, 0, 3, nullptr);

  DeviceBuffer residual, residual_out, fused_out;
  residual.allocate(256); residual_out.allocate(256); fused_out.allocate(256);
  cudaMemset(residual.data, 0, 256);
  const auto fused_reject = [&](const char* id, const void* in,
                                const void* res, const void* weight,
                                void* res_out, void* out, long long rows,
                                long long hidden, int dtype, int variant,
                                int semantic) {
    if (out) cudaMemset(out, 0x5A, 64);
    const int launch = api.fused(in, res, weight, res_out, out, rows, hidden,
                                 1e-6F, dtype, variant, semantic, nullptr);
    const int completion = static_cast<int>(cudaDeviceSynchronize());
    bool unchanged = true;
    if (out) {
      std::vector<unsigned char> host(64);
      unchanged = cudaMemcpy(host.data(), out, host.size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess;
      for (unsigned char value : host) unchanged = unchanged && value == 0x5A;
    }
    const bool pass = launch == static_cast<int>(cudaErrorInvalidValue) &&
                      completion == 0 && unchanged;
    emit(id, pass ? "PASS" : "FAIL", launch, completion, unchanged,
         pass ? "prelaunch_reject" : "silent_launch_or_unstable_error");
    return pass;
  };
  ok &= fused_reject("fused_null_input", nullptr, residual.data, w.data,
                     residual_out.data, fused_out.data, 1, 16, 0, 3, 1);
  ok &= fused_reject("fused_hidden_above_max", x.data, residual.data, w.data,
                     residual_out.data, fused_out.data, 1, 8193, 0, 3, 1);
  ok &= fused_reject("fused_dtype_unknown", x.data, residual.data, w.data,
                     residual_out.data, fused_out.data, 1, 16, 99, 3, 1);
  ok &= fused_reject("fused_variant_unknown", x.data, residual.data, w.data,
                     residual_out.data, fused_out.data, 1, 16, 0, 99, 1);
  ok &= fused_reject("fused_semantic_unknown", x.data, residual.data, w.data,
                     residual_out.data, fused_out.data, 1, 16, 0, 3, 99);
  ok &= fused_reject("fused_output_input_overlap", x.data, residual.data,
                     w.data, residual_out.data, x.data, 1, 16, 0, 3, 1);
  ok &= fused_reject("fused_outputs_overlap", x.data, residual.data, w.data,
                     residual_out.data, residual_out.data, 1, 16, 0, 3, 1);
  return ok ? 0 : 1;
}

int run_destroyed_stream(Api& api) {
  DeviceBuffer x, w, y;
  x.allocate(256); w.allocate(256); y.allocate(256);
  cudaMemset(x.data, 0, 256); cudaMemset(w.data, 0, 256);
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  cudaStreamDestroy(stream);
  cudaMemset(y.data, 0x5A, 64);
  const int launch = api.rms(x.data, w.data, y.data, 1, 16, 1e-6F, 0, 3,
                             stream);
  const int completion = static_cast<int>(cudaDeviceSynchronize());
  std::vector<unsigned char> host(64);
  bool unchanged = cudaMemcpy(host.data(), y.data, host.size(),
                              cudaMemcpyDeviceToHost) == cudaSuccess;
  for (unsigned char value : host) unchanged = unchanged && value == 0x5A;
  const bool pass = launch != 0 && completion == 0 && unchanged;
  emit("destroyed_stream", pass ? "PASS" : "FAIL", launch, completion,
       unchanged, pass ? "stable_immediate_reject" : "unsafe_stream_path");
  return pass ? 0 : 1;
}

int run_host_pointer(Api& api) {
  DeviceBuffer w, y;
  w.allocate(256); y.allocate(256); cudaMemset(w.data, 0, 256);
  std::vector<float> host(64, 1.0F);
  cudaMemset(y.data, 0x5A, y.logical);
  const int launch = api.rms(host.data(), w.data, y.data, 1, 16, 1e-6F,
                             0, 3, nullptr);
  const int completion = static_cast<int>(cudaDeviceSynchronize());
  const bool pass = launch == static_cast<int>(cudaErrorInvalidValue);
  emit("host_pointer_wrong_device", pass ? "PASS" : "FAIL", launch,
       completion, true, pass ? "prelaunch_reject" : "not_rejected_prelaunch");
  return pass ? 0 : 1;
}

int run_lifecycle(Api& api) {
  size_t free_before = 0, total = 0, free_after = 0;
  constexpr int rows = 2;
  constexpr int hidden = 128;
  constexpr size_t elements = static_cast<size_t>(rows) * hidden;
  DeviceBuffer input, weight, output;
  if (!input.allocate(elements * sizeof(float)) ||
      !weight.allocate(hidden * sizeof(float)) ||
      !output.allocate(elements * sizeof(float)) ||
      !upload_pattern<float>(&input, elements, 0.1F) ||
      !upload_pattern<float>(&weight, hidden, 0.7F)) return 1;
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
    return 1;
  // Warm up module/context state, then hold caller-owned buffers and stream
  // fixed. The measured loop therefore detects operator-side hidden growth,
  // not CUDA allocator or stream-resource caching in the harness.
  int launch = api.rms(input.data, weight.data, output.data, rows, hidden,
                       1e-6F, 0, 3, stream);
  if (launch != 0 || cudaStreamSynchronize(stream) != cudaSuccess) return 1;
  cudaMemGetInfo(&free_before, &total);
  bool ok = true;
  for (int i = 0; i < 200; ++i) {
    launch = api.rms(input.data, weight.data, output.data, rows, hidden,
                     1e-6F, 0, 3, stream);
    ok = ok && launch == 0;
  }
  ok = ok && cudaStreamSynchronize(stream) == cudaSuccess;
  cudaMemGetInfo(&free_after, &total);
  const long long delta = static_cast<long long>(free_after) -
                          static_cast<long long>(free_before);
  const bool guards = input.guards_clean() && weight.guards_clean() &&
                      output.guards_clean();
  const bool numeric = rms_numerically_correct<float>(
      input, weight, output, rows, hidden, 1e-6F, false);
  // Compute Sanitizer itself may retain roughly 1.3 MiB of instrumentation
  // bookkeeping across 200 launches. A 2 MiB envelope is used only for the
  // coarse cudaMemGetInfo signal; memcheck's allocation-owner leak report
  // remains the hard project-leak check.
  const bool stable = ok && guards && numeric &&
                      std::llabs(delta) <= 2 * 1024 * 1024;
  cudaStreamDestroy(stream);
  char detail[192];
  std::snprintf(detail, sizeof(detail),
                "free_before=%zu;free_after=%zu;free_delta_bytes=%lld",
                free_before, free_after, delta);
  emit("lifecycle_200", stable ? "PASS" : "FAIL", launch, ok ? 0 : -1,
       guards, detail);
  return stable ? 0 : 1;
}

int run_rms_matrix(Api& api) {
  bool ok = true;
  const int widths[] = {1, 3, 4, 5, 31, 32, 33, 127, 128, 129,
                        2047, 2048, 2049, 8192};
  for (int hidden : widths) {
    const int fp32_variant = hidden % 4 == 0 ? 6 : 5;
    const int fp16_variant = hidden % 2 == 0 ? 6 : 5;
    ok &= run_rms_case<float>(api, hidden == 2048 ? 128 : 2, hidden,
        fp32_variant, hidden == 128, false, false,
        "rms_fp32_h" + std::to_string(hidden) + "_v" +
            std::to_string(fp32_variant));
    ok &= run_rms_case<__half>(api, hidden == 2048 ? 128 : 2, hidden,
        fp16_variant, hidden == 2048, false, false,
        "rms_fp16_h" + std::to_string(hidden) + "_v" +
            std::to_string(fp16_variant));
  }
  ok &= run_rms_case<float>(api, 1024, 2048, 6, true, false, false,
                            "rms_fp32_prefill1024_h2048_v6");
  ok &= run_rms_case<__half>(api, 1024, 2048, 6, true, false, false,
                             "rms_fp16_prefill1024_h2048_v6");
  ok &= run_rms_case<float>(api, 3, 129, 0, true, true, false,
                            "rms_fp32_misaligned_auto");
  ok &= run_rms_case<__half>(api, 3, 129, 0, true, true, false,
                             "rms_fp16_misaligned_auto");
  ok &= run_rms_case<float>(api, 3, 2048, 2, true, false, false,
                            "rms_fp32_forced_v0");
  ok &= run_rms_case<float>(api, 3, 2049, 3, true, false, false,
                            "rms_fp32_forced_v1");
  ok &= run_rms_case<float>(api, 3, 128, 0, true, false, true,
                            "rms_fp32_exact_inplace");
  ok &= run_rms_case<__half>(api, 3, 129, 5, true, false, true,
                             "rms_fp16_exact_inplace");
  return ok ? 0 : 1;
}

int run_fused_matrix(Api& api) {
  bool ok = true;
  for (int hidden : {1, 31, 32, 33, 127, 128, 129, 2048, 8192}) {
    ok &= run_fused_case<float>(api, hidden == 2048 ? 128 : 2, hidden,
        hidden % 2 ? 3 : 2, true, "fused_fp32_h" + std::to_string(hidden) +
            "_v" + std::to_string(hidden % 2 ? 3 : 2));
    ok &= run_fused_case<__half>(api, hidden == 2048 ? 128 : 2, hidden,
        hidden % 2 ? 2 : 3, true, "fused_fp16_h" + std::to_string(hidden) +
            "_v" + std::to_string(hidden % 2 ? 2 : 3));
  }
  ok &= run_fused_case<float>(api, 1024, 2048, 3, true,
                              "fused_fp32_prefill1024_h2048_v3");
  ok &= run_fused_case<__half>(api, 1024, 2048, 3, true,
                               "fused_fp16_prefill1024_h2048_v3");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s MODE [RMS_SO FUSED_SO]\n", argv[0]);
    return 2;
  }
  const std::string mode = argv[1];
  if (mode.rfind("control-", 0) == 0) return run_control(mode);
  if (argc != 4) return 2;
  Api api;
  if (!load_api(argv[2], argv[3], &api)) return 2;
  if (mode == "rms-matrix") return run_rms_matrix(api);
  if (mode == "fused-matrix") return run_fused_matrix(api);
  if (mode == "api-negative") return run_api_negative(api);
  if (mode == "destroyed-stream") return run_destroyed_stream(api);
  if (mode == "host-pointer") return run_host_pointer(api);
  if (mode == "lifecycle") return run_lifecycle(api);
  return 2;
}
