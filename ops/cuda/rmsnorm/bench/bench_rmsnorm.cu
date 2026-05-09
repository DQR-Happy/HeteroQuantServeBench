#include "hqsb/rmsnorm.h"
#include "rmsnorm_launchers.h"

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
using hqsb::RmsNormVariant;

struct Device {
  int device = 0;
  cudaDeviceProp prop{};
};

void report_device(const Device& d) {
  std::printf("# device=%s cc=%d.%d sms=%d\n", d.prop.name, d.prop.major,
              d.prop.minor, d.prop.multiProcessorCount);
}

double theoretical_bandwidth_gbps(const Device& d) {
  // memoryClockRate is kHz; memoryBusWidth is bits. DDR => factor 2.
  return 2.0 * d.prop.memoryClockRate * 1.0e3 * d.prop.memoryBusWidth / 8.0 /
         1.0e9;
}

size_t element_bytes(DType dtype) {
  return (dtype == DType::kFloat16) ? sizeof(__half) : sizeof(float);
}

std::vector<float> gen(size_t n) {
  std::vector<float> v(n);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> d(-1.0F, 1.0F);
  for (float& x : v) x = d(rng);
  return v;
}

std::vector<__half> to_half(const std::vector<float>& v) {
  std::vector<__half> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = __float2half_rn(v[i]);
  return o;
}

struct Case {
  DType dtype;
  int64_t rows;
  int64_t hidden;
  RmsNormVariant variant;
  int block_size;
};

const char* dtype_name(DType d) {
  return d == DType::kFloat16 ? "fp16" : "fp32";
}

void run_case(const Device& dev, const Case& c, int warmup, int iterations) {
  const size_t n = static_cast<size_t>(c.rows) * c.hidden;
  const size_t in_bytes = n * element_bytes(c.dtype);
  const size_t w_bytes = static_cast<size_t>(c.hidden) * element_bytes(c.dtype);

  auto h_in = gen(n);
  auto h_w = gen(c.hidden);

  void* d_in = nullptr;
  void* d_w = nullptr;
  void* d_out = nullptr;
  cudaMalloc(&d_in, in_bytes);
  cudaMalloc(&d_w, w_bytes);
  cudaMalloc(&d_out, in_bytes);

  if (c.dtype == DType::kFloat16) {
    auto hi = to_half(h_in), hw = to_half(h_w);
    cudaMemcpy(d_in, hi.data(), in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, hw.data(), w_bytes, cudaMemcpyHostToDevice);
  } else {
    cudaMemcpy(d_in, h_in.data(), in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), w_bytes, cudaMemcpyHostToDevice);
  }

  auto launch = [&](cudaStream_t stream) -> cudaError_t {
    switch (c.variant) {
      case RmsNormVariant::kV0Shared:
        return hqsb::rmsnorm_v0(static_cast<const float*>(d_in),
                                static_cast<const float*>(d_w),
                                static_cast<float*>(d_out), c.rows, c.hidden,
                                1e-5F, c.block_size, stream);
      case RmsNormVariant::kV1WarpShuffle:
        return hqsb::rmsnorm_v1(static_cast<const float*>(d_in),
                                static_cast<const float*>(d_w),
                                static_cast<float*>(d_out), c.rows, c.hidden,
                                1e-5F, c.block_size, stream);
      case RmsNormVariant::kV2Vectorized:
        return hqsb::rmsnorm_v2(d_in, d_w, d_out, c.rows, c.hidden, 1e-5F,
                                c.dtype, c.block_size, stream);
      default:
        return cudaErrorInvalidValue;
    }
  };

  for (int i = 0; i < warmup; ++i) {
    launch(0);
  }
  cudaDeviceSynchronize();
  if (cudaGetLastError() != cudaSuccess) {
    std::fprintf(stderr, "launch error for %s/%s\n",
                 hqsb::rmsnorm_variant_name(c.variant), dtype_name(c.dtype));
    cudaFree(d_in); cudaFree(d_w); cudaFree(d_out);
    return;
  }

  // Device-event timing (kernel-only, via events on the default stream).
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < iterations; ++i) {
    launch(0);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float device_ms = 0.0F;
  cudaEventElapsedTime(&device_ms, start, stop);
  device_ms /= static_cast<float>(iterations);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Host submit + sync timing (includes launch + sync overhead).
  auto host_start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    launch(0);
    cudaDeviceSynchronize();
  }
  auto host_end = std::chrono::steady_clock::now();
  const double host_ms =
      std::chrono::duration<double, std::milli>(host_end - host_start).count() /
      iterations;

  // Effective bandwidth: RMSNorm reads input twice (sum + normalize), reads
  // weight once, writes output once => (3*n + hidden) * elem bytes.
  const double traffic_bytes =
      static_cast<double>(3 * n + c.hidden) * element_bytes(c.dtype);
  const double device_bw = traffic_bytes / (device_ms / 1e3) / 1e9;
  const double host_bw = traffic_bytes / (host_ms / 1e3) / 1e9;

  int occupancy = 0;
  if (c.variant == RmsNormVariant::kV0Shared) {
    occupancy = hqsb::rmsnorm_v0_occupancy(c.block_size);
  } else if (c.variant == RmsNormVariant::kV1WarpShuffle) {
    occupancy = hqsb::rmsnorm_v1_occupancy(c.block_size);
  } else {
    occupancy = hqsb::rmsnorm_v2_occupancy(c.dtype, c.block_size);
  }

  std::printf("%s,%lld,%lld,%s,%d,%.4f,%.4f,%.2f,%.2f,%d\n",
              dtype_name(c.dtype), (long long)c.rows, (long long)c.hidden,
              hqsb::rmsnorm_variant_name(c.variant), c.block_size, device_ms,
              host_ms, device_bw, host_bw, occupancy);

  cudaFree(d_in);
  cudaFree(d_w);
  cudaFree(d_out);
}

}  // namespace

int main(int argc, char** argv) {
  Device dev;
  if (cudaGetDevice(&dev.device) != cudaSuccess) {
    std::fprintf(stderr, "no CUDA device\n");
    return EXIT_FAILURE;
  }
  cudaGetDeviceProperties(&dev.prop, dev.device);

  int warmup = 20;
  int iterations = 200;
  int64_t rows = 512;
  int64_t hidden = 2048;
  int block = 256;
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
    else if (a == "--help") {
      std::printf("usage: bench_rmsnorm [--rows N --hidden N --block N "
                  "--dtype fp32|fp16 --variant v0|v1|v2|all]\n");
      return EXIT_SUCCESS;
    }
  }

  report_device(dev);
  std::printf("# theoretical_bandwidth_gbps=%.2f\n",
              theoretical_bandwidth_gbps(dev));
  std::printf(
      "dtype,rows,hidden,variant,block_size,device_ms,host_ms,"
      "device_bw_gbps,host_bw_gbps,occupancy\n");

  DType dt = (dtype == "fp16") ? DType::kFloat16 : DType::kFloat32;
  std::vector<RmsNormVariant> variants;
  if (variant == "all") {
    if (dt == DType::kFloat16) {
      variants = {RmsNormVariant::kV2Vectorized};
    } else {
      variants = {RmsNormVariant::kV0Shared, RmsNormVariant::kV1WarpShuffle,
                  RmsNormVariant::kV2Vectorized};
    }
  } else if (variant == "v0") {
    variants = {RmsNormVariant::kV0Shared};
  } else if (variant == "v1") {
    variants = {RmsNormVariant::kV1WarpShuffle};
  } else if (variant == "v2") {
    variants = {RmsNormVariant::kV2Vectorized};
  } else {
    std::fprintf(stderr, "unknown --variant %s\n", variant.c_str());
    return EXIT_FAILURE;
  }

  for (auto v : variants) {
    run_case(dev, Case{dt, rows, hidden, v, block}, warmup, iterations);
  }
  return EXIT_SUCCESS;
}
