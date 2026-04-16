#include "cuda_check.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

struct Options {
  int rows = 512;
  int hidden = 1024;
  int warmup = 20;
  int iterations = 200;
  int threads = 256;
  float epsilon = 1.0e-5F;
};

int parse_positive_int(const char* value, const char* name) {
  const int parsed = std::stoi(value);
  if (parsed <= 0) {
    std::cerr << name << " must be positive." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return parsed;
}

Options parse_options(int argc, char** argv) {
  Options options;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];

    auto require_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << std::endl;
        std::exit(EXIT_FAILURE);
      }

      // TODO(cli): Validate that argv[i + 1] is not another flag (e.g., starting with '-')
      // to handle missing option values before flags and prevent downstream string argument pollution.
      return argv[++i];
    };

    if (argument == "--rows") {
      options.rows = parse_positive_int(require_value("--rows"), "--rows");
    } else if (argument == "--hidden") {
      options.hidden =
          parse_positive_int(require_value("--hidden"), "--hidden");
    } else if (argument == "--warmup") {
      options.warmup =
          parse_positive_int(require_value("--warmup"), "--warmup");
    } else if (argument == "--iterations") {
      options.iterations =
          parse_positive_int(require_value("--iterations"), "--iterations");
    } else if (argument == "--threads") {
      options.threads =
          parse_positive_int(require_value("--threads"), "--threads");
    } else if (argument == "--help") {
      std::cout
          << "Usage: hqsb_rmsnorm_baseline [options]\n"
          << "  --rows N\n"
          << "  --hidden N\n"
          << "  --warmup N\n"
          << "  --iterations N\n"
          << "  --threads N\n";
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << argument << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  if ((options.threads & (options.threads - 1)) != 0 ||
      options.threads > 1024) {
    std::cerr << "--threads must be a power of two and <= 1024."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return options;
}

void rmsnorm_cpu(const std::vector<float>& input,
                 const std::vector<float>& weight,
                 std::vector<float>& output,
                 int rows,
                 int hidden,
                 float epsilon) {
  for (int row = 0; row < rows; ++row) {
    const std::size_t offset =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(hidden);

    double square_sum = 0.0;
    for (int column = 0; column < hidden; ++column) {
      const double value = input[offset + column];
      square_sum += value * value;
    }

    const float inverse_rms =
        1.0F / std::sqrt(static_cast<float>(square_sum / hidden) + epsilon);

    for (int column = 0; column < hidden; ++column) {
      output[offset + column] =
          input[offset + column] * inverse_rms * weight[column];
    }
  }
}

__global__ void rmsnorm_cuda_kernel(const float* input,
                                    const float* weight,
                                    float* output,
                                    int hidden,
                                    float epsilon) {
  extern __shared__ float shared_square_sum[];

  const int row = static_cast<int>(blockIdx.x);
  const int thread = static_cast<int>(threadIdx.x);
  const std::size_t row_offset =
      static_cast<std::size_t>(row) * static_cast<std::size_t>(hidden);

  float local_square_sum = 0.0F;

  for (int column = thread; column < hidden; column += blockDim.x) {
    const float value = input[row_offset + column];
    local_square_sum += value * value;
  }

  shared_square_sum[thread] = local_square_sum;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread < static_cast<int>(stride)) {
      shared_square_sum[thread] +=
          shared_square_sum[thread + static_cast<int>(stride)];
    }
    __syncthreads();
  }

  const float inverse_rms =
      rsqrtf(shared_square_sum[0] / static_cast<float>(hidden) + epsilon);

  for (int column = thread; column < hidden; column += blockDim.x) {
    output[row_offset + column] =
        input[row_offset + column] * inverse_rms * weight[column];
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);

  int current_device = 0;
  CUDA_CHECK(cudaGetDevice(&current_device));

  cudaDeviceProp device_properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&device_properties, current_device));

  const std::size_t element_count =
      static_cast<std::size_t>(options.rows) *
      static_cast<std::size_t>(options.hidden);

  std::vector<float> host_input(element_count);
  std::vector<float> host_weight(options.hidden);
  std::vector<float> host_reference(element_count);
  std::vector<float> host_output(element_count);

  std::mt19937 random_generator(42);
  std::uniform_real_distribution<float> input_distribution(-1.0F, 1.0F);
  std::uniform_real_distribution<float> weight_distribution(0.5F, 1.5F);

  for (float& value : host_input) {
    value = input_distribution(random_generator);
  }

  for (float& value : host_weight) {
    value = weight_distribution(random_generator);
  }

  rmsnorm_cpu(host_input,
              host_weight,
              host_reference,
              options.rows,
              options.hidden,
              options.epsilon);

  float* device_input = nullptr;
  float* device_weight = nullptr;
  float* device_output = nullptr;

  CUDA_CHECK(cudaMalloc(&device_input, element_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_weight, options.hidden * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, element_count * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(device_input,
                        host_input.data(),
                        element_count * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy(device_weight,
                        host_weight.data(),
                        options.hidden * sizeof(float),
                        cudaMemcpyHostToDevice));

  const dim3 grid(static_cast<unsigned int>(options.rows));
  const dim3 block(static_cast<unsigned int>(options.threads));
  const std::size_t shared_memory_bytes =
      static_cast<std::size_t>(options.threads) * sizeof(float);

  for (int iteration = 0; iteration < options.warmup; ++iteration) {
    rmsnorm_cuda_kernel<<<grid, block, shared_memory_bytes>>>(
        device_input,
        device_weight,
        device_output,
        options.hidden,
        options.epsilon);
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start_event{};
  cudaEvent_t stop_event{};

  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  CUDA_CHECK(cudaEventRecord(start_event));

  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    rmsnorm_cuda_kernel<<<grid, block, shared_memory_bytes>>>(
        device_input,
        device_weight,
        device_output,
        options.hidden,
        options.epsilon);
  }

  CUDA_CHECK(cudaEventRecord(stop_event));
  CUDA_CHECK(cudaEventSynchronize(stop_event));
  CUDA_CHECK(cudaGetLastError());

  float total_milliseconds = 0.0F;
  CUDA_CHECK(
      cudaEventElapsedTime(&total_milliseconds, start_event, stop_event));

  CUDA_CHECK(cudaMemcpy(host_output.data(),
                        device_output,
                        element_count * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float maximum_absolute_error = 0.0F;
  double mean_absolute_error = 0.0;

  for (std::size_t index = 0; index < element_count; ++index) {
    const float error =
        std::abs(host_output[index] - host_reference[index]);

    maximum_absolute_error =
        std::max(maximum_absolute_error, error);
    mean_absolute_error += static_cast<double>(error);
  }

  mean_absolute_error /= static_cast<double>(element_count);

  const double average_milliseconds =
      static_cast<double>(total_milliseconds) /
      static_cast<double>(options.iterations);

  const double bytes_per_iteration =
      static_cast<double>(element_count) *
      static_cast<double>(sizeof(float)) *
      3.0;

  const double effective_bandwidth_gb_per_second =
      bytes_per_iteration /
      (average_milliseconds / 1000.0) /
      1.0e9;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "device=" << device_properties.name << "\n";
  std::cout << "compute_capability="
            << device_properties.major << "."
            << device_properties.minor << "\n";
  std::cout << "rows=" << options.rows << "\n";
  std::cout << "hidden=" << options.hidden << "\n";
  std::cout << "threads=" << options.threads << "\n";
  std::cout << "warmup=" << options.warmup << "\n";
  std::cout << "iterations=" << options.iterations << "\n";
  std::cout << "max_abs_error=" << maximum_absolute_error << "\n";
  std::cout << "mean_abs_error=" << mean_absolute_error << "\n";
  std::cout << "average_kernel_ms=" << average_milliseconds << "\n";
  std::cout << "effective_bandwidth_gbps="
            << effective_bandwidth_gb_per_second << "\n";

  const bool correctness_passed = maximum_absolute_error <= 5.0e-4F;
  std::cout << "correctness="
            << (correctness_passed ? "PASS" : "FAIL") << "\n";

  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));

  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_weight));
  CUDA_CHECK(cudaFree(device_output));

  return correctness_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
