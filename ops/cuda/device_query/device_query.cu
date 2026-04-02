#include "cuda_check.cuh"

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>

int main() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  if (device_count <= 0) {
    std::cerr << "No CUDA device detected." << std::endl;
    return 1;
  }

  int driver_version = 0;
  int runtime_version = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

  std::cout << "CUDA device count: " << device_count << "\n";
  std::cout << "CUDA driver version: "
            << driver_version / 1000 << "."
            << (driver_version % 1000) / 10 << "\n";
  std::cout << "CUDA runtime version: "
            << runtime_version / 1000 << "."
            << (runtime_version % 1000) / 10 << "\n\n";

  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    CUDA_CHECK(cudaSetDevice(device));

    std::size_t free_memory = 0;
    std::size_t total_memory = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));

    std::cout << "Device " << device << "\n";
    std::cout << "  Name: " << prop.name << "\n";
    std::cout << "  Compute capability: "
              << prop.major << "." << prop.minor << "\n";
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
    std::cout << "  Global memory: "
              << std::fixed << std::setprecision(2)
              << static_cast<double>(total_memory) / (1024.0 * 1024.0 * 1024.0)
              << " GiB\n";
    std::cout << "  Currently free memory: "
              << static_cast<double>(free_memory) / (1024.0 * 1024.0 * 1024.0)
              << " GiB\n";
    std::cout << "  Shared memory per block: "
              << prop.sharedMemPerBlock / 1024.0 << " KiB\n";
    std::cout << "  Registers per block: " << prop.regsPerBlock << "\n";
    std::cout << "  Warp size: " << prop.warpSize << "\n";
    std::cout << "  Max threads per block: "
              << prop.maxThreadsPerBlock << "\n";
    std::cout << "  Max block dimensions: "
              << prop.maxThreadsDim[0] << " x "
              << prop.maxThreadsDim[1] << " x "
              << prop.maxThreadsDim[2] << "\n";
    std::cout << "  Memory bus width: "
              << prop.memoryBusWidth << " bits\n";
    std::cout << "  Integrated GPU: "
              << (prop.integrated ? "yes" : "no") << "\n";
    std::cout << "  Unified addressing: "
              << (prop.unifiedAddressing ? "yes" : "no") << "\n";
    std::cout << "  Managed memory: "
              << (prop.managedMemory ? "yes" : "no") << "\n";
    std::cout << "  Concurrent kernels: "
              << (prop.concurrentKernels ? "yes" : "no") << "\n";
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
