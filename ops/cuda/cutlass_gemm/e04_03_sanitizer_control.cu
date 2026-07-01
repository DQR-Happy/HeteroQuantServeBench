#include <cuda_runtime.h>

#include <cstdio>
#include <string>

__global__ void guarded_write(float* out, int count, bool inject_oob) {
  const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < count) out[index] = float(index);
  if (inject_oob && index == 0) out[count] = 1.0f;
}

int main(int argc, char** argv) {
  const bool inject_oob = argc == 3 && std::string(argv[1]) == "--mode" &&
                          std::string(argv[2]) == "oob";
  float* out = nullptr;
  constexpr int count = 32;
  if (cudaMalloc(&out, count * sizeof(float)) != cudaSuccess) return 2;
  guarded_write<<<1, 32>>>(out, count, inject_oob);
  const cudaError_t completion = cudaDeviceSynchronize();
  std::printf("mode=%s completion=%s\n", inject_oob ? "oob" : "clean",
              cudaGetErrorString(completion));
  cudaFree(out);
  return completion == cudaSuccess ? 0 : 3;
}
