#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t error = (call);                                           \
    if (error != cudaSuccess) {                                                 \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "      \
                << cudaGetErrorString(error) << " (" << static_cast<int>(error) \
                << ")" << std::endl;                                            \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                          \
  } while (0)
