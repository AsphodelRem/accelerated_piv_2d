#pragma once

#define CUDA_CHECK_ERROR(result)                                               \
  do {                                                                         \
    cudaError_t err = static_cast<cudaError_t>(result);                        \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)
  
  