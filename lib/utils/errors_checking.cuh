#pragma once

#define CUDA_CHECK_ERROR(func)                                                 \
  do {                                                                         \
    cudaError_t err = func;                                                    \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)
  