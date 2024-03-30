#pragma once

#include <cuComplex.h>
#include <cufft.h>

#include <core/parameters.hpp>
#include <utils/device_smart_pointer.hpp>

__global__ static
void Conjugate_kernel(cuComplex *spectrum, unsigned int height, unsigned int width)
{
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height)
  {
    unsigned int idx = y * width + x;
    spectrum[idx].y = -spectrum[idx].y;
  }
}

static void
Conjugate(cuComplex *spectrum, unsigned int height, unsigned int width)
{
  dim3 threads_per_block = {32, 32};
  dim3 grid_size = {(width + 31) / 32, (height + 31) / 32};

  Conjugate_kernel<<<grid_size, threads_per_block>>>(spectrum, height, width);
}

__global__ static
void Multiplicate_kernel(cuComplex *a, cuComplex *b,
                        unsigned int height, unsigned int width)
{
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (y < height && x < width)
  {
    unsigned int idx = y * width + x;
    a[idx] = cuCmulf(a[idx], b[idx]);
  }
}

__global__ static
void SpectrumShift2D_kernel(float *spectrum,
                            float *shifted_spectrum,
                            unsigned int width,
                            unsigned int height)
{
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int window = blockIdx.z;

  unsigned int window_offset = window * width * height;

  if (x < width && y < height)
  {
    unsigned int x1 = (x + (width / 2)) % width;
    unsigned int y1 = (y + (height / 2)) % height;

    shifted_spectrum[window_offset + y1 * width + x1] = spectrum[window_offset + y * width + x];
    shifted_spectrum[window_offset + y * width + x] = spectrum[window_offset + y1 * width + x1];
  }
}

static void ElementwiseMultiplication(cuComplex *a,
                              cuComplex *b,
                              unsigned int height,
                              unsigned int width)
{
  dim3 threads_per_block = {32, 32};
  dim3 grid_size = {(width + 31) / 32, (height + 31) / 32};

  Multiplicate_kernel<<<grid_size, threads_per_block>>>(a, b, height, width);
}

static void ShiftSpectrum(float *spectrum,
                  float *shifted_spectrum,
                  unsigned int width,
                  unsigned int height,
                  unsigned int number_of_windows)
{
  dim3 threads_per_block(16, 16);
  dim3 num_blocks(width / 16, height / 16, number_of_windows);

  SpectrumShift2D_kernel<<<num_blocks, threads_per_block>>>(spectrum, shifted_spectrum, width, height);
}
