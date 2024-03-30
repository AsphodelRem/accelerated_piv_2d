#pragma once

#include <tuple>

#include <cuComplex.h>
#include <cufft.h>

#include <core/parameters.hpp>
#include <utils/device_smart_pointer.hpp>

#include "math_operation.hpp"

template <typename T> class FFTHandler : public IOperation<T> {
public:
  explicit FFTHandler(const PIVParameters &parameters);

protected:
  int rank;
  int n[2];
  int i_dist;
  int o_dist;
  int in_embed[2];
  int on_embed[2];
  int stride;
  int batch_size;

  const PIVParameters &parameters_;
};

class ForwardFFTHandler : public FFTHandler<cuComplex> {
public:
  explicit ForwardFFTHandler(const PIVParameters &parameters);

  ForwardFFTHandler &operator*=(const ForwardFFTHandler &other);

  void ComputeForwardFFT(const SharedPtrGPU<float> &image,
                         bool to_conjugate = false);

private:
  cufftHandle cufft_handler_;
};

class BackwardFFTHandler : public FFTHandler<float> {
public:
  explicit BackwardFFTHandler(const PIVParameters &parameters);

  void ComputeBackwardFFT(const SharedPtrGPU<cuComplex> &image);

private:
  SharedPtrGPU<float> buffer_;
  cufftHandle cufft_handler_;
};