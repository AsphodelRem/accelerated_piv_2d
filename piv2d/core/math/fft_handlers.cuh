#pragma once

#include <cuComplex.h>
#include <cufft.h>

#include <parameters.cuh>
#include <utils/device_smart_pointer.hpp>

class FFTHandler
{
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

class ForwardFFTHandler : public FFTHandler
{
public:
    explicit ForwardFFTHandler(const PIVParameters &parameters);

    ForwardFFTHandler &operator*=(const ForwardFFTHandler &other);

    void ComputeForwardFFT(const SharedPtrGPU<float> &image, bool to_conjugate = false);

    SharedPtrGPU<cufftComplex> result;

private:
    cufftHandle cufft_handler_;
};

class BackwardFFTHandler : public FFTHandler
{
public:
    explicit BackwardFFTHandler(const PIVParameters &parameters);

    void ComputeBackwardFFT(const SharedPtrGPU<cuComplex> &image);

    SharedPtrGPU<float> result;

private:
    SharedPtrGPU<float> buffer_;
    cufftHandle cufft_handler_;
};