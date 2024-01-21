#pragma once

#include <memory>

#include <cuComplex.h>
#include <cufft.h>

#include <cub/util_type.cuh>
#include <cub/device/device_segmented_reduce.cuh>

#include <parameters.cuh>
#include <device_smart_pointer.hpp>
#include <errors_checking.cuh>

template <typename T>
struct Point2D
{
    T x, y;
};

// template <typename T>
// struct MovementDescription
// {
//     int centerX, centerY;
//     T u, v;
//     T correlation;
// };

class MultiArgMaxSearch
{
public:
    MultiArgMaxSearch(PIVParameters &parameters);
    ~MultiArgMaxSearch() = default;

    void GetMaxForAllWindows(SharedPtrGPU<float> &input);

    SharedPtrGPU<cub::KeyValuePair<int, float>> result;

private:
    std::shared_ptr<int[]> offsets_;
    SharedPtrGPU<int> dev_cub_offsets_;
    SharedPtrGPU<char> buffer_;

    unsigned int number_of_windows_;
};

class FFTHandler
{
public:
    FFTHandler(PIVParameters &parameters) : parameters_(parameters)
    {
        const int segment_size = parameters.image_parameters.window_size;

        rank = 2;
        n[0] = n[1] = segment_size;
        i_dist = segment_size * segment_size;
        o_dist = segment_size * (segment_size / 2 + 1);
        in_embed[0] = in_embed[1] = segment_size;
        on_embed[0] = segment_size;
        on_embed[1] = segment_size / 2 + 1;
        stride = 1;
        batch_size = parameters.image_parameters.GetNumberOfWindows();
    }

protected:
    int rank;
    int n[2];
    int i_dist;
    int o_dist;
    int in_embed[2];
    int on_embed[2];
    int stride;
    int batch_size;

    PIVParameters &parameters_;
};

class ForwardFFTHandler : public FFTHandler
{
public:
    ForwardFFTHandler(PIVParameters &parameters) : FFTHandler(parameters)
    {
        cufftPlanMany(&cufft_handler_, rank, n, in_embed, stride, i_dist, on_embed, stride, o_dist, CUFFT_R2C, batch_size);
        this->result = make_shared_gpu<cuComplex>(parameters.image_parameters.height * parameters.image_parameters.width);
    }

    ForwardFFTHandler &operator*=(const ForwardFFTHandler &other);

    void ComputeForwardFFT(SharedPtrGPU<float> &image, bool to_conjugate = false);

    SharedPtrGPU<cufftComplex> result;

private:
    cufftHandle cufft_handler_;
};

class BackwardFFTHandler : public FFTHandler
{
public:
    BackwardFFTHandler(PIVParameters &parameters) : FFTHandler(parameters)
    {
        cufftPlanMany(&cufft_handler_, rank, n, on_embed, stride, o_dist, in_embed, stride, i_dist, CUFFT_C2R, batch_size);
        this->result = make_shared_gpu<float>(parameters.image_parameters.height * parameters.image_parameters.width);
        this->buffer_ = make_shared_gpu<float>(parameters.image_parameters.height * parameters.image_parameters.width);
    }

    void ComputeBackwardFFT(SharedPtrGPU<cuComplex> &image);

    SharedPtrGPU<float> result;

private:
    SharedPtrGPU<float> buffer_;
    cufftHandle cufft_handler_;
};

class Interpolation
{
public:
    explicit Interpolation(PIVParameters &parameters);

    void Interpolate(SharedPtrGPU<float> &correlation_function, SharedPtrGPU<cub::KeyValuePair<int, float>> &input);

    SharedPtrGPU<Point2D<float>> result;

private:
    PIVParameters &parameters_;
};