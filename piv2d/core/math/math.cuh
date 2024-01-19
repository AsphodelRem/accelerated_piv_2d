#pragma once

#include <memory>

#include <cuComplex.h>
#include <cufft.h>

#include <cub/util_type.cuh>
#include <cub/device/device_segmented_reduce.cuh>

#include <parameters.cuh>
#include <device_smart_pointer.hpp>

template <typename T>
struct Point2D
{
    T x, y;
};

template <typename T>
struct MovementDescription
{
    int centerX, centerY;
    T u, v;
    T correlation;
};

void conjugate(cuComplex *complexData, unsigned int height, unsigned int width);

void fftShift(float *data, float *dst, unsigned int width, unsigned int height, unsigned int numberOfSegments);

class MultiArgMaxSearch
{
public:
    MultiArgMaxSearch(PIVParameters &params);
    ~MultiArgMaxSearch() = default;

    void getMaxForAllWindows(SharedPtrGPU<float> &input);

    SharedPtrGPU<cub::KeyValuePair<int, float>> result;

private:
    std::shared_ptr<int[]> _offsets;
    SharedPtrGPU<int> _dOffsets;
    SharedPtrGPU<char> _buffer;

    unsigned int _number_of_windows;
};

class FFTHandler
{
public:
    FFTHandler(PIVParameters &parameters) : parameters_(parameters)
    {
        const int segment_size = parameters.image_params.window_size;

        rank = 2;
        n[0] = n[1] = segment_size;
        iDist = segment_size * segment_size;
        oDist = segment_size * (segment_size / 2 + 1);
        inEmbed[0] = inEmbed[1] = segment_size;
        onEmbed[0] = segment_size;
        onEmbed[1] = segment_size / 2 + 1;
        stride = 1;
        batchSize = parameters.image_params.getNumberOfWindows();
    }

protected:
    int rank;
    int n[2];
    int iDist;
    int oDist;
    int inEmbed[2];
    int onEmbed[2];
    int stride;
    int batchSize;

    PIVParameters &parameters_;
};

class ForwardFFTHandler : public FFTHandler
{
public:
    ForwardFFTHandler(PIVParameters &parameters) : FFTHandler(parameters)
    {
        cufftPlanMany(&cufftHandler, rank, n, inEmbed, stride, iDist, onEmbed, stride, oDist, CUFFT_R2C, batchSize);
        this->result = make_shared_gpu<cuComplex>(parameters.image_params.height * parameters.image_params.width);
    }

    ForwardFFTHandler &operator*=(const ForwardFFTHandler &other);

    void computeForwardFFT(SharedPtrGPU<float> &image, bool to_conjugate = false);

    SharedPtrGPU<cufftComplex> result;

private:
    cufftHandle cufftHandler;
};

class BackwardFFTHandler : public FFTHandler
{
public:
    BackwardFFTHandler(PIVParameters &parameters) : FFTHandler(parameters)
    {
        cufftPlanMany(&cufftHandler, rank, n, onEmbed, stride, oDist, inEmbed, stride, iDist, CUFFT_C2R, batchSize);
        this->result = make_shared_gpu<float>(parameters.image_params.height * parameters.image_params.width);
        this->buffer = make_shared_gpu<float>(parameters.image_params.height * parameters.image_params.width);
    }

    void computeBackwardFFT(SharedPtrGPU<cuComplex> &image);

    SharedPtrGPU<float> result;

private:
    SharedPtrGPU<float> buffer;
    cufftHandle cufftHandler;
};

class Interpolation
{
public:
    explicit Interpolation(PIVParameters &parameters);

    void interpolate(SharedPtrGPU<float> &correlationFunction, SharedPtrGPU<cub::KeyValuePair<int, float>> &input);

    SharedPtrGPU<Point2D<float>> result;

private:
    PIVParameters &parameters_;
};