#pragma once

#include <cub/util_type.cuh>

#include <math/point.cuh>
#include <utils/device_smart_pointer.hpp>
#include <parameters.cuh>

class Interpolation
{
public:
    explicit Interpolation(PIVParameters &parameters);

    void Interpolate(const SharedPtrGPU<float> &correlation_function,
        const SharedPtrGPU<cub::KeyValuePair<int, float>> &input);

    SharedPtrGPU<Point2D<float>> result;

private:
    PIVParameters &parameters_;
};