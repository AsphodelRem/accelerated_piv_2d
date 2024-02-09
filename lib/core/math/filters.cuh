#pragma once

#include <cuComplex.h>

#include <parameters.cuh>
#include <utils/device_smart_pointer.hpp>

class Filter
{
public:
    explicit Filter(PIVParameters &parameters);

    void filter(SharedPtrGPU<cuComplex> &input);

    // SharedPtrGPU<float> result;

private:
    PIVParameters &parameters_;
};