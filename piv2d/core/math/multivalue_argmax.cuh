#pragma once

#include <memory>

#include <cub/util_type.cuh>
#include <cub/device/device_segmented_reduce.cuh>

#include <parameters.cuh>
#include <device_smart_pointer.hpp>
#include <utils/errors_checking.cuh>

class MultiArgMaxSearch
{
public:
    explicit MultiArgMaxSearch(const PIVParameters &parameters);
    ~MultiArgMaxSearch() = default;

    void GetMaxForAllWindows(const SharedPtrGPU<float> &input);

    SharedPtrGPU<cub::KeyValuePair<int, float>> result;

private:
    std::shared_ptr<int[]> offsets_;
    SharedPtrGPU<int> dev_cub_offsets_;
    SharedPtrGPU<char> buffer_;

    unsigned int number_of_windows_;
};
