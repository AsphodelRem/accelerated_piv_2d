#pragma once

#include <memory>

#include <cufft.h>
#include <cuComplex.h>

#include <cub/cub.cuh>

#include <utils/device_smart_pointer.hpp>
#include <image_container.cuh>
#include <parameters.cuh>
#include <math/math.cuh>

class PIVDataContainer
{
public:
    explicit PIVDataContainer(PIVParameters &parameters);

    void SaveDataInCSV() = delete;

    // std::vector<Point2D<float>> data;
    std::shared_ptr<Point2D<float>[]> data;

    void StoreData(SharedPtrGPU<Point2D<float>> &data);

private:
    PIVParameters &parameters_;

    std::shared_ptr<cub::KeyValuePair<int, float>> buffer_;
    SharedPtrGPU<Point2D<float>> preprocessed_data_;
};

PIVDataContainer StartPIV2D(ImageContainer &container, PIVParameters &parameters);
