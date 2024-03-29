#pragma once

#include <memory>

#include <cuComplex.h>
#include <cufft.h>

#include <cub/util_type.cuh>

#include <core/math/point.cuh>
#include <core/parameters.cuh>
#include <image/image_container.cuh>
#include <utils/device_smart_pointer.hpp>

class PIVDataContainer {
public:
  explicit PIVDataContainer(PIVParameters &parameters);

  void SaveDataInCSV() = delete;

  std::shared_ptr<Point2D<float>[]> data;

  void StoreData(SharedPtrGPU<Point2D<float>> &data);

private:
  PIVParameters &parameters_;

  std::shared_ptr<cub::KeyValuePair<int, float>> buffer_;
  SharedPtrGPU<Point2D<float>> preprocessed_data_;
};

PIVDataContainer StartPIV2D(IDataContainer &container,
                            PIVParameters &parameters);
