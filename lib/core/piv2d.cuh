#pragma once

#include <fstream>
#include <optional>

#include <cuComplex.h>
#include <cufft.h>

#include <core/math/point.cuh>
#include <core/parameters.hpp>
#include <image/image_container.cuh>
#include <utils/device_smart_pointer.hpp>

class PIVDataContainer {
public:
  explicit PIVDataContainer(PIVParameters &parameters);

  void StoreData(SharedPtrGPU<Point2D<float>> &data);

  void SaveAsBinary(size_t index);
  void SaveDataInCSV() = delete;

  std::shared_ptr<Point2D<float>[]>&
  operator[](size_t index);

  friend PIVDataContainer StartPIV2D(IDataContainer &container,
                                     PIVParameters &parameters);

private:
  struct RingBuffer {
    friend PIVDataContainer;

    RingBuffer() = default;
    RingBuffer(PIVParameters &parameters);

    ~RingBuffer() = default;

    std::shared_ptr<Point2D<float>[]> &GetNextPtr();
    std::shared_ptr<Point2D<float>[]> &operator[](size_t index);

  private:
    std::deque<std::shared_ptr<Point2D<float>[]>> data_;
    size_t current_id_;
    size_t capacity_;
  };

  RingBuffer data_container_;
  PIVParameters &parameters_;
  SharedPtrGPU<Point2D<float>> buffer_;
};

PIVDataContainer StartPIV2D(IDataContainer &container,
                            PIVParameters &parameters);
