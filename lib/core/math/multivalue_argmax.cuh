#pragma once

#include <memory>

#include <cub/device/device_segmented_reduce.cuh>
#include <cub/util_type.cuh>

#include <core/math/math_operation.hpp>
#include <core/parameters.cuh>
#include <utils/device_smart_pointer.hpp>
#include <utils/errors_checking.cuh>

class MultiArgMaxSearch final
    : public IOperation<cub::KeyValuePair<int, float>> {
public:
  explicit MultiArgMaxSearch(const PIVParameters &parameters);
  ~MultiArgMaxSearch() override = default;

  void GetMaxForAllWindows(const SharedPtrGPU<float> &input);

  // SharedPtrGPU<cub::KeyValuePair<int, float>> result;

private:
  std::shared_ptr<int[]> offsets_;
  SharedPtrGPU<int> dev_cub_offsets_;
  SharedPtrGPU<char> buffer_;

  unsigned int number_of_windows_;
};
