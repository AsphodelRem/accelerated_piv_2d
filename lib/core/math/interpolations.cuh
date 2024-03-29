#pragma once

#include <cub/util_type.cuh>

#include <core/math/point.cuh>
#include <core/parameters.cuh>
#include <utils/device_smart_pointer.hpp>

#include "math_operation.hpp"

class Interpolation final : public IOperation<Point2D<float>> {
public:
  explicit Interpolation(PIVParameters &parameters);

  void Interpolate(const SharedPtrGPU<float> &correlation_function,
                   const SharedPtrGPU<cub::KeyValuePair<int, float>> &input);

  // SharedPtrGPU<Point2D<float>> result;

private:
  PIVParameters &parameters_;
};
