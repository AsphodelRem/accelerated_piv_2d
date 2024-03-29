#pragma once

#include <cuComplex.h>

#include <core/math/math_operation.hpp>
#include <core/parameters.cuh>
#include <utils/device_smart_pointer.hpp>

class Filter final : public IOperation<float> {
public:
  explicit Filter(PIVParameters &parameters);

  void filter(const SharedPtrGPU<cuComplex> &input);

private:
  PIVParameters &parameters_;
};