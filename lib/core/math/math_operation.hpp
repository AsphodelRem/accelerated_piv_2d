#pragma once

#include <utils/device_smart_pointer.hpp>

template <typename T> class IOperation {
public:
  IOperation() = default;
  virtual ~IOperation() = default;

  const SharedPtrGPU<T> &GetResult() const { return result; }

  SharedPtrGPU<T> result;
};
