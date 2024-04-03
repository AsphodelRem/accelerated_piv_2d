#include <stdexcept>
#include <string>

#include <cub/cub.cuh>

#include <core/math/point.cuh>
#include <core/parameters.hpp>
#include <core/math/fft_handlers.cuh>
#include <core/math/filters.cuh>
#include <core/math/interpolations.cuh>
#include <core/math/multivalue_argmax.cuh>
#include <core/piv2d.cuh>

PIVDataContainer::RingBuffer::RingBuffer(PIVParameters& parameters) 
  : capacity_(parameters.memory_settings.GetContainerCapacity())
  , current_id_(0) {

    // Memory preallocating
    for (size_t i = 0; i < capacity_; i++) {
      data_.push_back(std::make_shared<Point2D<float>[]>(parameters.GetNumberOfWindows()));
    }
}

std::shared_ptr<Point2D<float>[]> &
PIVDataContainer::RingBuffer::operator[](size_t index) {
    if (index >= capacity_) {
      throw std::out_of_range(
          "Index is out of range. Number of elements in container is " +
          std::to_string(capacity_) + ", but the index is " +
          std::to_string(index));
    }

    return data_[index];
}

std::shared_ptr<Point2D<float>[]> &PIVDataContainer::RingBuffer::GetNextPtr() {
    if (current_id_ >= capacity_) {
      current_id_ = 0;
    }
    auto &item = data_[current_id_];
    current_id_++;
    return item;
}

PIVDataContainer::PIVDataContainer(PIVParameters &parameters) 
  : parameters_(parameters)
  , data_container_(RingBuffer(parameters)) {
  auto number_of_window = parameters.GetNumberOfWindows();

  this->buffer_ = make_shared_gpu<Point2D<float>>(number_of_window);
}

std::shared_ptr<Point2D<float>[]> &PIVDataContainer::operator[](size_t index) {
  return data_container_.data_[index];
}

__global__ 
void FindMovements_kernel(Point2D<float> *interpolated_coordinates,
                                     Point2D<float> *output_speed,
                                     unsigned int length, float scale_factor,
                                     float time, unsigned int window_size,
                                     bool to_physical_view = false) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < length) {
    bool is_there_movements_by_x = (interpolated_coordinates[idx].x != 0);
    bool is_there_movements_by_y = (interpolated_coordinates[idx].y != 0);

    output_speed[idx].x =
        is_there_movements_by_x *
        (interpolated_coordinates[idx].x - (window_size / 2)) * scale_factor /
        time;

    output_speed[idx].y =
        is_there_movements_by_y *
        (interpolated_coordinates[idx].y - (window_size / 2)) * scale_factor /
        time;

    if (to_physical_view) {
      output_speed[idx].y = -output_speed[idx].y;
    }
  }
}

void FindMovements(SharedPtrGPU<Point2D<float>> &input,
                   SharedPtrGPU<Point2D<float>> &output,
                   PIVParameters &parameters) {
  auto length = parameters.GetNumberOfWindows();

  dim3 grid_size = {(length + 127) / 128};
  dim3 threads_per_block = {128};

  FindMovements_kernel<<<grid_size, threads_per_block>>>(
      input.get(), output.get(), length, 1, 1,
      parameters.image_parameters.GetWindowSize());
}

void PIVDataContainer::StoreData(SharedPtrGPU<Point2D<float>> &data) {
  FindMovements(data, buffer_, parameters_);
  buffer_.CopyDataToHost(this->data_container_.GetNextPtr().get());
}

// TODO: Add an ability to create folders
void PIVDataContainer::SaveAsBinary(size_t index) {
  auto &item = this->data_container_[index];
  auto size = this->parameters_.GetNumberOfWindows() * sizeof(Point2D<float>);

  std::ofstream fout("test" + std::to_string(index) + ".bin_piv",
                     std::ios_base::binary);
  fout.write(reinterpret_cast<const char *>(item.get()), size);
}

PIVDataContainer StartPIV2D(IDataContainer &container,
                            PIVParameters &parameters) {
  ForwardFFTHandler fourier_image_1(parameters);
  ForwardFFTHandler fourier_image_2(parameters);

  BackwardFFTHandler correlation_function(parameters);

  Filter filter(parameters);

  MultiArgMaxSearch multi_max_search(parameters);

  Interpolation interpolation(parameters);

  PIVDataContainer data(parameters);

  auto new_data = container.GetImages();
  while (new_data.has_value()) {
    fourier_image_1.ComputeForwardFFT(new_data.value().get().GetFirstImage(),
                                      true);
    fourier_image_2.ComputeForwardFFT(new_data.value().get().GetSecondImage());

    fourier_image_1 *= fourier_image_2;

    filter.filter(fourier_image_1.GetResult());

    correlation_function.ComputeBackwardFFT(fourier_image_1.GetResult());

    multi_max_search.GetMaxForAllWindows(correlation_function.GetResult());

    interpolation.Interpolate(correlation_function.GetResult(),
                              multi_max_search.GetResult());

    data.StoreData(interpolation.result);

    new_data = container.GetImages();
  }

  return data;
}