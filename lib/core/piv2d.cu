#include <core/piv2d.cuh>

#include <cub/cub.cuh>

#include <core/math/fft_handlers.cuh>
#include <core/math/filters.cuh>
#include <core/math/interpolations.cuh>
#include <core/math/multivalue_argmax.cuh>

PIVDataContainer::PIVDataContainer(PIVParameters &parameters)
    : parameters_(parameters) {
  auto number_of_window = parameters.GetNumberOfWindows();

  this->data = std::make_shared<Point2D<float>[]>(number_of_window);
  this->preprocessed_data_ = make_shared_gpu<Point2D<float>>(number_of_window);
}

__global__ void FindMovements_kernel(Point2D<float> *interpolated_coordinates,
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
  FindMovements(data, preprocessed_data_, parameters_);

  preprocessed_data_.CopyDataToHost(this->data.get());
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