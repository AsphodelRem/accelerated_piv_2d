#include <core/math/interpolations.cuh>

constexpr float epsilon = 1.0e-8;
constexpr unsigned int kBlockSize = 32;

__device__ void
DevGaussInterpolation(Point2D<float> *output_interpolated_values,
                      unsigned int current_window_index, unsigned int x,
                      unsigned int y, float center, float up, float down,
                      float left, float right) {
  float delta_y = (log(left + epsilon) - log(right + epsilon)) /
                  (2 * log(left + epsilon) - 4 * log(center + epsilon) +
                   2 * log(right + epsilon));

  float delta_x = (log(down + epsilon) - log(up + epsilon)) /
                  (2 * log(down + epsilon) - 4 * log(center + epsilon) +
                   2 * log(up + epsilon));

  output_interpolated_values[current_window_index].x =
      (static_cast<float>(x) + delta_x);
  output_interpolated_values[current_window_index].y =
      (static_cast<float>(y) + delta_y);
}

__device__ void
DevParabolicInterpolation(Point2D<float> *output_interpolated_values,
                          unsigned int current_window_index, unsigned int x,
                          unsigned int y, float center, float up, float down,
                          float left, float right) {
  float delta_y = (left - right) / (2 * left - 4 * center + 2 * right);
  float delta_x = (down - up) / (2 * down - 4 * center + 2 * up);

  output_interpolated_values[current_window_index].x =
      static_cast<float>(x) + delta_x;
  output_interpolated_values[current_window_index].y =
      static_cast<float>(y) + delta_y;
}

__global__ void
SubpixelInterpolation_kernel(const float *correlation_function,
                             cub::KeyValuePair<int, float> *maximum_values,
                             Point2D<float> *output_interpolated_values,
                             unsigned length, unsigned int window_size,
                             InterpolationType subpixel_interpolation_type) {
  float center = 1.0, up = 1.0, down = 1.0, left = 1.0, right = 1.0;

  unsigned int window = blockDim.x * blockIdx.x + threadIdx.x;

  if (window >= length) {
    return;
  }

  unsigned int current_x_idx = maximum_values[window].key % window_size,
               current_y_idx = maximum_values[window].key / window_size;

  // Check border
  if ((current_y_idx <= 0) || (current_x_idx <= 0) ||
      (current_y_idx >= (window_size - 1) ||
       (current_x_idx >= (window_size - 1)))) {
    output_interpolated_values[window].x = static_cast<float>(current_x_idx);
    output_interpolated_values[window].y = static_cast<float>(current_y_idx);

    return;
  }

  center = correlation_function[window_size * window_size * window +
                                (current_y_idx * window_size + current_x_idx)],

  up =
      correlation_function[window_size * window_size * window +
                           ((current_y_idx + 1) * window_size + current_x_idx)],

  down =
      correlation_function[window_size * window_size * window +
                           ((current_y_idx - 1) * window_size + current_x_idx)];

  right =
      correlation_function[window_size * window_size * window +
                           (current_y_idx * window_size + (current_x_idx + 1))],

  left =
      correlation_function[window_size * window_size * window +
                           (current_y_idx * window_size + (current_x_idx - 1))];

  // Avoiding negative values when gaussian is used
  if ((up <= 0 || left <= 0 || down <= 0 || right <= 0 || center <= 0) &&
      subpixel_interpolation_type == InterpolationType::kGaussian) {
    subpixel_interpolation_type = InterpolationType::kParabolic;
  }

  switch (subpixel_interpolation_type) {
  case InterpolationType::kGaussian:
    DevGaussInterpolation(output_interpolated_values, window, current_x_idx,
                          current_y_idx, center, up, down, left, right);
    break;

  case InterpolationType::kParabolic:
    DevParabolicInterpolation(output_interpolated_values, window, current_x_idx,
                              current_y_idx, center, up, down, left, right);
    break;

  default:
    output_interpolated_values[window].x = static_cast<float>(current_x_idx);
    output_interpolated_values[window].y = static_cast<float>(current_y_idx);
    break;
  }
}

Interpolation::Interpolation(PIVParameters &parameters)
    : parameters_(parameters) {
  this->result = make_shared_gpu<Point2D<float>>(
      parameters.GetNumberOfWindows());
}

void Interpolation::Interpolate(
    const SharedPtrGPU<float> &correlation_function,
    const SharedPtrGPU<cub::KeyValuePair<int, float>> &input) {
  auto length = parameters_.GetNumberOfWindows();
  auto window_size = parameters_.image_parameters.GetWindowSize();

  dim3 grid_size = {(length + kBlockSize - 1) / kBlockSize};
  dim3 threads_per_block = {kBlockSize};

  SubpixelInterpolation_kernel<<<grid_size, threads_per_block>>>(
      correlation_function.get(), input.get(), this->result.get(), length,
      window_size, parameters_.interpolation_parameters.GetInterpolationType());
}
