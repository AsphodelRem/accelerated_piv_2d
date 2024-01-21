#include <math/math.cuh>

constexpr unsigned int kBlockSize = 32;
constexpr float epsilon = 1e-8;

MultiArgMaxSearch::MultiArgMaxSearch(PIVParameters &parameters)
{
  this->number_of_windows_ = parameters.image_parameters.GetNumberOfWindows();
  size_t size_of_temp_buffer = 0;

  this->offsets_ = std::make_shared<int[]>(this->number_of_windows_ + 1);
  this->result = make_shared_gpu<cub::KeyValuePair<int, float>>(this->number_of_windows_);

  float *dummy_pointer = NULL;
  CUDA_CHECK_ERROR(cub::DeviceSegmentedReduce::ArgMax(
      NULL, size_of_temp_buffer, dummy_pointer, this->result.get(),
      this->number_of_windows_, this->offsets_.get(), this->offsets_.get() + 1));

  this->buffer_ = make_shared_gpu<char>(size_of_temp_buffer);

  for (int i = 0; i < this->number_of_windows_ + 1; i++)
  {
    this->offsets_[i] = parameters.image_parameters.window_size * parameters.image_parameters.window_size * i;
  }

  this->dev_cub_offsets_ =
      make_shared_gpu<int>(this->number_of_windows_ + 1)
          .UploadHostData(this->offsets_.get(),
                          (this->number_of_windows_ + 1) * sizeof(int));
}

void MultiArgMaxSearch::GetMaxForAllWindows(SharedPtrGPU<float> &input)
{
  size_t size = this->buffer_.size();
  CUDA_CHECK_ERROR(cub::DeviceSegmentedReduce::ArgMax(
      this->buffer_.get(), size, input.get(), this->result.get(),
      this->number_of_windows_, this->dev_cub_offsets_.get(),
      this->dev_cub_offsets_.get() + 1));
}

__global__
void Conjugate_kernel(cuComplex *spectrum, unsigned int height, unsigned int width)
{
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height)
  {
    unsigned int idx = y * width + x;
    spectrum[idx].y = -spectrum[idx].y;
  }
}

void Conjugate(cuComplex *spectrum, unsigned int height, unsigned int width)
{
  dim3 threads_per_block = {32, 32};
  dim3 grid_size = {(width + 31) / 32, (height + 31) / 32};

  Conjugate_kernel<<<grid_size, threads_per_block>>>(spectrum, height, width);
}

__global__
void Multiplicate_kernel(cuComplex *a, cuComplex *b,
                                    unsigned int height, unsigned int width)
{
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (y < height && x < width)
  {
    unsigned int idx = y * width + x;
    a[idx] = cuCmulf(a[idx], b[idx]);
  }
}

__global__
void SpectrumShift2D_kernel(float *spectrum,
                            float *shifted_spectrum,
                            unsigned int width,
                            unsigned int height)
{
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int window = blockIdx.z;

  unsigned int window_offset = window * width * height;

  if (x < width && y < height)
  {
    unsigned int x1 = (x + (width / 2)) % width;
    unsigned int y1 = (y + (height / 2)) % height;

    shifted_spectrum[window_offset + y1 * width + x1] = spectrum[window_offset + y * width + x];
    shifted_spectrum[window_offset + y * width + x] = spectrum[window_offset + y1 * width + x1];
  }
}

void ElementwiseMultiplication(cuComplex *a,
                              cuComplex *b,
                              unsigned int height,
                              unsigned int width)
{
  dim3 threads_per_block = {32, 32};
  dim3 grid_size = {(width + 31) / 32, (height + 31) / 32};

  Multiplicate_kernel<<<grid_size, threads_per_block>>>(a, b, height, width);
}

void ShiftSpectrum(float *spectrum,
                  float *shifted_spectrum,
                  unsigned int width,
                  unsigned int height,
                  unsigned int number_of_windows)
{
  dim3 threads_per_block(16, 16);
  dim3 num_blocks(width / 16, height / 16, number_of_windows);

  SpectrumShift2D_kernel<<<num_blocks, threads_per_block>>>(spectrum, shifted_spectrum, width, height);
}

void ForwardFFTHandler::ComputeForwardFFT(SharedPtrGPU<float> &image, bool to_conjugate)
{
  cufftExecR2C(this->cufft_handler_, image.get(), this->result.get());

  if (to_conjugate)
  {
    Conjugate(this->result.get(), this->parameters_.image_parameters.height,
              this->parameters_.image_parameters.width);
  }
}

ForwardFFTHandler &
ForwardFFTHandler::operator*=(const ForwardFFTHandler &other)
{
  auto window_size = this->parameters_.image_parameters.window_size;
  auto image_height = this->parameters_.image_parameters.height;
  auto image_width = this->parameters_.image_parameters.width;

  auto spectrum_height = image_height - (image_height % window_size);
  auto spectrum_width = (image_width - (image_width % window_size)) / window_size * (window_size / 2 + 1);

  ElementwiseMultiplication(this->result.get(), other.result.get(), spectrum_height, spectrum_width);

  return *this;
}

void BackwardFFTHandler::ComputeBackwardFFT(SharedPtrGPU<cuComplex> &image)
{
  cufftExecC2R(this->cufft_handler_, image.get(), this->buffer_.get());

  ShiftSpectrum(this->buffer_.get(), this->result.get(),
           this->parameters_.image_parameters.window_size,
           this->parameters_.image_parameters.window_size,
           this->parameters_.image_parameters.GetNumberOfWindows());
}

__device__
void DevGaussInterpolation(Point2D<float> *output_interpolated_values,
                                      unsigned int current_window_index,
                                      unsigned int x,
                                      unsigned int y,
                                      float center,
                                      float up,
                                      float down,
                                      float left,
                                      float right)
{
  float delta_y = (log(left + epsilon) - log(right + epsilon)) /
                 (2 * log(left + epsilon) - 4 * log(center + epsilon) + 2 * log(right + epsilon));

  float delta_x = (log(down + epsilon) - log(up + epsilon)) /
                 (2 * log(down + epsilon) - 4 * log(center + epsilon) + 2 * log(up + epsilon));

  output_interpolated_values[current_window_index].x = (static_cast<float>(x) + delta_x);
  output_interpolated_values[current_window_index].y = (static_cast<float>(y) + delta_y);
}

__device__
void DevParabolicInterpolation(Point2D<float> *output_interpolated_values,
                                          unsigned int current_window_index,
                                          unsigned int x,
                                          unsigned int y,
                                          float center,
                                          float up,
                                          float down,
                                          float left,
                                          float right)
{
  float delta_y = (left - right) / (2 * left - 4 * center + 2 * right);
  float delta_x = (down - up) / (2 * down - 4 * center + 2 * up);

  output_interpolated_values[current_window_index].x = static_cast<float>(x) + delta_x;
  output_interpolated_values[current_window_index].y = static_cast<float>(y) + delta_y;
}

__global__
void SubpixelInterpolation_kernel(const float *correlation_function,
                                             cub::KeyValuePair<int, float> *maximum_values,
                                             Point2D<float> *output_interpolated_values,
                                             unsigned length,
                                             unsigned int window_size,
                                             int subpixel_interpolation_type)
{
  float center = 1.0, up = 1.0, down = 1.0, left = 1.0, right = 1.0;

  unsigned int window = blockDim.x * blockIdx.x + threadIdx.x;

  if (window >= length)
    return;

  unsigned int current_x_idx = maximum_values[window].key % window_size,
               current_y_idx = maximum_values[window].key / window_size;

  // Check border
  if (
      (current_y_idx <= 0) ||
      (current_x_idx <= 0) ||
      (current_y_idx >= (window_size - 1) ||
      (current_x_idx >= (window_size - 1)))
      )
  {
    output_interpolated_values[window].x = static_cast<float>(current_x_idx);
    output_interpolated_values[window].y = static_cast<float>(current_y_idx);

    return;
  }

  center = correlation_function[window_size * window_size * window + (current_y_idx * window_size + current_x_idx)],

  up = correlation_function[window_size * window_size * window +
                           ((current_y_idx + 1) * window_size + current_x_idx)],

  down = correlation_function[window_size * window_size * window +
                             ((current_y_idx - 1) * window_size + current_x_idx)];

  right = correlation_function[window_size * window_size * window +
                              (current_y_idx * window_size + (current_x_idx + 1))],

  left = correlation_function[window_size * window_size * window +
                             (current_y_idx * window_size + (current_x_idx - 1))];

  // Avoiding negative values when gaussian is used
  if ((up <= 0 || left <= 0 || down <= 0 || right <= 0 || center <= 0) &&
      subpixel_interpolation_type == InterpolationType::kGaussian)
  {
    subpixel_interpolation_type = InterpolationType::kParabolic;
  }

  switch (subpixel_interpolation_type)
  {
  case InterpolationType::kGaussian:
    DevGaussInterpolation(output_interpolated_values, window, current_x_idx, current_y_idx,
                          center, up, down, left, right);
    break;

  case InterpolationType::kParabolic:
    DevParabolicInterpolation(output_interpolated_values, window, current_x_idx, current_y_idx,
                              center, up, down, left, right);
    break;

  default:
    output_interpolated_values[window].x = static_cast<float>(current_x_idx);
    output_interpolated_values[window].y = static_cast<float>(current_y_idx);
    break;
  }
}

Interpolation::Interpolation(PIVParameters &parameters) : parameters_(parameters)
{
  this->result = make_shared_gpu<Point2D<float>>(parameters.image_parameters.GetNumberOfWindows());
}

void Interpolation::Interpolate(SharedPtrGPU<float> &correlation_function,
                                SharedPtrGPU<cub::KeyValuePair<int, float>> &input)
{
  auto length = parameters_.image_parameters.GetNumberOfWindows();
  auto window_size = parameters_.image_parameters.window_size;

  dim3 grid_size = {(length + kBlockSize - 1) / kBlockSize};
  dim3 threads_per_block = {kBlockSize};

  SubpixelInterpolation_kernel<<<grid_size, threads_per_block>>>(correlation_function.get(), input.get(),
                                                              this->result.get(),
                                                              length, window_size, parameters_.interpolation_parameters.interpolation_type);
}