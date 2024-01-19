#include <math/math.cuh>

MultiArgMaxSearch::MultiArgMaxSearch(PIVParameters &params)
{
  this->_number_of_windows = params.image_params.getNumberOfWindows();
  size_t size_of_temp_buffer = 0;

  this->_offsets = std::make_shared<int[]>(this->_number_of_windows + 1);
  this->result = make_shared_gpu<cub::KeyValuePair<int, float>>(this->_number_of_windows);

  float *dummy_pointer = NULL;
  auto status = cub::DeviceSegmentedReduce::ArgMax(
      NULL, size_of_temp_buffer, dummy_pointer, this->result.get(),
      this->_number_of_windows, this->_offsets.get(), this->_offsets.get() + 1);

  this->_buffer = make_shared_gpu<char>(size_of_temp_buffer);

  for (int i = 0; i < this->_number_of_windows + 1; i++)
  {
    this->_offsets[i] = params.image_params.window_size * params.image_params.window_size * i;
  }

  this->_dOffsets =
      make_shared_gpu<int>(this->_number_of_windows + 1)
          .uploadHostData(this->_offsets.get(),
                          (this->_number_of_windows + 1) * sizeof(int));
}

void MultiArgMaxSearch::getMaxForAllWindows(SharedPtrGPU<float> &input)
{
  size_t size = this->_buffer.size();
  auto status = cub::DeviceSegmentedReduce::ArgMax(
      this->_buffer.get(), size, input.get(), this->result.get(),
      this->_number_of_windows, this->_dOffsets.get(),
      this->_dOffsets.get() + 1);
}

__global__ void conjugated_kernel(cuComplex *complexData, unsigned int height,
                                  unsigned int width)
{
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height)
  {
    unsigned int idx = y * width + x;
    complexData[idx].y = -(complexData[idx].y);
  }
}

void conjugate(cuComplex *complexData, unsigned int height,
               unsigned int width)
{
  dim3 threadsPerBlock = {32, 32};
  dim3 gridSize = {(width + 31) / 32, (height + 31) / 32};

  conjugated_kernel<<<gridSize, threadsPerBlock>>>(complexData, height, width);
}

__global__ void multiplicate_kernel(cuComplex *a, cuComplex *b,
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

__global__ void fftShift2D_kernel(float *data, float *dst, unsigned int width,
                                  unsigned int height)
{
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int window = blockIdx.z;

  unsigned int windowOffset = window * width * height;

  if (x < width && y < height)
  {
    unsigned int x1 = (x + (width / 2)) % width;
    unsigned int y1 = (y + (height / 2)) % height;

    dst[windowOffset + y1 * width + x1] = data[windowOffset + y * width + x];
    dst[windowOffset + y * width + x] = data[windowOffset + y1 * width + x1];
  }
}

void multiplicate(cuComplex *a, cuComplex *b, unsigned int height,
                  unsigned int width)
{
  dim3 threadsPerBlock = {32, 32};
  dim3 gridSize = {(width + 31) / 32, (height + 31) / 32};

  multiplicate_kernel<<<gridSize, threadsPerBlock>>>(a, b, height, width);
}

void fftShift(float *data, float *dst, unsigned int width, unsigned int height,
              unsigned int numberOfSegments)
{
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(width / 16, height / 16, numberOfSegments);

  fftShift2D_kernel<<<numBlocks, threadsPerBlock>>>(data, dst, width, height);
}

void ForwardFFTHandler::computeForwardFFT(SharedPtrGPU<float> &image, bool to_conjugate)
{
  auto status = cufftExecR2C(this->cufftHandler, image.get(), this->result.get());

  if (to_conjugate)
  {
    conjugate(this->result.get(), this->parameters_.image_params.height,
              this->parameters_.image_params.width);
  }
}

ForwardFFTHandler &
ForwardFFTHandler::operator*=(const ForwardFFTHandler &other)
{
  auto window_size = this->parameters_.image_params.window_size;

  auto spectrum_height = this->parameters_.image_params.height -
                         (this->parameters_.image_params.height % this->parameters_.image_params.window_size);

  auto spectrum_width = 279;

  multiplicate(this->result.get(), other.result.get(),
               spectrum_height,
               spectrum_width);

  return *this;
}

void BackwardFFTHandler::computeBackwardFFT(SharedPtrGPU<cuComplex> &image)
{
  auto status = cufftExecC2R(this->cufftHandler, image.get(), this->buffer.get());

  fftShift(this->buffer.get(), this->result.get(),
           this->parameters_.image_params.window_size,
           this->parameters_.image_params.window_size,
           this->parameters_.image_params.getNumberOfWindows());
}

constexpr unsigned int kBlockSize = 32;
constexpr float epsilon = 1e-8;

__device__ void devGaussInterpolation(Point2D<float> *outputInterpolatedValues, unsigned int currentWindow, unsigned int XIndex, unsigned int YIndex,
                                      float center, float up, float down, float left, float right)
{
  float deltaY = (log(left + epsilon) - log(right + epsilon)) /
                 (2 * log(left + epsilon) - 4 * log(center + epsilon) + 2 * log(right + epsilon));

  float deltaX = (log(down + epsilon) - log(up + epsilon)) /
                 (2 * log(down + epsilon) - 4 * log(center + epsilon) + 2 * log(up + epsilon));

  outputInterpolatedValues[currentWindow].x = (static_cast<float>(XIndex) + deltaX);
  outputInterpolatedValues[currentWindow].y = (static_cast<float>(YIndex) + deltaY);
}

__device__ void devParabolicInterpolation(Point2D<float> *outputInterpolatedValues, unsigned int currentWindow, unsigned int XIndex, unsigned int YIndex,
                                          float center, float up, float down, float left, float right)
{
  float deltaY = (left - right) / (2 * left - 4 * center + 2 * right);
  float deltaX = (down - up) / (2 * down - 4 * center + 2 * up);

  outputInterpolatedValues[currentWindow].x = static_cast<float>(XIndex) + deltaX;
  outputInterpolatedValues[currentWindow].y = static_cast<float>(YIndex) + deltaY;
}

__global__ void subpixelInterpolation_kernel(const float *correlationFunction,
                                             cub::KeyValuePair<int, float> *maximumValues,
                                             Point2D<float> *outputInterpolatedValues,
                                             unsigned length, unsigned int segmentSize,
                                             int subpixelInterpolationType)
{
  float center = 1.0, up = 1.0, down = 1.0, left = 1.0, right = 1.0;

  unsigned int window = blockDim.x * blockIdx.x + threadIdx.x;

  if (window >= length)
    return;

  unsigned int currentXIdx = maximumValues[window].key % segmentSize,
               currentYIdx = maximumValues[window].key / segmentSize;

  // Check border
  if (
      (currentYIdx <= 0) ||
      (currentXIdx <= 0) ||
      (currentYIdx >= (segmentSize - 1) ||
       (currentXIdx >= (segmentSize - 1))))
  {
    outputInterpolatedValues[window].x = static_cast<float>(currentXIdx);
    outputInterpolatedValues[window].y = static_cast<float>(currentYIdx);

    return;
  }

  center = correlationFunction[segmentSize * segmentSize * window + (currentYIdx * segmentSize + currentXIdx)],

  up = correlationFunction[segmentSize * segmentSize * window +
                           ((currentYIdx + 1) * segmentSize + currentXIdx)],

  down = correlationFunction[segmentSize * segmentSize * window +
                             ((currentYIdx - 1) * segmentSize + currentXIdx)];

  right = correlationFunction[segmentSize * segmentSize * window +
                              (currentYIdx * segmentSize + (currentXIdx + 1))],

  left = correlationFunction[segmentSize * segmentSize * window +
                             (currentYIdx * segmentSize + (currentXIdx - 1))];

  // Avoiding negative values when gaussian is used
  if ((up <= 0 || left <= 0 || down <= 0 || right <= 0 || center <= 0) &&
      subpixelInterpolationType == InterpolationType::kGaussian)
  {
    subpixelInterpolationType = InterpolationType::kParabolic;
  }

  switch (subpixelInterpolationType)
  {
  case InterpolationType::kGaussian:
    devGaussInterpolation(outputInterpolatedValues, window, currentXIdx, currentYIdx,
                          center, up, down, left, right);
    break;

  case InterpolationType::kParabolic:
    devParabolicInterpolation(outputInterpolatedValues, window, currentXIdx, currentYIdx,
                              center, up, down, left, right);
    break;

  default:
    outputInterpolatedValues[window].x = static_cast<float>(currentXIdx);
    outputInterpolatedValues[window].y = static_cast<float>(currentYIdx);
    break;
  }
}

Interpolation::Interpolation(PIVParameters &parameters) : parameters_(parameters)
{
  this->result = make_shared_gpu<Point2D<float>>(parameters.image_params.getNumberOfWindows());
}

void Interpolation::interpolate(SharedPtrGPU<float> &correlationFunction,
                                SharedPtrGPU<cub::KeyValuePair<int, float>> &input)
{
  auto length = parameters_.image_params.getNumberOfWindows();
  auto window_size = parameters_.image_params.window_size;

  dim3 gridSize = {(length + kBlockSize - 1) / kBlockSize};
  dim3 threadsPerBlock = {kBlockSize};

  subpixelInterpolation_kernel<<<gridSize, threadsPerBlock>>>(correlationFunction.get(), input.get(),
                                                              this->result.get(),
                                                              length, window_size, parameters_.interpolation_params.interpolationType);
}