#include <image/preprocessing.cuh>

static __global__ void
PreprocessImage_kernel(const uchar *dev_image, float *dev_sliced_image,
                       unsigned int grid_size_x, unsigned int grid_size_y,
                       unsigned int window_size, unsigned int image_width,
                       float top_hat = 0.0) {
  unsigned int j = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int i = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int window_y_idx = blockIdx.z;

  unsigned long long offset_x = 0, offset_y = 0;

  if (window_y_idx >= grid_size_y) {
    return;
  }
  if (i >= window_size || j >= window_size) {
    return;
  }

  for (int window_x_idx = 0; window_x_idx < grid_size_x; window_x_idx++) {
    offset_x = j + window_x_idx * window_size;
    offset_y = i + window_y_idx * window_size;

    float value = 0.0;
    if (top_hat * window_size < i < window_size - top_hat * window_size &&
        top_hat * window_size < j < window_size - top_hat * window_size) {
      value = dev_image[offset_y * image_width + offset_x];
    }

    dev_sliced_image[(window_y_idx * grid_size_x + window_x_idx) * window_size *
                         window_size +
                     (i * window_size + j)] = value;
  }
}

void SplitImageIntoWindows(const SharedPtrGPU<unsigned char> &gray_image,
                           const SharedPtrGPU<float> &sliced_image,
                           const PIVParameters &parameters) {
  const unsigned window_size = parameters.image_parameters.GetWindowSize();
  const unsigned grid_size_x = parameters.image_parameters.GetWidth() / window_size;
  const unsigned grid_size_y = parameters.image_parameters.GetHeight() / window_size;
  const unsigned image_width = parameters.image_parameters.GetWidth();

  dim3 grid_size = {window_size / 16, window_size / 16, grid_size_y};
  dim3 thread_per_block = {16, 16};

  PreprocessImage_kernel<<<grid_size, thread_per_block>>>(
      gray_image.get(), sliced_image.get(), grid_size_x, grid_size_y,
      window_size, image_width);
}

static __global__ void RgbToGray_kernel(const unsigned char *rgb_image,
                                        unsigned char *gray_image, int width,
                                        int height, int channels) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    unsigned int rgbOffset = y * width * channels + x * channels;
    unsigned int gray_offset = y * width + x;

    uchar r = rgb_image[rgbOffset];
    uchar g = rgb_image[rgbOffset + 1];
    uchar b = rgb_image[rgbOffset + 2];

    gray_image[gray_offset] =
        static_cast<uchar>(0.299f * r + 0.587f * g + 0.114f * b);
  }
}

void MakeGrayScale(const SharedPtrGPU<unsigned char> &rgb_image,
                   SharedPtrGPU<unsigned char> &gray_image, const int width,
                   const int height, const int channels) {
  dim3 grid = {static_cast<unsigned int>(width / 32),
               static_cast<unsigned int>(height / 32)};

  dim3 thread_per_block = {32, 32};

  RgbToGray_kernel<<<grid, thread_per_block>>>(
      rgb_image.get(), gray_image.get(), width, height, channels);
}

static __global__ void Normalize_kernel(float *input_1, double *mean,
                                        double *var, int length) {
  unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

  if (index >= length) {
    return;
  }

  const float mean_ = *mean;
  const float var_ = *var;

  input_1[index] -= mean_;
  input_1[index] /= var_;
}

void NormalizeImage(SharedPtrGPU<float> &input,
                    const SharedPtrGPU<double> &mean,
                    const SharedPtrGPU<double> &var,
                    const PIVParameters &parameters) {
  const unsigned length =
      parameters.image_parameters.GetHeight() * parameters.image_parameters.GetWidth();

  int threads_per_block = 1024;
  int grid_size = (length + threads_per_block - 1) / threads_per_block;

  Normalize_kernel<<<grid_size, threads_per_block>>>(input.get(), mean.get(),
                                                     var.get(), length);
}
