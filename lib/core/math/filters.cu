#include <core/math/filters.cuh>

constexpr unsigned int kBlockSize = 32;

__global__ void LowPassFilter_kernel(cuComplex *spectrum, int U, int V,
                                     float k) {
  unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int u = blockDim.y * blockIdx.y + threadIdx.y;

  if (v == 0 && u == 0) {
    spectrum[u * V + v].x = 0;
    spectrum[u * V + v].y = 0;
  }

  float value = exp(-k * k * (u * u + v * v) / (U * V));

  if (u < U && v < V) {
    spectrum[u * V + v].x *= value;
    spectrum[u * V + v].y *= value;
  }
}

__global__ void BandPassFilter_kernel(cuComplex *spectrum, int U, int V) {
  unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int u = blockDim.y * blockIdx.y + threadIdx.y;

  if (u < U && v < V) {
    float r =
        sqrtf((2.0 * u / U) * (2.0 * u / U) + (2.0 * v / V) * (2.0 * v / V));
    float value = 1.0;

    if (r <= 0.25) {
      value = 4 * r;
    } else if (r > 0.25 && r <= 0.75) {
      value = 1;
    } else if (r > 0.75 && r <= 1) {
      value = 4 - 4 * r;
    } else if (r > 1) {
      // value = 0;
    }

    spectrum[u * V + v].x *= value;
    spectrum[u * V + v].y *= value;
  }
}

void LowPassFilter(cuComplex *spectrum, unsigned int height, unsigned int width,
                   float k) {
  dim3 threads_per_block(kBlockSize, kBlockSize);
  dim3 grid_size((width + kBlockSize - 1) / kBlockSize,
                 (height + kBlockSize - 1) / kBlockSize);

  LowPassFilter_kernel<<<grid_size, threads_per_block>>>(spectrum, height,
                                                         width, k);
}

void BandPassFilter(cuComplex *spectrum, unsigned int height,
                    unsigned int width) {
  dim3 threads_per_block(kBlockSize, kBlockSize);
  dim3 grid_size((width + kBlockSize - 1) / kBlockSize,
                 (height + kBlockSize - 1) / kBlockSize);

  BandPassFilter_kernel<<<grid_size, threads_per_block>>>(spectrum, height,
                                                          width);
}

void DoSpectrumFiltering(cuComplex *spectrum, const PIVParameters &parameters) {
  auto [height, width] = parameters.image_parameters.GetSpectrumSize();
  const unsigned int spectrum_height = height;
  const unsigned int spectrum_width = width;

  auto filter_parameter = parameters.filter_parameters.filter_parameter;

  switch (parameters.filter_parameters.filter_type) {
  case FilterType::kLowPass:
    LowPassFilter(spectrum, spectrum_height, spectrum_width, filter_parameter);
    break;

  case FilterType::kBandPass:
    BandPassFilter(spectrum, spectrum_height, spectrum_width);
    break;

  default:
    break;
  }
}

Filter::Filter(PIVParameters &parameters) : parameters_(parameters) {}

void Filter::filter(const SharedPtrGPU<cuComplex> &input) {
  DoSpectrumFiltering(input.get(), parameters_);
}
