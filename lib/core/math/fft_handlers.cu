#include <core/math/additional.cuh>
#include <core/math/fft_handlers.cuh>
#include <utils/errors_checking.cuh>

template <typename T>
FFTHandler<T>::FFTHandler(const PIVParameters &parameters)
    : parameters_(parameters) {
  const int segment_size = parameters.image_parameters.GetWindowSize();

  rank = 2;
  n[0] = n[1] = segment_size;
  i_dist = segment_size * segment_size;
  o_dist = segment_size * (segment_size / 2 + 1);
  in_embed[0] = in_embed[1] = segment_size;
  on_embed[0] = segment_size;
  on_embed[1] = segment_size / 2 + 1;
  stride = 1;
  batch_size = parameters.GetNumberOfWindows();
}

ForwardFFTHandler::ForwardFFTHandler(const PIVParameters &parameters)
    : FFTHandler(parameters) {
  cufftPlanMany(&cufft_handler_, rank, n, in_embed, stride, i_dist, on_embed,
                stride, o_dist, CUFFT_R2C, batch_size);

  auto [height, width] = this->parameters_.GetSpectrumSize();
  const int spectrum_height = height;
  const int spectrum_width = width;

  this->result = make_shared_gpu<cuComplex>(spectrum_height * spectrum_width);
}

void ForwardFFTHandler::ComputeForwardFFT(const SharedPtrGPU<float> &image,
                                          bool to_conjugate) {
  CUDA_CHECK_ERROR(cufftExecR2C(this->cufft_handler_, image.get(), this->result.get()));

  if (to_conjugate) {
    Conjugate(this->result.get(), this->parameters_.image_parameters.GetHeight(),
              this->parameters_.image_parameters.GetWidth());
  }
}

ForwardFFTHandler &
ForwardFFTHandler::operator*=(const ForwardFFTHandler &other) {
  auto [height, width] = this->parameters_.GetSpectrumSize();
  const int spectrum_height = height;
  const int spectrum_width = width;

  ElementwiseMultiplication(this->result.get(), other.result.get(),
                            spectrum_height, spectrum_width);

  return *this;
}

BackwardFFTHandler::BackwardFFTHandler(const PIVParameters &parameters)
    : FFTHandler(parameters) {
  cufftPlanMany(&cufft_handler_, rank, n, on_embed, stride, o_dist, in_embed,
                stride, i_dist, CUFFT_C2R, batch_size);
  this->result = make_shared_gpu<float>(parameters.image_parameters.GetHeight() *
                                        parameters.image_parameters.GetWidth());
  this->buffer_ = make_shared_gpu<float>(parameters.image_parameters.GetHeight() *
                                         parameters.image_parameters.GetWidth());
}

void BackwardFFTHandler::ComputeBackwardFFT(
    const SharedPtrGPU<cuComplex> &image) {
  CUDA_CHECK_ERROR(cufftExecC2R(this->cufft_handler_, image.get(), this->buffer_.get()));

  ShiftSpectrum(this->buffer_.get(), this->result.get(),
                this->parameters_.image_parameters.GetWindowSize(),
                this->parameters_.image_parameters.GetWindowSize(),
                this->parameters_.GetNumberOfWindows());
}
