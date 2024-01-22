#include <math/fft_handlers.cuh>
#include <math/additional.cuh>

FFTHandler::FFTHandler(const PIVParameters &parameters) :parameters_(parameters)
{
  const int segment_size = parameters.image_parameters.window_size;

  rank = 2;
  n[0] = n[1] = segment_size;
  i_dist = segment_size * segment_size;
  o_dist = segment_size * (segment_size / 2 + 1);
  in_embed[0] = in_embed[1] = segment_size;
  on_embed[0] = segment_size;
  on_embed[1] = segment_size / 2 + 1;
  stride = 1;
  batch_size = parameters.image_parameters.GetNumberOfWindows();
}

ForwardFFTHandler::ForwardFFTHandler(const PIVParameters &parameters) : FFTHandler(parameters)
{
  cufftPlanMany(&cufft_handler_, rank, n, in_embed, stride,
    i_dist, on_embed, stride, o_dist, CUFFT_R2C, batch_size);
  this->result = make_shared_gpu<cuComplex>(parameters.image_parameters.height *
    parameters.image_parameters.width);
}

void ForwardFFTHandler::ComputeForwardFFT(const SharedPtrGPU<float> &image, bool to_conjugate)
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

BackwardFFTHandler::BackwardFFTHandler(const PIVParameters &parameters) : FFTHandler(parameters)
{
  cufftPlanMany(&cufft_handler_, rank, n, on_embed, stride, o_dist, in_embed, stride, i_dist, CUFFT_C2R, batch_size);
  this->result = make_shared_gpu<float>(parameters.image_parameters.height * parameters.image_parameters.width);
  this->buffer_ = make_shared_gpu<float>(parameters.image_parameters.height * parameters.image_parameters.width);
}

void BackwardFFTHandler::ComputeBackwardFFT(const SharedPtrGPU<cuComplex> &image)
{
  cufftExecC2R(this->cufft_handler_, image.get(), this->buffer_.get());

  ShiftSpectrum(this->buffer_.get(), this->result.get(),
           this->parameters_.image_parameters.window_size,
           this->parameters_.image_parameters.window_size,
           this->parameters_.image_parameters.GetNumberOfWindows());
}

