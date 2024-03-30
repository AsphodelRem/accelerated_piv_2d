#include <core/math/multivalue_argmax.cuh>

MultiArgMaxSearch::MultiArgMaxSearch(const PIVParameters &parameters) {
  this->number_of_windows_ = parameters.GetNumberOfWindows();
  size_t size_of_temp_buffer = 0;

  this->offsets_ = std::make_shared<int[]>(this->number_of_windows_ + 1);
  this->result =
      make_shared_gpu<cub::KeyValuePair<int, float>>(this->number_of_windows_);

  float *dummy_pointer = NULL;
  CUDA_CHECK_ERROR(cub::DeviceSegmentedReduce::ArgMax(
      NULL, size_of_temp_buffer, dummy_pointer, this->result.get(),
      this->number_of_windows_, this->offsets_.get(),
      this->offsets_.get() + 1));

  this->buffer_ = make_shared_gpu<char>(size_of_temp_buffer);

  for (int i = 0; i < this->number_of_windows_ + 1; i++) {
    this->offsets_[i] = parameters.image_parameters.GetWindowSize() *
                        parameters.image_parameters.GetWindowSize() * i;
  }

  this->dev_cub_offsets_ =
      make_shared_gpu<int>(this->number_of_windows_ + 1)
          .UploadHostData(this->offsets_.get(),
                          (this->number_of_windows_ + 1) * sizeof(int));
}

void MultiArgMaxSearch::GetMaxForAllWindows(const SharedPtrGPU<float> &input) {
  size_t size = this->buffer_.size();

  CUDA_CHECK_ERROR(cub::DeviceSegmentedReduce::ArgMax(
      this->buffer_.get(), size, input.get(), this->result.get(),
      this->number_of_windows_, this->dev_cub_offsets_.get(),
      this->dev_cub_offsets_.get() + 1));
}
