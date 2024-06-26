#pragma once

#include <string>

#include <nppdefs.h>
#include <nppi.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <core/parameters.hpp>
#include <image/preprocessing.cuh>
#include <utils/device_smart_pointer.hpp>

template <typename T> class ImagePair {
public:
  ImagePair() = default;
  ImagePair(const std::string &image_1, const std::string &image_2);
  ImagePair(const PIVParameters& parameters);

  ImagePair &UploadNewImages(const std::string &image_1,
                             const std::string &image_2);

  ImagePair &UploadNewImages(cv::Mat &image_1, cv::Mat &image_2);

  SharedPtrGPU<T> &GetFirstImage();
  SharedPtrGPU<T> &GetSecondImage();

private:
  SharedPtrGPU<T> image_a_, image_b_;
  SharedPtrGPU<T> buffer_1_, buffer_2_;

  void LoadData_(cv::Mat &image_1, cv::Mat &image_2);
};

template <typename T, typename T2> class PreprocessedImagesPair {
public:
  PreprocessedImagesPair() = default;
  PreprocessedImagesPair(const PIVParameters &parameters);

  PreprocessedImagesPair &UploadNewImages(ImagePair<T> &new_images);

  SharedPtrGPU<T2> &GetFirstImage();
  SharedPtrGPU<T2> &GetSecondImage();

private:
  SharedPtrGPU<T2> output_first_image_, output_second_image_;
  PIVParameters parameters_;

  // This struct store mean and variance of images and contains a buffer for
  // nppiMeanStdDevGetBufferHostSize_32f_C1R function
  struct ImageStatictic {
    SharedPtrGPU<double> dev_mean;
    SharedPtrGPU<double> dev_var;

    int size_of_buffer_;
    NppiSize roi_size;
    SharedPtrGPU<Npp8u> buffer_;
  } image_statictic_;
};

template <typename T>
ImagePair<T>::ImagePair(const PIVParameters& parameters) {
  auto size = parameters.image_parameters.GetHeight() * 
      parameters.image_parameters.GetWidth() * parameters.image_parameters.GetChannels();

  auto grayscaled_image_size = parameters.image_parameters.GetHeight() * 
      parameters.image_parameters.GetWidth();

  // Creating buffer for storing rgb image
  this->buffer_1_ =
      make_shared_gpu<unsigned char>(size);

  this->buffer_2_ =
      make_shared_gpu<unsigned char>(size);

  // Creating buffer for storing gray scale image
  this->image_a_ = make_shared_gpu<unsigned char>(grayscaled_image_size);
  this->image_b_ = make_shared_gpu<unsigned char>(grayscaled_image_size);    
}

template <typename T>
ImagePair<T> &ImagePair<T>::UploadNewImages(const std::string &image_1,
                                            const std::string &image_2) {
  cv::Mat image_a = cv::imread(image_1);
  cv::Mat image_b = cv::imread(image_2);

  this->LoadData_(image_a, image_b);

  return *this;
}

template <typename T>
ImagePair<T> &ImagePair<T>::UploadNewImages(cv::Mat &image_1,
                                            cv::Mat &image_2) {
  this->LoadData_(image_1, image_2);
  return *this;
}

template <typename T>
void ImagePair<T>::LoadData_(cv::Mat &image_1, cv::Mat &image_2) {
  if (image_1.cols != image_2.cols || image_1.rows != image_2.rows) {
    throw std::runtime_error("Images should have same shape");
  }

  unsigned int image_width = image_1.cols;
  unsigned int image_height = image_1.rows;
  unsigned int channels = image_1.channels();
  unsigned long long size = image_height * image_width * channels;

  this->buffer_1_.UploadHostData(image_1.data, size);
  this->buffer_2_.UploadHostData(image_2.data, size);

  if (channels != 1) {
    MakeGrayScale(this->buffer_1_, this->image_a_, image_width, image_height,
                  channels);

    MakeGrayScale(this->buffer_2_, this->image_b_, image_width, image_height,
                  channels);
  }
}

template <typename T> SharedPtrGPU<T> &ImagePair<T>::GetFirstImage() {
  return this->image_a_;
}

template <typename T> SharedPtrGPU<T> &ImagePair<T>::GetSecondImage() {
  return this->image_b_;
}

template <typename T, typename T2>
PreprocessedImagesPair<T, T2>::PreprocessedImagesPair(const PIVParameters &parameters)
    : parameters_(parameters) {

  auto image_size = parameters.image_parameters.GetHeight() *
        parameters.image_parameters.GetWidth();

  this->output_first_image_ = make_shared_gpu<T2>(image_size);
  this->output_second_image_ = make_shared_gpu<T2>(image_size);

  this->image_statictic_.dev_mean = make_shared_gpu<double>(1);
  this->image_statictic_.dev_var = make_shared_gpu<double>(1);

  this->image_statictic_.roi_size.height = parameters.image_parameters.GetHeight();
  this->image_statictic_.roi_size.width = parameters.image_parameters.GetWidth();

  nppiMeanStdDevGetBufferHostSize_32f_C1R(
      this->image_statictic_.roi_size, &this->image_statictic_.size_of_buffer_);

  this->image_statictic_.buffer_ =
      make_shared_gpu<Npp8u>(this->image_statictic_.size_of_buffer_);
}

template <typename T, typename T2>
PreprocessedImagesPair<T, T2> &
PreprocessedImagesPair<T, T2>::UploadNewImages(ImagePair<T> &new_images) {
  auto first_image = new_images.GetFirstImage();
  auto second_image = new_images.GetSecondImage();

  SplitImageIntoWindows(first_image, this->output_first_image_,
                        this->parameters_);
  SplitImageIntoWindows(second_image, this->output_second_image_,
                        this->parameters_);

  const int image_width_in_bytes =
      parameters_.image_parameters.GetWidth() * sizeof(float);

  // Getting mean and variance for the first image and normalizing it
  nppiMean_StdDev_32f_C1R(this->output_first_image_.get(), image_width_in_bytes,
                          this->image_statictic_.roi_size,
                          this->image_statictic_.buffer_.get(),
                          this->image_statictic_.dev_mean.get(),
                          this->image_statictic_.dev_var.get());

  NormalizeImage(this->output_first_image_, this->image_statictic_.dev_mean,
                 this->image_statictic_.dev_var, parameters_);

  // Do the same to the second image
  nppiMean_StdDev_32f_C1R(this->output_second_image_.get(),
                          image_width_in_bytes, this->image_statictic_.roi_size,
                          this->image_statictic_.buffer_.get(),
                          this->image_statictic_.dev_mean.get(),
                          this->image_statictic_.dev_var.get());

  NormalizeImage(this->output_second_image_, this->image_statictic_.dev_mean,
                 this->image_statictic_.dev_var, parameters_);

  return *this;
}

template <typename T, typename T2>
SharedPtrGPU<T2> &PreprocessedImagesPair<T, T2>::GetFirstImage() {
  return this->output_first_image_;
}

template <typename T, typename T2>
SharedPtrGPU<T2> &PreprocessedImagesPair<T, T2>::GetSecondImage() {
  return this->output_second_image_;
}
