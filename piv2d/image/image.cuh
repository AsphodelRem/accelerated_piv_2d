#pragma once

#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <device_smart_pointer.hpp>
#include <image_statistic.cuh>
#include <parameters.cuh>


static __global__
void PreprocessImage_kernel(
    const uchar *dev_image,
    float *dev_sliced_image,
    unsigned int grid_size_x,
    unsigned int grid_size_y,
    unsigned int window_size,
    unsigned int image_width,
    float top_hat = 0.0)
{
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

    for (int window_x_idx = 0; window_x_idx < grid_size_x; window_x_idx++)
    {
        offset_x = j + window_x_idx * window_size;
        offset_y = i + window_y_idx * window_size;

        float value = 0.0;
        if (((top_hat * window_size) < i < (window_size - top_hat * window_size)) &&
            ((top_hat * window_size) < j < (window_size - top_hat * window_size)))
        {
            value = dev_image[offset_y * image_width + offset_x];
        }

        dev_sliced_image[(window_y_idx * grid_size_x + window_x_idx) * window_size * window_size +
            (i * window_size + j)] = value;
    }
}

static void
PreprocessImage(SharedPtrGPU<unsigned char> &gray_image,
                SharedPtrGPU<float> &sliced_image,
                PIVParameters &parameters,
                cudaStream_t stream = 0,
                float top_hat = 0.0)
{
    int window_size = parameters.image_parameters.window_size;
    int grid_size_x = parameters.image_parameters.width / window_size;
    int grid_size_y = parameters.image_parameters.height / window_size;
    int image_width = parameters.image_parameters.width;

    dim3 grid_size = {window_size / 16, window_size / 16, grid_size_y};
    dim3 thread_per_block = {16, 16};

    PreprocessImage_kernel<<<grid_size, thread_per_block, 0, stream>>>(gray_image.get(),
                                                                    sliced_image.get(),
                                                                    grid_size_x,
                                                                    grid_size_y,
                                                                    window_size,
                                                                    image_width);
}

static __global__
void RgbToGray_kernel(unsigned char *rgb_image, unsigned char *gray_image, int width, int height, int channels)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        unsigned int rgbOffset = y * width * channels + x * channels;
        unsigned int gray_offset = y * width + x;

        uchar r = rgb_image[rgbOffset];
        uchar g = rgb_image[rgbOffset + 1];
        uchar b = rgb_image[rgbOffset + 2];

        gray_image[gray_offset] = static_cast<uchar>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

static void
MakeGrayScale(SharedPtrGPU<unsigned char> &rgb_image, SharedPtrGPU<unsigned char> &gray_image, int width, int height, int channels)
{
    dim3 grid = {static_cast<unsigned int>(width / 32), static_cast<unsigned int>(height / 32)};
    dim3 thread_per_block = {32, 32};

    RgbToGray_kernel<<<grid, thread_per_block>>>(rgb_image.get(), gray_image.get(), width, height, channels);
}

template <typename T>
class ImagePair
{
public:
    ImagePair();
    ImagePair(std::string image_1, std::string image_2);

    ImagePair<T> &UploadNewImages(std::string image_1, std::string image_2);
    ImagePair<T> &UploadNewImages(cv::Mat &image_1, cv::Mat &image_2);

    SharedPtrGPU<T> &GetFirstImage();
    SharedPtrGPU<T> &GetSecondImage();

private:
    SharedPtrGPU<T> image_a_, image_b_;
    SharedPtrGPU<T> buffer_1_, buffer_2_;

    void LoadData_(cv::Mat &image_1, cv::Mat &image_2);
};

template <typename T, typename T2>
class SlicedImagePair
{
public:
    SlicedImagePair();
    SlicedImagePair(ImagePair<T> &input_images, const PIVParameters &parameters);

    SlicedImagePair<T, T2> &
    UploadNewImages(ImagePair<T> &new_images);

    SharedPtrGPU<T2> &GetFirstImage();
    SharedPtrGPU<T2> &GetSecondImage();

private:
    SharedPtrGPU<T2> output_first_image_, output_second_images_;
    PIVParameters parameters_;
};

template <typename T>
ImagePair<T>::ImagePair() = default;

template <typename T>
ImagePair<T>::ImagePair(std::string image_1, std::string image_2)
{
    cv::Mat image_a = cv::imread(image_1);
    cv::Mat image_b = cv::imread(image_2);

    if (image_a.cols != image_b.cols || image_a.rows != image_b.rows)
    {
        throw std::runtime_error("Images should have same shape");
    }

    unsigned int image_width = image_a.cols,
                 image_height = image_b.rows;

    unsigned int channels = image_a.channels();

    unsigned long long int size = image_width * image_height * channels;

    // Creating buffer for storing rgb image
    this->buffer_1_ = make_shared_gpu<unsigned char>(size).UploadHostData(image_a.data, size);
    this->buffer_2_ = make_shared_gpu<unsigned char>(size).UploadHostData(image_b.data, size);

    this->image_a_ = make_shared_gpu<unsigned char>(image_width * image_height);
    this->image_b_ = make_shared_gpu<unsigned char>(image_width * image_height);

    if (channels != 1)
    {
        MakeGrayScale(this->buffer_1_, this->image_a_, image_width, image_height, channels);
        MakeGrayScale(this->buffer_2_, this->image_b_, image_width, image_height, channels);
    }
}
template <typename T>
ImagePair<T> &ImagePair<T>::UploadNewImages(std::string image_1, std::string image_2)
{
    cv::Mat image_a = cv::imread(image_1);
    cv::Mat image_b = cv::imread(image_2);

    this->LoadData_(image_a, image_b);

    return *this;
}

template <typename T>
ImagePair<T> &ImagePair<T>::UploadNewImages(cv::Mat &image_1, cv::Mat &image_2)
{
    this->LoadData_(image_1, image_2);

    return *this;
}

template <typename T>
void ImagePair<T>::LoadData_(cv::Mat &image_1, cv::Mat &image_2)
{
    if (image_1.cols != image_2.cols || image_1.rows != image_2.rows)
    {
        throw std::runtime_error("Images should have same shape");
    }

    unsigned int image_width = image_1.cols;
    unsigned int image_height = image_1.rows;
    unsigned int channels = image_1.channels(), size = image_height * image_width * channels;

    this->buffer_1_.UploadHostData(image_1.data, size);
    this->buffer_2_.UploadHostData(image_2.data, size);

    if (channels != 1)
    {
        MakeGrayScale(this->buffer_1_, this->image_a_, image_width, image_height, channels);
        MakeGrayScale(this->buffer_2_, this->image_b_, image_width, image_height, channels);
    }
}

template <typename T>
SharedPtrGPU<T> &ImagePair<T>::GetFirstImage()
{
    return this->image_a_;
}

template <typename T>
SharedPtrGPU<T> &ImagePair<T>::GetSecondImage()
{
    return this->image_b_;
}

template <typename T, typename T2>
SlicedImagePair<T, T2>::SlicedImagePair() = default;

template <typename T, typename T2>
SlicedImagePair<T, T2>::SlicedImagePair(ImagePair<T> &input_images, const PIVParameters &parameters) : parameters_(parameters)
{
    this->output_first_image_ = make_shared_gpu<T2>(input_images.GetFirstImage().size() / sizeof(T));
    this->output_second_images_ = make_shared_gpu<T2>(input_images.GetFirstImage().size() / sizeof(T));

    this->UploadNewImages(input_images);
}

template <typename T, typename T2>
SlicedImagePair<T, T2> &SlicedImagePair<T, T2>::UploadNewImages(ImagePair<T> &new_images)
{
    auto first_image = new_images.GetFirstImage();
    auto second_image = new_images.GetSecondImage();

    PreprocessImage(first_image, this->output_first_image_, this->parameters_);
    PreprocessImage(second_image, this->output_second_images_, this->parameters_);

    return *this;
}

template <typename T, typename T2>
SharedPtrGPU<T2> &SlicedImagePair<T, T2>::GetFirstImage()
{
    return this->output_first_image_;
}

template <typename T, typename T2>
SharedPtrGPU<T2> &SlicedImagePair<T, T2>::GetSecondImage()
{
    return this->output_second_images_;
}