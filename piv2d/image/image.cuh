#pragma once

#include <string>
#include <queue>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <device_smart_pointer.hpp>
#include <image_statistic.cuh>
#include <parameters.cuh>

static __global__ void preprocessImage_kernel(
    const uchar *deviceImage,
    float *deviceSlicedImage,
    unsigned int gridSizeX,
    unsigned int gridSizeY,
    unsigned int sizeOfSegment,
    unsigned int imageWidth,
    float topHat = 0.0)
{
    unsigned int j = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int i = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int windowYIdx = blockIdx.z;

    unsigned long long offsetX = 0, offsetY = 0;

    if (windowYIdx >= gridSizeY)
        return;
    if (i >= sizeOfSegment || j >= sizeOfSegment)
        return;

    for (int windowXIdx = 0; windowXIdx < gridSizeX; windowXIdx++)
    {
        offsetX = j + windowXIdx * sizeOfSegment;
        offsetY = i + windowYIdx * sizeOfSegment;

        float value = 0.0;
        if (((topHat * sizeOfSegment) < i < (sizeOfSegment - topHat * sizeOfSegment)) &&
            ((topHat * sizeOfSegment) < j < (sizeOfSegment - topHat * sizeOfSegment)))
        {
            value = deviceImage[offsetY * imageWidth + offsetX];
        }

        deviceSlicedImage[(windowYIdx * gridSizeX + windowXIdx) * sizeOfSegment * sizeOfSegment +
                          (i * sizeOfSegment + j)] = value;
    }
}

static void
preprocessImage(SharedPtrGPU<unsigned char> &grayImage,
                SharedPtrGPU<float> &slicedImage,
                PIVParameters &params,
                cudaStream_t stream = 0,
                float topHat = 0.0)
{
    int window_size = params.image_params.window_size;
    int grid_size_x = params.image_params.width / window_size;
    int grid_size_y = params.image_params.height / window_size;
    int image_width = params.image_params.width;

    dim3 gridSize = {window_size / 16, window_size / 16, grid_size_y};
    dim3 threadPerBlock = {16, 16};

    preprocessImage_kernel<<<gridSize, threadPerBlock, 0, stream>>>(grayImage.get(),
                                                                    slicedImage.get(),
                                                                    grid_size_x,
                                                                    grid_size_y,
                                                                    window_size,
                                                                    image_width);
}

static __global__ void rgbToGray_kernel(unsigned char *rgbImage, unsigned char *grayImage, int width, int height, int channels)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        unsigned int rgbOffset = y * width * channels + x * channels;
        unsigned int grayOffset = y * width + x;

        uchar r = rgbImage[rgbOffset];
        uchar g = rgbImage[rgbOffset + 1];
        uchar b = rgbImage[rgbOffset + 2];

        grayImage[grayOffset] = static_cast<uchar>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

static void
makeGrayScale(SharedPtrGPU<unsigned char> &rgbImage, SharedPtrGPU<unsigned char> &grayImage, int width, int height, int channels)
{
    dim3 grid = {static_cast<unsigned int>(width / 32), static_cast<unsigned int>(height / 32)};
    dim3 threadPerBlock = {32, 32};

    rgbToGray_kernel<<<grid, threadPerBlock>>>(rgbImage.get(), grayImage.get(), width, height, channels);
}

template <typename T>
class ImagePair
{
public:
    ImagePair();
    ImagePair(std::string image_1, std::string image_2);

    ImagePair<T> &uploadNewImages(std::string image_1, std::string image_2);
    ImagePair<T> &uploadNewImages(cv::Mat &image_1, cv::Mat &image_2);

    SharedPtrGPU<T> &getFirstImage();
    SharedPtrGPU<T> &getSecondImage();

private:
    SharedPtrGPU<T> image_a, image_b;
    SharedPtrGPU<T> buffer_1, buffer_2;

    void _loadData(cv::Mat &image_1, cv::Mat &image_2);
};

template <typename T, typename T2>
class SlicedImagePair
{
public:
    SlicedImagePair();
    SlicedImagePair(ImagePair<T> &inputImages, PIVParameters &params);

    SlicedImagePair<T, T2> &
    uploadNewImages(ImagePair<T> &newImages);

    SharedPtrGPU<T2> &getFirstImage();
    SharedPtrGPU<T2> &getSecondImage();

private:
    SharedPtrGPU<T2> outputFirstImage, outputSecondImage;

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

    unsigned int imageWidth = image_a.cols,
                 imageHeight = image_b.rows;

    unsigned int channels = image_a.channels();

    unsigned long long int size = imageWidth * imageHeight * channels;

    // Creating buffer for storing rgb image
    this->buffer_1 = make_shared_gpu<unsigned char>(size).uploadHostData(image_a.data, size);
    this->buffer_2 = make_shared_gpu<unsigned char>(size).uploadHostData(image_b.data, size);

    this->image_a = make_shared_gpu<unsigned char>(imageWidth * imageHeight);
    this->image_b = make_shared_gpu<unsigned char>(imageWidth * imageHeight);

    if (channels != 1)
    {
        makeGrayScale(this->buffer_1, this->image_a, imageWidth, imageHeight, channels);
        makeGrayScale(this->buffer_2, this->image_b, imageWidth, imageHeight, channels);
    }
}
template <typename T>
ImagePair<T> &ImagePair<T>::uploadNewImages(std::string image_1, std::string image_2)
{
    cv::Mat image_a = cv::imread(image_1);
    cv::Mat image_b = cv::imread(image_2);

    this->_loadData(image_a, image_b);

    return *this;
}

template <typename T>
ImagePair<T> &ImagePair<T>::uploadNewImages(cv::Mat &image_1, cv::Mat &image_2)
{
    this->_loadData(image_1, image_2);

    return *this;
}

template <typename T>
void ImagePair<T>::_loadData(cv::Mat &image_1, cv::Mat &image_2)
{
    if (image_1.cols != image_2.cols || image_1.rows != image_2.rows)
    {
        throw std::runtime_error("Images should have same shape");
    }

    unsigned int imageWidth = image_1.cols, imageHeight = image_1.rows;
    unsigned int channels = image_1.channels(), size = imageHeight * imageWidth * channels;

    this->buffer_1.uploadHostData(image_1.data, size);
    this->buffer_2.uploadHostData(image_2.data, size);

    if (channels != 1)
    {
        makeGrayScale(this->buffer_1, this->image_a, imageWidth, imageHeight, channels);
        makeGrayScale(this->buffer_2, this->image_b, imageWidth, imageHeight, channels);
    }
}

template <typename T>
SharedPtrGPU<T> &ImagePair<T>::getFirstImage()
{
    return this->image_a;
}

template <typename T>
SharedPtrGPU<T> &ImagePair<T>::getSecondImage()
{
    return this->image_b;
}

template <typename T, typename T2>
SlicedImagePair<T, T2>::SlicedImagePair() = default;

template <typename T, typename T2>
SlicedImagePair<T, T2>::SlicedImagePair(ImagePair<T> &inputImages, PIVParameters &params) : parameters_(params)
{
    int grid_size_x = params.image_params.width / params.image_params.window_size;
    int grid_size_y = params.image_params.height / params.image_params.window_size;

    this->outputFirstImage = make_shared_gpu<T2>(inputImages.getFirstImage().size() / sizeof(T));
    this->outputSecondImage = make_shared_gpu<T2>(inputImages.getFirstImage().size() / sizeof(T));

    this->uploadNewImages(inputImages);
}

template <typename T, typename T2>
SlicedImagePair<T, T2> &SlicedImagePair<T, T2>::uploadNewImages(ImagePair<T> &newImages)
{
    auto firstImage = newImages.getFirstImage();
    auto secondImage = newImages.getSecondImage();

    preprocessImage(firstImage, this->outputFirstImage, this->parameters_);
    preprocessImage(secondImage, this->outputSecondImage, this->parameters_);
}

template <typename T, typename T2>
SharedPtrGPU<T2> &SlicedImagePair<T, T2>::getFirstImage()
{
    return this->outputFirstImage;
}

template <typename T, typename T2>
SharedPtrGPU<T2> &SlicedImagePair<T, T2>::getSecondImage()
{
    return this->outputSecondImage;
}