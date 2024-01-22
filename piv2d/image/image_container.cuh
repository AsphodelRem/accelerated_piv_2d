#pragma once

#include <queue>

#include <image.cuh>
#include <parameters.cuh>

class IDataContainer
{
public:
    explicit IDataContainer(const PIVParameters& parameters)
        : parameters_(parameters) {};

    virtual ~IDataContainer() = default;

    virtual SlicedImagePair<unsigned char, float>& GetImages() = 0;

private:
    // void cutImagesIntoWindows() {};
    // void normalize() {};

    // bool use_run_statistic_;
    // int batch_number_;

    const PIVParameters& parameters_;
};

class ImageContainer : IDataContainer
{
public:
    using ListOfFiles = std::queue<std::string>;

    ImageContainer(ListOfFiles &file_names, const PIVParameters &parameters);
    ~ImageContainer() = default;

    SlicedImagePair<unsigned char, float>& GetImages() override;

    bool IsEmpty() const;

private:
    ImagePair<unsigned char> input_images_;
    SlicedImagePair<unsigned char, float> output_images_;

    ListOfFiles& file_names_;

    SharedPtrGPU<float> buffer_1_, buffer_2_;
};

class VideoContainer : IDataContainer
{
public:
    VideoContainer(std::string file_name);
    ~VideoContainer();

    SlicedImagePair<unsigned char, float>& GetImages();
};