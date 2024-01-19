#pragma once

#include <image.cuh>

class IDataContainer
{
public:
    IDataContainer() = default;
    virtual ~IDataContainer() = default;

    virtual SlicedImagePair<unsigned char, float>& getImages() = 0;

private:
    void cutImagesIntoWindows() {};
    void normalize() {};

    bool _use_run_statistic;
    int _batch_number;
};

class ImageContainer : IDataContainer
{
public:
    using ListOfFiles = std::queue<std::string>;

    ImageContainer(ListOfFiles& file_names, PIVParameters& params);
    ~ImageContainer() = default;

    SlicedImagePair<unsigned char, float>& getImages();

    bool isEmpty();

private:
    ImagePair<unsigned char> inputImages;
    SlicedImagePair<unsigned char, float> outputImages;

    ListOfFiles& _file_names;
    PIVParameters& _parameters;

    SharedPtrGPU<float> buffer_1, buffer_2;
};

class VideoContainer : IDataContainer
{
public:
    VideoContainer(std::string file_name);
    ~VideoContainer();

    SlicedImagePair<unsigned char, float>& getImages();
};