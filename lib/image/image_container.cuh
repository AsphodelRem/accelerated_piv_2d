#pragma once

#include <queue>

#include <image.cuh>
#include <parameters.cuh>

class IDataContainer
{
public:
    explicit IDataContainer(const PIVParameters& parameters);

    virtual ~IDataContainer() = default;

    virtual PreprocessedImagesPair<unsigned char, float>& GetImages() = 0;

protected:
    const PIVParameters& parameters_;
};

class ImageContainer : IDataContainer
{
public:
    using ListOfFiles = std::queue<std::string>;

    ImageContainer(ListOfFiles &file_names, const PIVParameters &parameters);
    ~ImageContainer() = default;

    PreprocessedImagesPair<unsigned char, float>& GetImages() override;

    bool IsEmpty() const;

private:
    ImagePair<unsigned char> input_images_;
    PreprocessedImagesPair<unsigned char, float> output_images_;

    ListOfFiles& file_names_;

    SharedPtrGPU<float> buffer_1_, buffer_2_;
};

class VideoContainer : IDataContainer
{
public:
    VideoContainer(std::string file_name);
    ~VideoContainer();

    PreprocessedImagesPair<unsigned char, float>& GetImages();
};