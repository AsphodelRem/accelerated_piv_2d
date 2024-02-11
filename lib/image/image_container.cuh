#pragma once

#include <queue>
#include <type_traits>

#include <image/image.cuh>
#include <core/parameters.cuh>

// A wrapper over std::queue in order to be able to change it in python code
class ImagesQueue
{
public:
    ImagesQueue() = default;

    void push(const std::string& item)
    {
        data_.push(item);
    }

    void pop()
    {
        data_.pop();
    }

    std::string& front()
    {
        return data_.front();
    }

    [[nodiscard]] bool empty() const
    {
        return data_.empty();
    }

    [[nodiscard]] size_t size() const
    {
        return data_.size();
    }

private:
    std::queue<std::string> data_;
};

class IDataContainer
{
public:
    IDataContainer() = default;
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

    ImageContainer(ImagesQueue &file_names, const PIVParameters &parameters);
    ~ImageContainer() override = default;

    PreprocessedImagesPair<unsigned char, float>& GetImages() override;

    bool IsEmpty() const;

private:
    ImagePair<unsigned char> input_images_;
    PreprocessedImagesPair<unsigned char, float> output_images_;

    ImagesQueue& file_names_;

    SharedPtrGPU<float> buffer_1_, buffer_2_;
};

// class VideoContainer : IDataContainer
// {
// public:
//     explicit VideoContainer(std::string file_name);
//     ~VideoContainer() override;
//
//     PreprocessedImagesPair<unsigned char, float>& GetImages() override;
// };



