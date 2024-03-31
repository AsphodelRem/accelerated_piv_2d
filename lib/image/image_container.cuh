#pragma once

#include <opencv2/core/mat.hpp>
#include <optional>
#include <queue>
#include <type_traits>

#include <core/parameters.hpp>
#include <image/image.cuh>

class IDataContainer {
public:
  explicit IDataContainer(const PIVParameters &parameters);

  virtual ~IDataContainer() = default;

  virtual std::optional<
      std::reference_wrapper<PreprocessedImagesPair<unsigned char, float>>>
  GetImages() = 0;

protected:
  const PIVParameters &parameters_;
};

class ImageContainer final : public IDataContainer {
public:
  ImageContainer(std::deque<std::string> &file_names,
                 const PIVParameters &parameters);

  ~ImageContainer() override = default;

  std::optional<
      std::reference_wrapper<PreprocessedImagesPair<unsigned char, float>>>
  GetImages() override;

private:
  ImagePair<unsigned char> input_images_;
  PreprocessedImagesPair<unsigned char, float> output_images_;

  std::deque<std::string> file_names_;
};

class VideoContainer final : public IDataContainer
{
public:
    VideoContainer(std::string path_to_video_file, PIVParameters& parameters);
    ~VideoContainer() override = default;

  std::optional<
      std::reference_wrapper<PreprocessedImagesPair<unsigned char, float>>>
  GetImages() override;

private:
  cv::VideoCapture video_;
  cv::Mat buffer_1, buffer_2;

  ImagePair<unsigned char> input_frames_;
  PreprocessedImagesPair<unsigned char, float> output_frames_;
};
