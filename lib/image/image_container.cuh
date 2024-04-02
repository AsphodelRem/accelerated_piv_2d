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

  using ReturnType = std::optional<
      std::reference_wrapper<PreprocessedImagesPair<unsigned char, float>>>;

  virtual ReturnType GetImages() = 0;

protected:
  const PIVParameters &parameters_;

  ImagePair<unsigned char> input_data_;
  PreprocessedImagesPair<unsigned char, float> output_data_;
};

class ImageContainer final : public IDataContainer {
public:
  ImageContainer(std::deque<std::string> &file_names,
                 const PIVParameters &parameters);

  ~ImageContainer() override = default;

  ReturnType GetImages() override;

private:
  std::deque<std::string> file_names_;
};

class VideoContainer final : public IDataContainer {
public:
  VideoContainer(const std::string &path_to_video_file,
                 const PIVParameters &parameters);

  ~VideoContainer() override = default;

  ReturnType GetImages() override;

private:
  cv::VideoCapture video_;
  cv::Mat buffer_1, buffer_2;
};
