#pragma once

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
  using ListOfFiles = std::deque<std::string>;

  ImageContainer(std::deque<std::string> &file_names,
                 const PIVParameters &parameters);

  ~ImageContainer() override = default;

  std::optional<
      std::reference_wrapper<PreprocessedImagesPair<unsigned char, float>>>
  GetImages() override;

private:
  ImagePair<unsigned char> input_images_;
  PreprocessedImagesPair<unsigned char, float> output_images_;

  std::deque<std::string> &file_names_;

  SharedPtrGPU<float> buffer_1_, buffer_2_;
};

// TODO: 
// class VideoContainer : IDataContainer
// {
// public:
//     explicit VideoContainer(std::string file_name);
//     ~VideoContainer() override;
//
//     PreprocessedImagesPair<unsigned char, float>& GetImages() override;
// };
