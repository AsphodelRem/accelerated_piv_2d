#include <functional>
#include <opencv2/videoio.hpp>
#include <optional>

#include <image/image_container.cuh>
#include <stdexcept>

IDataContainer::IDataContainer(const PIVParameters &parameters)
    : parameters_(parameters){};

ImageContainer::ImageContainer(std::deque<std::string> &file_names,
                               const PIVParameters &parameters)
    : IDataContainer(parameters)
    , file_names_(file_names)

{
  if (file_names.size() == 0 || file_names.size() % 2 != 0) {
    throw std::runtime_error("No files");
  }

  this->input_data_ = ImagePair<unsigned char>(parameters);
  this->output_data_ = PreprocessedImagesPair<unsigned char, float>(parameters);
}

std::optional<std::reference_wrapper<PreprocessedImagesPair<unsigned char, float>>>
ImageContainer::GetImages() {
  if (!this->file_names_.empty()) {
    const std::string file_1_name = this->file_names_.front();
    this->file_names_.pop_front();

    const std::string file_2_name = this->file_names_.front();
    this->file_names_.pop_front();

    this->input_data_.UploadNewImages(file_1_name, file_2_name);
    this->output_data_.UploadNewImages(this->input_data_);

    return {this->output_data_};
  }

  return std::nullopt;
}

VideoContainer::VideoContainer(std::string path_to_video_file,
                               PIVParameters &parameters)
    : IDataContainer(parameters) {
  this->video_ = cv::VideoCapture(path_to_video_file);
  if (!this->video_.isOpened()) {
    throw std::runtime_error("Unable to open " + path_to_video_file + " file!");
  }

  this->input_data_ = ImagePair<unsigned char>(parameters);
  this->output_data_ = PreprocessedImagesPair<unsigned char, float>(parameters);
}

std::optional<std::reference_wrapper<PreprocessedImagesPair<unsigned char, float>>>
VideoContainer::GetImages() {
  if (video_.read(buffer_1) && video_.read(buffer_2)) {
    input_data_.UploadNewImages(buffer_1, buffer_2);
    output_data_.UploadNewImages(input_data_);

    return { this->output_data_ };
  }
  
  return std::nullopt;
}