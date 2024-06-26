#include <core/parameters.hpp>
#include <extern/toml.hpp>

PIVParameters::PIVParameters() {
  this->image_parameters
    .SetHeight(1)
    .SetWidth(1)
    .SetChannels(1)
    .SetWindowSize(1)
    .SetOverlap(0);

  this->interpolation_parameters
    .SetInterpolationType(InterpolationType::kGaussian);

  this->filter_parameters
    .SetFilterType(FilterType::kNoFilter)
    .SetFilterParameter(0.0f);

  this->correction_parameters
    .SetCorrectionType(CorrectionType::kNoCorrection)
    .SetCorrectionParameter(0);

  this->memory_settings
    .SetSaveOnDiskMode(false)
    .SetContainerCapacity(1200);
}

PIVParameters::PIVParameters(const std::string &path_to_toml_config) {
  this->LoadFromToml(path_to_toml_config);
}

PIVParameters::PIVParameters(
  unsigned int width, unsigned int height, unsigned int channels,
  unsigned int window_size, unsigned int overlap,
  FilterType filter_type, float filter_parameter,
  InterpolationType interpolation_type,
  CorrectionType correction_type,
  int correction_parameter,
  bool to_save_on_disk, 
  unsigned int capacity) {

  this->filter_parameters
      .SetFilterParameter(filter_parameter)
      .SetFilterType(filter_type);

  this->interpolation_parameters
      .SetInterpolationType(interpolation_type);

  this->correction_parameters
      .SetCorrectionParameter(correction_parameter)
      .SetCorrectionType(correction_type);

  this->image_parameters
      .SetHeight(height)
      .SetWidth(width)
      .SetChannels(channels)
      .SetWindowSize(window_size)
      .SetOverlap(overlap);

  this->memory_settings
      .SetSaveOnDiskMode(to_save_on_disk)
      .SetContainerCapacity(capacity);
}

PIVParameters::PIVParameters(const PIVParameters& other) {
  this->image_parameters = other.image_parameters;
  this->interpolation_parameters = other.interpolation_parameters;
  this->filter_parameters = other.filter_parameters;
  this->correction_parameters = other.correction_parameters;
  this->memory_settings = other.memory_settings;
}

void PIVParameters::LoadFromToml(const std::string &path_to_toml) {
  auto config = toml::parse_file(path_to_toml);

  // Image parameters
  this->image_parameters.SetHeight(
      config["image_parameters"]["height"].value_or(0));
  this->image_parameters.SetWidth(
      config["image_parameters"]["width"].value_or(0));
  this->image_parameters.SetChannels(
    config["image_parameters"]["channels"].value_or(0));
  this->image_parameters.SetWindowSize(
      config["image_parameters"]["window_size"].value_or(0));
  this->image_parameters.SetOverlap(
      config["image_parameters"]["overlap"].value_or(0));

  // Interpolation
  this->interpolation_parameters.SetInterpolationType(
      config["interpolation"]["type"].value_or(InterpolationType::kGaussian));

  // Filter
  this->filter_parameters.SetFilterType(
      config["filter"]["type"].value_or(FilterType::kNoFilter));
  this->filter_parameters.SetFilterParameter(
      config["filter"]["parameter"].value_or(0));

  // Correction
  this->correction_parameters.SetCorrectionType(
      config["correction"]["type"].value_or(CorrectionType::kNoCorrection));
  this->correction_parameters.SetCorrectionParameter(
      config["correction"]["parameter"].value_or(0));

  // Memory management settings
  this->memory_settings.SetSaveOnDiskMode(
      config["memory_management"]["save_on_disk"].value_or(false));
  this->memory_settings.SetContainerCapacity(
      config["memory_management"]["capacity"].value_or(1200));
}

void PIVParameters::SaveToToml(const std::string &path_to_toml) {
  toml::table toml_content{
      {"image_parameters",
      toml::table{{"height", this->image_parameters.GetHeight()},
                  {"width", this->image_parameters.GetWidth()},
                  {"channels", this->image_parameters.GetChannels()},
                  {"window_size", this->image_parameters.GetWindowSize()},
                  {"overlap", this->image_parameters.GetOverlap()}}},

      {"interpolation",
      toml::table{{"type", this->interpolation_parameters.GetInterpolationType()}}},

      {"filter", 
      toml::table{{"type", this->filter_parameters.GetFilterType()},
                  {"parameter", this->filter_parameters.GetFilterParameter()}}},

      {"correction",
      toml::table{{"type", this->correction_parameters.GetCorrectionType()},
                  {"parameter", this->correction_parameters.GetCorrectionParameter()}}},

  {"memory_management",
      toml::table{{"save_on_disk", this->memory_settings.GetSaveOnDiskMode()},
                  {"capacity", this->memory_settings.GetContainerCapacity()}}}
      };

  std::ofstream toml_fout(path_to_toml);

  toml_fout << toml_content << std::endl;
}

FilterParameters::FilterParameters()
    : filter_type_(FilterType::kNoFilter), filter_parameter_(0.0f) {}

VectorCorrectionsParameters::VectorCorrectionsParameters()
    : correction_type_(CorrectionType::kNoCorrection),
      correction_parameter_(0) {}

InterpolationParameters::InterpolationParameters()
    : interpolation_type_(InterpolationType::kGaussian) {}

unsigned int PIVParameters::GetNumberOfWindows() const {
  auto height = image_parameters.GetHeight();
  auto width = image_parameters.GetWidth();
  auto window_size = image_parameters.GetWindowSize();

  return (width / window_size) * (height / window_size);
}

std::pair<int, int> PIVParameters::GetGridSize() const {
  auto height = image_parameters.GetHeight();
  auto width = image_parameters.GetWidth();
  auto window_size = image_parameters.GetWindowSize();

  return {(height / window_size), (width / window_size)};
}

std::pair<int, int> PIVParameters::GetSpectrumSize() const {
  auto height = image_parameters.GetHeight();
  auto width = image_parameters.GetWidth();
  auto window_size = image_parameters.GetWindowSize();

  return {height - (height % window_size), (width - (width % window_size)) /
                                               window_size *
                                               (window_size / 2 + 1)};
}

ImageParameters &ImageParameters::SetWidth(unsigned int width) {
  if (width <= 0) {
    throw std::invalid_argument("Width of the image must be greater than 0");
  }
  this->width_ = width;
  return *this;
}

ImageParameters &ImageParameters::SetHeight(unsigned int height) {
  if (height <= 0) {
    throw std::invalid_argument("height of the image must be greater than 0");
  }
  this->height_ = height;
  return *this;
}

ImageParameters &ImageParameters::SetWindowSize(unsigned int window_size) {
  if (window_size <= 0) {
    throw std::invalid_argument("Window size must be greater than 0");
  }
  this->window_size_ = window_size;
  return *this;
}

ImageParameters &ImageParameters::SetOverlap(unsigned int overlap) {
  if (overlap < 0) {
    throw std::invalid_argument("Overlap must be greater than 0");
  }
  this->overlap_ = overlap;
  return *this;
}

ImageParameters &ImageParameters::SetChannels(unsigned int channels) {
  if (channels <= 0) {
    throw std::invalid_argument("Number of channels must be greater than 0");
  }
  this->channels_ = channels;
  return *this;
}

unsigned int ImageParameters::GetChannels() const { return channels_; }

unsigned int ImageParameters::GetWidth() const { return width_; }

unsigned int ImageParameters::GetHeight() const { return height_; }

unsigned int ImageParameters::GetWindowSize() const { return window_size_; }

unsigned int ImageParameters::GetOverlap() const { return overlap_; }

FilterParameters &FilterParameters::SetFilterType(FilterType type) {
  filter_type_ = type;
  return *this;
}

FilterParameters &FilterParameters::SetFilterParameter(float value) {
  filter_parameter_ = value;
  return *this;
}

float FilterParameters::GetFilterParameter() const { return filter_parameter_; }

FilterType FilterParameters::GetFilterType() const { return filter_type_; }

InterpolationParameters &
InterpolationParameters::SetInterpolationType(InterpolationType type) {
  interpolation_type_ = type;
  return *this;
}

InterpolationType &InterpolationParameters::GetInterpolationType() {
  return interpolation_type_;
}

CorrectionType &VectorCorrectionsParameters::GetCorrectionType() {
  return correction_type_;
}

int VectorCorrectionsParameters::GetCorrectionParameter() {
  return correction_parameter_;
}

VectorCorrectionsParameters &
VectorCorrectionsParameters::SetCorrectionType(CorrectionType type) {
  correction_type_ = type;
  return *this;
}

VectorCorrectionsParameters &
VectorCorrectionsParameters::SetCorrectionParameter(int value) {
  correction_parameter_ = value;
  return *this;
}

MemorySettings::MemorySettings() {
  container_capacity_ = 1200;
  to_save_on_disk_ = false;
};

MemorySettings& MemorySettings::SetSaveOnDiskMode(bool mode) {
  to_save_on_disk_ = mode;
  return *this;
}

MemorySettings& MemorySettings::SetContainerCapacity(unsigned int number) {
  container_capacity_ = number;
  return *this;
}

bool MemorySettings::GetSaveOnDiskMode() const {
  return to_save_on_disk_;
}

unsigned int MemorySettings::GetContainerCapacity() const {
  return container_capacity_;
}
