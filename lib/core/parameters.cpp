#include <core/parameters.hpp>

PIVParameters::PIVParameters(unsigned int width, 
                unsigned int height, 
                unsigned int window_size,
                unsigned int overlap,
                FilterType filter_type, 
                float filter_parameter,
                InterpolationType interpolation_type,
                CorrectionType correction_type,
                int correction_parameter) {

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
    .SetWindowSize(window_size)
    .SetOverlap(overlap);
}

FilterParameters::FilterParameters() 
  : filter_type_(FilterType::kNoFilter)
  , filter_parameter_(0.0f) {}

VectorCorrectionsParameters::VectorCorrectionsParameters() 
  : correction_type_(CorrectionType::kNoCorrection)
  , correction_parameter_(0) {}

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

  return {
    (height / window_size), 
    (width / window_size)
  };
}

std::pair<int, int> PIVParameters::GetSpectrumSize() const {
  auto height = image_parameters.GetHeight();
  auto width = image_parameters.GetWidth();
  auto window_size = image_parameters.GetWindowSize();

  return {
  height - (height % window_size), 
  (width - (width % window_size)) / window_size * (window_size / 2 + 1)
  };
}

ImageParameters& 
ImageParameters::SetWidth(unsigned int width) {
  if (width <= 0) {
    throw std::invalid_argument("Width of the image must be greater than 0");
  }
    this->width_ = width;
    return *this;
}

ImageParameters& 
ImageParameters::SetHeight(unsigned int height) {
  if (height <= 0) {
    throw std::invalid_argument("height of the image must be greater than 0");
  }
    this->height_ = height;
    return *this;
}

ImageParameters& 
ImageParameters::SetWindowSize(unsigned int window_size) {
  if (window_size <= 0) {
    throw std::invalid_argument("Window size must be greater than 0");
  }
    this->window_size_ = window_size;
    return *this;
}

ImageParameters& 
ImageParameters::SetOverlap(unsigned int overlap) {
  if (overlap < 0) {
    throw std::invalid_argument("Overlap must be greater than 0");
  }
    this->overlap_ = overlap;
    return *this;
}

unsigned int ImageParameters::GetWidth() const {
    return width_;
}

unsigned int ImageParameters::GetHeight() const {
    return height_;
}

unsigned int ImageParameters::GetWindowSize() const {
    return window_size_;
}

unsigned int ImageParameters::GetOverlap() const {
    return overlap_;
}

FilterParameters& 
FilterParameters::SetFilterType(FilterType type) {
    filter_type_ = type;
    return *this;
}

FilterParameters& 
FilterParameters::SetFilterParameter(float value) {
    filter_parameter_ = value;
    return *this;
}

float FilterParameters::GetFilterParameter() const {
    return filter_parameter_;
}

FilterType FilterParameters::GetFilterType() const {
    return filter_type_;
}

InterpolationParameters& 
InterpolationParameters::SetInterpolationType(InterpolationType type) {
    interpolation_type_ = type;
    return *this;
}

InterpolationType& 
InterpolationParameters::GetInterpolationType() {
    return interpolation_type_;
}

CorrectionType& 
VectorCorrectionsParameters::GetCorrectionType() {
    return correction_type_;
}

int VectorCorrectionsParameters::GetCorrectionParameter() {
    return correction_parameter_;
}

VectorCorrectionsParameters& 
VectorCorrectionsParameters::SetCorrectionType(CorrectionType type) {
    correction_type_ = type;
    return *this;
}

VectorCorrectionsParameters& 
VectorCorrectionsParameters::SetCorrectionParameter(int value) {
    correction_parameter_ = value;
    return *this;
}
