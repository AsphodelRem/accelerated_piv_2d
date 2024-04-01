#pragma once

#include <stdexcept>
#include <string>

enum class InterpolationType {
  kGaussian = 0,
  kParabolic,
  kCentroid,
};

enum class FilterType {
  kLowPass = 0,
  kBandPass,
  kNoDC,
  kNoFilter,
};

enum class CorrectionType {
  kCorrelationBasedCorrection = 0,
  kMedianCorrection,
  kNoCorrection,
};

class ImageParameters {
public:
  ImageParameters() = default;

  ImageParameters &SetWidth(unsigned int width);
  ImageParameters &SetHeight(unsigned int width);
  ImageParameters &SetWindowSize(unsigned int width);
  ImageParameters &SetOverlap(unsigned int width);
  ImageParameters &SetChannels(unsigned int channels);

  unsigned int GetWidth() const;
  unsigned int GetHeight() const;
  unsigned int GetWindowSize() const;
  unsigned int GetOverlap() const;
  unsigned int GetChannels() const;

private:
  unsigned int width_, height_, window_size_, overlap_, channels_;
};

class FilterParameters {
public:
  FilterParameters();

  FilterParameters &SetFilterType(FilterType type);
  FilterParameters &SetFilterParameter(float value);

  float GetFilterParameter() const;
  FilterType GetFilterType() const;

private:
  FilterType filter_type_;
  float filter_parameter_;
};

class InterpolationParameters {
public:
  InterpolationParameters();

  InterpolationParameters &SetInterpolationType(InterpolationType type);
  InterpolationType &GetInterpolationType();

private:
  InterpolationType interpolation_type_;
};

class VectorCorrectionsParameters {
public:
  VectorCorrectionsParameters();

  CorrectionType &GetCorrectionType();
  int GetCorrectionParameter();

  VectorCorrectionsParameters &SetCorrectionType(CorrectionType type);
  VectorCorrectionsParameters &SetCorrectionParameter(int value);

private:
  CorrectionType correction_type_;
  int correction_parameter_;
};

class PIVParameters {
public:
  PIVParameters() = default;

  PIVParameters(
      unsigned int width, unsigned int height,
      unsigned int channels, unsigned int window_size,
      unsigned int overlap = 0, 
      FilterType filter_type = FilterType::kNoFilter, 
      float filter_parameter = 0.0f,
      InterpolationType interpolation_type = InterpolationType::kGaussian,
      CorrectionType correction_type = CorrectionType::kNoCorrection,
      int correction_parameter = 0);

  PIVParameters(std::string &path_to_toml_config);

  void LoadFromToml(const std::string &path_to_toml);
  void SaveToToml(const std::string &path_to_toml);

  [[nodiscard]] unsigned int GetNumberOfWindows() const;

  [[nodiscard]] std::pair<int, int> GetGridSize() const;

  [[nodiscard]] std::pair<int, int> GetSpectrumSize() const;

  ImageParameters image_parameters;
  FilterParameters filter_parameters;
  VectorCorrectionsParameters correction_parameters;
  InterpolationParameters interpolation_parameters;
};
