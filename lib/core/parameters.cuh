#pragma once

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

struct ImageParameters {
  unsigned int width, height, window_size, overlap;

  [[nodiscard]] unsigned int GetNumberOfWindows() const {
    return (width / window_size) * (height / window_size);
  }

  [[nodiscard]] std::pair<int, int> GetGridSize() const {
    return {(height / window_size), (width / window_size)};
  }

  [[nodiscard]] std::pair<int, int> GetSpectrumSize() const {
    return {height - (height % window_size), (width - (width % window_size)) /
                                                 window_size *
                                                 (window_size / 2 + 1)};
  }
};

struct FilterParameters {
  FilterType filter_type;
  float filter_parameter;
};

struct InterpolationParameters {
  InterpolationType interpolation_type;
};

struct VectorCorrectionsParameters {
  CorrectionType correction_type;
  int correction_parameter;
};

struct PIVParameters {
  ImageParameters image_parameters;
  FilterParameters filter_parameters;
  VectorCorrectionsParameters correction_parameters;
  InterpolationParameters interpolation_parameters;
};
