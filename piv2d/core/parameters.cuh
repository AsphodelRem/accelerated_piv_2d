#pragma once

enum class InterpolationType
{
    kGaussian = 0,
    kParabolic,
    kCentroid,
};

enum class FilterType
{
    kLowPass = 0,
    kBandPass,
    kNoDC,
    kNoFilter,
};

enum class CorrectionType
{
    kCorrelationBasedCorrection = 0,
    kMedianCorrection,
    kNoCorrection,
};

struct ImageParameters
{
    unsigned int width, height,
        window_size,
        overlap;

    unsigned int GetNumberOfWindows() const
    {
        return (width / window_size) * (height / window_size);
    }

    std::pair<int, int> GetGridSize() const
    {
        return {(height / window_size), (width / window_size)};
    }
};

struct FilterParameters
{
    FilterType filter_type;
    int filter_parameter;
};

struct InterpolationParameters
{
    InterpolationType interpolation_type;
};

struct VectorCorrectionsParameters
{
    CorrectionType correction_type;
    int correction_parameter;
};

struct PIVParameters
{
    ImageParameters image_parameters;
    FilterParameters filter_parameters;
    VectorCorrectionsParameters correction_parameters;
    InterpolationParameters interpolation_parameters;
};
