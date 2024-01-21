#pragma once

enum InterpolationType
{
    kGaussian = 0,
    kParabolic,
    kCentroid,
};

enum FilterType
{
    kLowPass = 0,
    kBandPass,
    kNoDC,
    kNoFilter,
};

enum CorrectionType
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
    int filter_type, filter_parameter;
};

struct InterpolationParameters
{
    int interpolation_type;
};

struct VectorCorrectionsParameters
{
    int correction_type;
    int correction_parameter;
};

struct PIVParameters
{
    ImageParameters image_parameters;
    FilterParameters filter_parameters;
    VectorCorrectionsParameters correction_parameters;
    InterpolationParameters interpolation_parameters;
};
