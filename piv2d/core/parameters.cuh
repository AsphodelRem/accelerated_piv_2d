#pragma once

#include <vector>

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

    unsigned int getNumberOfWindows()
    {
        return (width / window_size) * (height / window_size);
    }

    std::pair<int, int> GetGridSize()
    {
        return {(height / window_size), (width / window_size)};
    }
};

struct FilterParameters
{
    int filterType, filterParameter;
};

struct InterpolationParameters
{
    int interpolationType;
};

struct VectorCorrectionsParameters
{
    int correctionType;
    int correctionParameter;
};

struct PIVParameters
{
    ImageParameters image_params;
    FilterParameters filter_params;
    VectorCorrectionsParameters correction_params;
    InterpolationParameters interpolation_params;
};
