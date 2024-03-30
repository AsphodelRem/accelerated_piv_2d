#pragma once

#include <opencv2/core.hpp>

#include <core/parameters.hpp>
#include <utils/device_smart_pointer.hpp>

void SplitImageIntoWindows(const SharedPtrGPU<unsigned char> &gray_image,
                           const SharedPtrGPU<float> &sliced_image,
                           const PIVParameters &parameters);

void MakeGrayScale(const SharedPtrGPU<unsigned char> &rgb_image,
                   SharedPtrGPU<unsigned char> &gray_image, const int width,
                   const int height, const int channels);

void NormalizeImage(SharedPtrGPU<float> &input,
                    const SharedPtrGPU<double> &mean,
                    const SharedPtrGPU<double> &var,
                    const PIVParameters &parameters);
