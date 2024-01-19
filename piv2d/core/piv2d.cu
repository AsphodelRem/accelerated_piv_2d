#include <piv2d.cuh>
#include <device_smart_pointer.hpp>
#include <math/math.cuh>
#include <parameters.cuh>

PIVDataContainer::PIVDataContainer(PIVParameters &parameters) : parameters_(parameters)
{
    auto number_of_window = parameters.image_params.getNumberOfWindows();

    this->host_data_ = std::make_shared<Point2D<float>[]>(number_of_window);
    this->preprocessed_data_ = make_shared_gpu<Point2D<float>>(number_of_window);

    this->data.reserve(number_of_window);
}

__global__ void findMovements_kernel(Point2D<float> *interpolatedCoordinates, Point2D<float> *outputSpeed, unsigned int length,
                                     float scaleFactor, float time, unsigned int segmentSize, bool toPhysicalView = false)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < length)
    {
        bool isThereMovementsByX = (interpolatedCoordinates[idx].x != 0);
        bool isThereMovementsByY = (interpolatedCoordinates[idx].y != 0);

        outputSpeed[idx].x = isThereMovementsByX * (interpolatedCoordinates[idx].x - (segmentSize / 2)) * scaleFactor / time;
        outputSpeed[idx].y = isThereMovementsByY * (interpolatedCoordinates[idx].y - (segmentSize / 2)) * scaleFactor / time;

        if (toPhysicalView)
        {
            outputSpeed[idx].y = -outputSpeed[idx].y;
        }
    }
}

void findMovements(SharedPtrGPU<Point2D<float>> &input, SharedPtrGPU<Point2D<float>> &output, PIVParameters &parameters)
{
    auto length = parameters.image_params.getNumberOfWindows();

    dim3 gridSize = {(length + 127) / 128};
    dim3 threadsPerBlock = {128};

    findMovements_kernel<<<gridSize, threadsPerBlock>>>(input.get(),
                                                        output.get(),
                                                        length,
                                                        1, 1,
                                                        parameters.image_params.window_size);
}

void PIVDataContainer::storeData(SharedPtrGPU<Point2D<float>> &data)
{
    findMovements(data, preprocessed_data_, parameters_);

    preprocessed_data_.copyDataToHost(this->host_data_.get());
}

PIVDataContainer startPIV2D(ImageContainer &container, PIVParameters &parameters)
{
    ForwardFFTHandler fourier_image_1(parameters);
    ForwardFFTHandler fourier_image_2(parameters);

    BackwardFFTHandler correlation_function(parameters);

    MultiArgMaxSearch multi_max_search(parameters);

    Interpolation interpolation(parameters);

    PIVDataContainer data(parameters);

    while (!container.isEmpty())
    {
        auto new_data = container.getImages();

        fourier_image_1.computeForwardFFT(new_data.getFirstImage(), true);
        fourier_image_2.computeForwardFFT(new_data.getSecondImage());

        fourier_image_1 *= fourier_image_2;

        correlation_function.computeBackwardFFT(fourier_image_1.result);

        multi_max_search.getMaxForAllWindows(correlation_function.result);

        interpolation.interpolate(correlation_function.result, multi_max_search.result);

        data.storeData(interpolation.result);
    }

    return data;
}
