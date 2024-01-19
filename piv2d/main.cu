// Just for debugging

#include <iostream>
#include <memory>

#include <nvtx3/nvToolsExt.h>

#include <utils/device_smart_pointer.hpp>
#include <image.cuh>
#include <image_container.cuh>
#include <core/parameters.cuh>
#include <piv2d.cuh>

std::vector<Point2D<int>>
getCentersOfSegments(PIVParameters &params)
{
    auto size_x = params.image_params.width / params.image_params.window_size;
    auto size_y = params.image_params.height / params.image_params.window_size;
    auto window_size = params.image_params.window_size;
    auto overlap = 0;

    std::vector<Point2D<int>> _data(size_x * size_y);

    for (int i = 0; i < size_y; i++)
    {
        for (int j = 0; j < size_x; j++)
        {
            _data[i * size_x + j].x =
                static_cast<float>(j * (window_size - overlap) + (window_size / 2));
            _data[i * size_x + j].y =
                static_cast<float>(i * (window_size - overlap) + (window_size / 2));
        }
    }

    return _data;
}

int main()
{
    std::queue<std::string> list;
    list.push("//home/asphodel/Code/piv/piv/data/simulationPixels0.bmp");
    list.push("/home/asphodel/Code/piv/piv/data/simulationPixels1.bmp");

    ImageParameters img_params = {500, 1000, 16, 0};
    FilterParameters filter_params = {0, 0};
    InterpolationParameters inter_params = {InterpolationType::kGaussian};
    VectorCorrectionsParameters vec_params = {0, 0};

    PIVParameters params = {img_params, filter_params, vec_params, inter_params};

    ImageContainer test(list, params);

    auto data = startPIV2D(test, params);

    cv::Mat VectorField = cv::imread("/home/asphodel/Code/piv/piv/data/simulationPixels0.bmp");

    auto grid_size = img_params.GetGridSize();
    int &x_size = grid_size.second;
    int &y_size = grid_size.first;

    auto res = data.host_data_;
    auto centers = getCentersOfSegments(params);

    for (int i = 0; i < y_size; i++)
    {
        for (int j = 0; j < x_size; j++)
        {
            cv::Point2d center(centers[i * x_size + j].x, centers[i * x_size + j].y);
            cv::Point2d maximum(res[i * x_size + j].x + centers[i * x_size + j].x,
                                res[i * x_size + j].y + centers[i * x_size + j].y);

            // if (res[i * x_size + j].correlation != 0) {
            cv::arrowedLine(VectorField, center, maximum, cv::Scalar(0, 255, 255), 1.5, 8, 0, 0.9);
            //}
        }
    }

    // std::cout << frameCount << std::endl;
    cv::imshow("test", VectorField);
    cv::waitKey(0);

    return 0;
}
