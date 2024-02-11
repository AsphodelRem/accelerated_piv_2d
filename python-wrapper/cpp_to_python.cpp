#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <image/image_container.cuh>
#include <core/piv2d.cuh>

namespace py = pybind11;

void StartPIV2D_wrapper(ImagesQueue& queue, PIVParameters& parameters)
{
    ImageContainer container(queue, parameters);
    StartPIV2D(container, parameters);
}

PYBIND11_MODULE(accelerated_piv_cpp, m)
{
    py::class_<ImagesQueue>(m, "ImagesQueue")
        .def(py::init<>())
        .def("push", &ImagesQueue::push)
        .def("pop", &ImagesQueue::pop)
        .def("front", &ImagesQueue::front, py::return_value_policy::reference)
        .def("empty", &ImagesQueue::empty)
        .def("size", &ImagesQueue::size);

    py::enum_<InterpolationType>(m, "InterpolationType")
        .value("kGaussian", InterpolationType::kGaussian)
        .value("kParabolic", InterpolationType::kParabolic)
        .value("kCentroid", InterpolationType::kCentroid);

    py::enum_<FilterType>(m, "FilterType")
        .value("kLowPass", FilterType::kLowPass)
        .value("kBandPass", FilterType::kBandPass)
        .value("kNoDC", FilterType::kNoDC)
        .value("kNoFilter", FilterType::kNoFilter);

    py::enum_<CorrectionType>(m, "CorrectionType")
        .value("kCorrelationBasedCorrection", CorrectionType::kCorrelationBasedCorrection)
        .value("kMedianCorrection", CorrectionType::kMedianCorrection)
        .value("kNoCorrection", CorrectionType::kNoCorrection);

    py::class_<ImageParameters>(m, "ImageParameters")
        .def(py::init<>())
        .def_readwrite("width", &ImageParameters::width)
        .def_readwrite("height", &ImageParameters::height)
        .def_readwrite("window_size", &ImageParameters::window_size)
        .def_readwrite("overlap", &ImageParameters::overlap);

    py::class_<FilterParameters>(m, "FilterParameters")
        .def(py::init<>())
        .def_readwrite("filter_type", &FilterParameters::filter_type)
        .def_readwrite("filter_parameter", &FilterParameters::filter_parameter);

    py::class_<InterpolationParameters>(m, "InterpolationParameters")
        .def(py::init<>())
        .def_readwrite("interpolation_type", &InterpolationParameters::interpolation_type);

    py::class_<VectorCorrectionsParameters>(m, "VectorCorrectionsParameters")
        .def(py::init<>())
        .def_readwrite("correction_type", &VectorCorrectionsParameters::correction_type)
        .def_readwrite("correction_parameter", &VectorCorrectionsParameters::correction_parameter);

    py::class_<PIVParameters>(m, "PIVParameters")
        .def(py::init<>())
        .def_readwrite("image_parameters", &PIVParameters::image_parameters)
        .def_readwrite("filter_parameters", &PIVParameters::filter_parameters)
        .def_readwrite("correction_parameters", &PIVParameters::correction_parameters)
        .def_readwrite("interpolation_parameters", &PIVParameters::interpolation_parameters);

    m.def("start_piv_2d", [](ImagesQueue &container, PIVParameters &parameters) {
        StartPIV2D_wrapper(container, parameters);
    });
}