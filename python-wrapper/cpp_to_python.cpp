#include "core/parameters.hpp"
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <core/piv2d.cuh>
#include <image/image_container.cuh>

namespace py = pybind11;

namespace {
class PyIDataContainer final : public IDataContainer {
public:
  using IDataContainer::IDataContainer;
  using ContainerReturnValue = std::optional<
      std::reference_wrapper<PreprocessedImagesPair<unsigned char, float>>>;

  ContainerReturnValue GetImages() override {
    PYBIND11_OVERRIDE_PURE(ContainerReturnValue, IDataContainer, GetImages);
  };
};
} // namespace

PYBIND11_MODULE(accelerated_piv_cpp, m) {
  // PIV parameters
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
      .value("kCorrelationBasedCorrection",
             CorrectionType::kCorrelationBasedCorrection)
      .value("kMedianCorrection", CorrectionType::kMedianCorrection)
      .value("kNoCorrection", CorrectionType::kNoCorrection);

  py::class_<PIVParameters>(m, "PIVParameters")
      .def(py::init<unsigned int, unsigned int, unsigned int, unsigned int,
                   FilterType, float, InterpolationType, CorrectionType, int>(),
                   py::arg("width"), py::arg("height"), py::arg("window_size"), py::arg("overlap") = 0,
                   py::arg("filter_type")=FilterType::kNoFilter, py::arg("filter_parameter")=0.0f, 
                   py::arg("interpolation_type") = InterpolationType::kGaussian,
                   py::arg("correction_type")=CorrectionType::kNoCorrection, py::arg("correction_parameter")=0);

  py::class_<IDataContainer, PyIDataContainer>(m, "IDataContainer")
      .def(py::init<const PIVParameters &>())
      .def("GetImages", &IDataContainer::GetImages);

  py::class_<ImageContainer, IDataContainer>(m, "ImageContainer")
      .def(py::init<std::deque<std::string> &, const PIVParameters &>());

  m.def("start_piv_2d",
        [](IDataContainer &container, PIVParameters &parameters) {
          StartPIV2D(container, parameters);
        });
}