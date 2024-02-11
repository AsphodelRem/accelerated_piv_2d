import enum
import toml

import accelerated_piv_cpp as cpp_module


class Interpolations(enum.Enum):
    GAUSS = cpp_module.InterpolationType.kGaussian
    CENTROID = cpp_module.InterpolationType.kCentroid
    PARABOLIC = cpp_module.InterpolationType.kParabolic


class Filters(enum.Enum):
    NO_DC = cpp_module.FilterType.kNoDC
    LOW_PASS = cpp_module.FilterType.kLowPass
    BAND_PASS = cpp_module.FilterType.kBandPass


class Corrections(enum.Enum):
    NO_CORRECTION = cpp_module.CorrectionType.kNoCorrection
    ITERATIVE = cpp_module.CorrectionType.kMedianCorrection
    CBC = cpp_module.CorrectionType.kCorrelationBasedCorrection


class PIVConfig:
    def __init__(self, config_name: str):
        self.config_name = config_name

        self.config = cpp_module.PIVParameters()

        self.image_parameters = self.config.image_parameters
        self.filter_parameters = self.config.filter_parameters
        self.correction_parameters = self.config.correction_parameters
        self.interpolation_parameters = self.config.interpolation_parameters

    def load_from_toml(self, path_to_toml: str) -> None:
        ...

    def save_to_toml(self, path_to_toml: str = None) -> None:
        data = self._get_dict()
        file = self.config_name if path_to_toml is None else path_to_toml
        with open(file, "w") as toml_file:
            toml.dump(data, toml_file)

    def _get_dict(self) -> dict:
        if self.filter_parameters.filter_type not in Filters:
            raise Exception(f"Invalid filter type {self.filter_parameters.filter_type}")
        if self.correction_parameters.correction_type not in Corrections:
            raise Exception(f"Invalid correction type {self.correction_parameters.correction_type}")
        if self.interpolation_parameters.interpolation_type not in Interpolations:
            raise Exception(f"Invalid interpolation type {self.interpolation_parameters.interpolation_type}")

        return {
            "image parameters": {
                "width": self.image_parameters.width,
                "height": self.image_parameters.height,
                "window_size": self.image_parameters.window_size,
                "overlap": self.image_parameters.overlap,
            },
            "filter parameters": {
                "filter_type": self.filter_parameters.filter_type.value,
                "filter_parameter": self.filter_parameters.filter_parameter,
            },
            "correction parameters": {
                "correction_type": self.correction_parameters.correction_type.value,
                "correction_parameter": self.correction_parameters.correction_parameter,
            },
            "interpolation parameters": {
                "interpolation_type": self.interpolation_parameters.interpolation_type.value,
            }
        }
