import enum
import toml


@enum.unique
class Interpolations(enum.Enum):
    GAUSS = "gauss",
    CENTROID = "centroid",
    PARABOLIC = "parabolic"


@enum.unique
class Filters(enum.Enum):
    NO_DC = "no_dc",
    LOW_PASS = "low_pass",
    BAND_PASS = "band_pass"


@enum.unique
class Corrections(enum.Enum):
    NO_CORRECTION = "no_correction",
    ITERATIVE = "iterative",
    CBC = "cbc"


class PIVParameters:
    def __init__(self):
        self.image_parameters = self.ImageParameters()
        self.filter_parameters = self.FilterParameters()
        self.correction_parameters = self.VectorCorrectionParameters()
        self.interpolation_parameters = self.InterpolationParameters()

    def load_from_toml(self, path_to_toml: str) -> None:
        ...

    def save_to_toml(self, path_to_toml: str) -> None:
        data = self._get_dict()
        with open(path_to_toml, "w") as toml_file:
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

    class ImageParameters:
        def __init__(self):
            self.width = 0
            self.height = 0
            self.window_size = 0
            self.overlap = 0

    class InterpolationParameters:
        def __init__(self):
            self.interpolation_type = Interpolations.GAUSS

    class FilterParameters:
        def __init__(self):
            self.filter_type = Filters.NO_DC
            self.filter_parameter = 0

    class VectorCorrectionParameters:
        def __init__(self):
            self.correction_type = Corrections.NO_CORRECTION
            self.correction_parameter = 0
