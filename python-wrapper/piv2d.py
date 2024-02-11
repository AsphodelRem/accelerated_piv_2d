from piv_parameters import PIVConfig
from piv_data_container import DataContainer
import accelerated_piv_cpp as piv


class PIV2D:
    def __init__(self, parameters: PIVConfig):
        self.parameters = parameters

    def compute(self, data_container: DataContainer):
        ...


