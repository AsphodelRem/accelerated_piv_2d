import piv_parameters
from piv_data_container import DataContainer
import accelerated_piv_cpp as piv


class PIV2D:
    def __init__(self, parameters: piv_parameters.PIVConfig):
        self.parameters = parameters

    def compute(self, data_container: DataContainer):
        piv.start_piv_2d(data_container.img_queue, self.parameters.config)
