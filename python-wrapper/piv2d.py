from piv_parameters import PIVParameters
from piv_data_container import DataContainer


class PIV2D:
    def __init__(self, data_container: DataContainer, parameters: PIVParameters):
        self.data_container = data_container
        self.parameters = parameters

    def compute(self):
        ...
