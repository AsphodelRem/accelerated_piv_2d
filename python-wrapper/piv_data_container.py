import os


class DataContainer:
    def __init__(self, files: list[str]):
        self.files = files

    def get_data(self):
        return self.files

    @classmethod
    def from_video(cls, path_to_video: str):
        ...

    @classmethod
    def from_folder(cls, folder: str):
        files = os.listdir(folder)
        if len(files) % 2 != 0:
            raise ValueError('Number of files must be evenly divisible by 2')

        return DataContainer(files)

    @classmethod
    def from_list(cls, files: list[str]):
        if len(files) % 2 != 0:
            raise ValueError('Number of files must be evenly divisible by 2')

        return DataContainer(files)
