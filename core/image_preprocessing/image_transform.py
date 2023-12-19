"""Created by agarc the 19/12/2023.
Features:
"""
from torchvision.transforms import Resize, Compose, ToTensor
from typing import Any


class Resizer:
    def __init__(self):
        self.target_x = 224
        self.target_y = 224

    def resize(self, data_in) -> Any:
        data_out = Compose([Resize((self.target_x, self.target_y)), ToTensor()])(data_in)
        return data_out



if __name__ == "__main__":
    data_downloader = DataDownloader()
    data_downloader.download_n_write()

