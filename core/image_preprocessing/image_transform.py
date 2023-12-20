"""Created by agarc the 19/12/2023.
Features:
"""
from typing import Any

import torch
from torchvision.transforms import Compose, Resize, ToTensor


class ImageTransformer:
    def __init__(self) -> None:
        self.target_x = 224
        self.target_y = 224

    def resize(self, data_in: torch.Tensor) -> Any:  # noqa: ANN401
        data_out = Compose([Resize((self.target_x, self.target_y)), ToTensor()])(data_in)
        return data_out
