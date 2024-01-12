"""
Author: {author} || Date: {date}
Features:
efficient net b0 pretrained model
"""

import timm
import torch
from torch import nn


class EfficientNetb0(nn.Module):
    """ efficient net b0 pretrained model with a linear classifier output"""

    def __init__(self,
                 num_classes: int = 53,
                 pretrained: bool = True) -> None:
        super(__class__, self).__init__()
        # Where we define all the parts of the model
        self.base_model = timm.create_model("efficientnet_b0", pretrained=pretrained)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280

        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output
