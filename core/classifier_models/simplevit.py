"""
Author: {author} || Date: {date}
Features:
"""
import torch
from torch import nn
from vit_pytorch import SimpleViT


class SimpleVitClassifier(nn.Module):
    """ Simple Vit Classifier"""
    def __init__(self,
                 image_size: tuple = (1, 28, 28),
                 channels: int = 1,
                 patch_size: int = 7,
                 num_classes: int = 10,
                 dim: int = 256,
                 depth: int = 6,
                 heads: int = 8,
                 mlp_dim: int = 512,
                 ) -> None:
        super(__class__, self).__init__()

        # prepare Vit model
        self.base_model = SimpleViT(
            image_size=image_size[-1],
            channels=channels,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Connect these parts and return the output
        output = self.base_model(x)
        return output


# %%
if __name__ == "__main__":
    pass
