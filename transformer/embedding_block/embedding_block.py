"""Created by agarc the 19/12/2023.
Features:
"""

import torch
from torch import nn
from torchinfo import summary

from core.data_fetcher.data_preparer import DataPreparer

BATCH_SIZE = 32
PATCH_SIZE = 16
IMAGE_WIDTH = 224
IMAGE_HEIGHT = IMAGE_WIDTH
IMAGE_CHANNELS = 3
EMBEDDING_DIMS = IMAGE_CHANNELS * PATCH_SIZE ** 2
NUM_OF_PATCHES = int((IMAGE_WIDTH * IMAGE_HEIGHT) / PATCH_SIZE ** 2)


class PatchEmbeddingLayer(nn.Module):
    def __init__(self, in_channels, patch_size, embedding_dim) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels
        self.conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim,
                                    kernel_size=patch_size, stride=patch_size)

        self.flatten_layer = nn.Flatten(start_dim=1, end_dim=2)

        self.class_token_embeddings = nn.Parameter(
            torch.rand((BATCH_SIZE, 1, EMBEDDING_DIMS), requires_grad=True))

        self.position_embeddings = nn.Parameter(
            torch.rand((1, NUM_OF_PATCHES + 1, EMBEDDING_DIMS), requires_grad=True))

    def forward(self, x) -> torch.Tensor:
        output = torch.cat((self.class_token_embeddings,
                            self.flatten_layer(self.conv_layer(x).permute((0, 2, 3, 1)))),
                           dim=1) + self.position_embeddings
        return output


if __name__ == "__main__":
    patch_embedding_layer = PatchEmbeddingLayer(in_channels=IMAGE_CHANNELS, patch_size=PATCH_SIZE,
                                                embedding_dim=IMAGE_CHANNELS * PATCH_SIZE ** 2)
    summary(model=patch_embedding_layer,
            input_size=(BATCH_SIZE, 3, 224, 224),
            # (batch_size, input_channels, img_width, img_height)
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

    data_prepare = DataPreparer()
    data = data_prepare.create_transformed_dataloader("test")
    batch = next(iter(data))
    random_images, random_labels = batch

    patch_embeddings = patch_embedding_layer(random_images)
    print("output_size", patch_embeddings.shape)  # noqa: T201