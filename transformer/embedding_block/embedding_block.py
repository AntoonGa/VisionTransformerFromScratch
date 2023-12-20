"""Created by agarc the 19/12/2023.
Features:
"""

import torch
from torch import nn
from torchinfo import summary

from core.data_fetcher.data_worker import DataWorker


class PatchEmbeddingBlock(nn.Module):

    def __init__(self,
                 channels: int,
                 patch_size: int,
                 num_of_patches: int,
                 embedding_dim: int,
                 batch_size: int,
                 ) -> None:
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.num_of_patches = num_of_patches
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        self.conv_layer = nn.Conv2d(in_channels=self.channels,
                                    out_channels=self.embedding_dim,
                                    kernel_size=self.patch_size,
                                    stride=self.patch_size)

        self.flatten_layer = nn.Flatten(start_dim=1, end_dim=2)

        self.class_token_embeddings = nn.Parameter(
            torch.rand((self.batch_size,
                        1,
                        self.embedding_dim),
                       requires_grad=True))

        self.position_embeddings = nn.Parameter(
            torch.rand((1, self.num_of_patches + 1, self.embedding_dim), requires_grad=True))

    def forward(self, x) -> torch.Tensor:
        output = torch.cat((self.class_token_embeddings,
                            self.flatten_layer(self.conv_layer(x).permute((0, 2, 3, 1)))),
                           dim=1) + self.position_embeddings
        return output


if __name__ == "__main__":
    im_size = 224
    channels_ = 3
    patch_size_ = 16
    num_of_patches_ = 196
    embedding_dim_ = 768
    batch_size_ = 32

    patch_embedding_layer = PatchEmbeddingBlock(channels_,
                                                patch_size_,
                                                num_of_patches_,
                                                embedding_dim_,
                                                batch_size_)
    summary(model=patch_embedding_layer,
            input_size=(batch_size_, channels_, im_size, im_size),
            # (batch_size, input_channels, img_width, img_height)
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

    # grab data using the worker
    data_worker_config = {"data_path": "data",
                          "train_test_tag": "train",
                          "batch_size": 32,
                          "device": "cuda"}

    data_worker = DataWorker(data_worker_config)
    random_images, random_labels = next(data_worker)

    patch_embeddings = patch_embedding_layer(random_images)
    print("output_size", patch_embeddings.shape)  # noqa: T201
