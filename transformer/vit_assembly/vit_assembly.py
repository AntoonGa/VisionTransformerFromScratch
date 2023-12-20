"""Created by agarc the 19/12/2023.
Features:
"""
import torch
from torch import nn
from torchinfo import summary

from core.data_fetcher.data_worker import DataWorker
from transformer.embedding_block.embedding_block import PatchEmbeddingBlock
from transformer.transformer_assembly.transformer_assembly import TransformerBlock

BATCH_SIZE = 32
PATCH_SIZE = 16
IMAGE_WIDTH = 224
IMAGE_HEIGHT = IMAGE_WIDTH
IMAGE_CHANNELS = 3
EMBEDDING_DIMS = IMAGE_CHANNELS * PATCH_SIZE ** 2
NUM_OF_PATCHES = int((IMAGE_WIDTH * IMAGE_HEIGHT) / PATCH_SIZE ** 2)


class ViT(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.img_size = config.get("img_size")
        self.in_channels = config.get("in_channels")
        self.patch_size = config.get("patch_size")
        self.num_of_patches = config.get("num_of_patches")
        self.embedding_dims = config.get("embedding_dims")
        self.num_transformer_layers = config.get("num_transformer_layers")
        self.mlp_dropout = config.get("mlp_dropout")
        self.attn_dropout = config.get("attn_dropout")
        self.mlp_size = config.get("mlp_size")
        self.num_heads = config.get("num_heads")
        self.num_classes = config.get("num_classes")
        self.batch_size = config.get("batch_size")

        self.patch_embedding_layer = PatchEmbeddingBlock(channels=self.in_channels,
                                                         patch_size=self.patch_size,
                                                         num_of_patches=self.num_of_patches,
                                                         embedding_dim=self.embedding_dims,
                                                         batch_size=self.batch_size,
                                                         )

        self.transformer_encoder = nn.Sequential(
            *[TransformerBlock(embedding_dims=self.embedding_dims,
                               mlp_dropout=self.mlp_dropout,
                               attn_dropout=self.attn_dropout,
                               mlp_size=self.mlp_size,
                               num_heads=self.num_heads) for _ in
              range(self.num_transformer_layers)])

        self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape=self.embedding_dims),
                                        nn.Linear(in_features=self.embedding_dims,
                                                  out_features=self.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.transformer_encoder(self.patch_embedding_layer(x))[:, 0])


if __name__ == "__main__":
    batch_size = 32
    vit_config = {"img_size": 224,
                  "in_channels": 3,
                  "patch_size": 16,
                  "num_of_patches": 196,
                  "embedding_dims": 768,
                  "num_transformer_layers": 12,
                  "mlp_dropout": 0.1,
                  "attn_dropout": 0.0,
                  "mlp_size": 3072,
                  "num_heads": 12,
                  "num_classes": 3,
                  "batch_size": batch_size}
    vit = ViT(vit_config)

    summary(model=vit,
            input_size=(vit_config["batch_size"],
                        vit_config["in_channels"],
                        vit_config["img_size"],
                        vit_config["img_size"]),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

    # grab data from the worker
    data_worker_config = {"data_path": "data",
                          "train_test_tag": "train",
                          "batch_size": batch_size,
                          "device": "cuda"}

    data_worker = DataWorker(data_worker_config)
    random_images, random_labels = next(data_worker)
    print("input_size", random_images.shape)  # noqa: T201

    # generate prediction
    classes_pred = vit(random_images)
    print("output_size", classes_pred.shape)  # noqa: T201
