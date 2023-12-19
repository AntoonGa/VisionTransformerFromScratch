"""Created by agarc the 19/12/2023.
Features:
"""
import torch
from torch import nn
from torchinfo import summary

from config.settings import Settings
from core.data_worker.data_worker import DataWorker
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
    def __init__(self,
                 img_size:int =224,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 num_of_patches: int = 196,
                 embedding_dims: int = 768,
                 num_transformer_layers: int = 12,  # from table 1 above
                 mlp_dropout: float = 0.1,
                 attn_dropout: float = 0.0,
                 mlp_size: int = 3072,
                 num_heads: int = 12,
                 num_classes: int = 3,
                 batch_size: int = 32) -> None:
        super().__init__()

        self.patch_embedding_layer = PatchEmbeddingBlock(channels=in_channels,
                                                         patch_size=patch_size,
                                                         num_of_patches=num_of_patches,
                                                         embedding_dim=embedding_dims,
                                                         batch_size=batch_size,
                                                         )

        self.transformer_encoder = nn.Sequential(*[TransformerBlock(embedding_dims=embedding_dims,
                                                                    mlp_dropout=mlp_dropout,
                                                                    attn_dropout=attn_dropout,
                                                                    mlp_size=mlp_size,
                                                                    num_heads=num_heads) for _ in
                                                   range(num_transformer_layers)])

        self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dims),
                                        nn.Linear(in_features=embedding_dims,
                                                  out_features=num_classes))

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.classifier(self.transformer_encoder(self.patch_embedding_layer(x))[:, 0])


if __name__ == "__main__":
    settings = Settings()
    batch_size_ = settings.training.batch_size
    vit = ViT()

    summary(model=vit,
            input_size=(batch_size_, 3, 224, 224),  # (batch_size, num_patches, embedding_dimension)
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

    data_source = "test"
    data_worker = DataWorker(data_source, "cuda")
    random_images, random_labels = next(data_worker)

    classes = vit(random_images)
    print("output_size", classes.shape)  # noqa: T201
