"""Created by agarc the 19/12/2023.
Features:
"""
import torch
from torch import nn
from torchinfo import summary


class MachineLearningPerceptronBlock(nn.Module):
    def __init__(self, embedding_dims, mlp_size, mlp_dropout) -> None:
        super().__init__()
        self.embedding_dims = embedding_dims
        self.mlp_size = mlp_size
        self.dropout = mlp_dropout

        self.layernorm = nn.LayerNorm(normalized_shape=embedding_dims)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dims, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dims),
            nn.Dropout(p=mlp_dropout)
        )

    def forward(self, x) -> torch.Tensor:
        return self.mlp(self.layernorm(x))


if __name__ == "__main__":
    mlp_block = MachineLearningPerceptronBlock(embedding_dims=768,
                                               mlp_size=3072,
                                               mlp_dropout=0.1)

    summary(model=mlp_block,
            input_size=(1, 197, 768),  # (batch_size, num_patches, embedding_dimension)
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

    x = torch.rand((32, 197, 768)).to("cuda")
    output = mlp_block(x)
    print("output_size", output.shape)  # noqa: T201
