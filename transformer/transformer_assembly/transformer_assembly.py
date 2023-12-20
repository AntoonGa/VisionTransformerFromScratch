"""Created by agarc the 19/12/2023.
Features:
"""
import torch
from torch import nn
from torchinfo import summary

from transformer.mlp_block.mlp_block import MachineLearningPerceptronBlock
from transformer.msa_block.msa_block import MultiHeadSelfAttentionBlock


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dims=768,
                 mlp_dropout=0.1,
                 attn_dropout=0.0,
                 mlp_size=3072,
                 num_heads=12,
                 ) -> None:
        super().__init__()

        self.msa_block = MultiHeadSelfAttentionBlock(embedding_dims=embedding_dims,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)

        self.mlp_block = MachineLearningPerceptronBlock(embedding_dims=embedding_dims,
                                                        mlp_size=mlp_size,
                                                        mlp_dropout=mlp_dropout,
                                                        )

    def forward(self, x) -> torch.Tensor:
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x


if __name__ == "__main__":
    transformer_block = TransformerBlock(embedding_dims=768)

    summary(model=transformer_block,
            input_size=(1, 197, 768),  # (batch_size, num_patches, embedding_dimension)
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

    x_data = torch.rand((32, 197, 768)).to("cuda")
    output = transformer_block(x_data)
    print("output_size", output.shape)  # noqa: T201
