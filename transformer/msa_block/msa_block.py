"""Created by agarc the 19/12/2023.
Features:
"""
import torch
from torch import nn
from torchinfo import summary

from config.settings import Settings


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dims: int = 768,  # Hidden Size D in the ViT Paper Table 1
                 num_heads: int = 12,  # Heads in the ViT Paper Table 1
                 attn_dropout: float = 0.0
                 # Default to Zero as there is no dropout for the the MSA Block as per the ViT Paper
                 ) -> None:
        super().__init__()

        self.embedding_dims = embedding_dims
        self.num_head = num_heads
        self.attn_dropout = attn_dropout

        self.layernorm = nn.LayerNorm(normalized_shape=embedding_dims)

        self.multihead_attention = nn.MultiheadAttention(num_heads=num_heads,
                                                         embed_dim=embedding_dims,
                                                         dropout=attn_dropout,
                                                         batch_first=True,
                                                         )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(x)
        output, _ = self.multihead_attention(query=x, key=x, value=x, need_weights=False)
        return output


if __name__ == "__main__":
    settings = Settings()
    embedding_dims_ = settings.encoder.embedding_dims
    multihead_self_attention_block = MultiHeadSelfAttentionBlock(embedding_dims=embedding_dims_)

    summary(model=multihead_self_attention_block,
            input_size=(1, 197, 768),  # (batch_size, num_patches, embedding_dimension)
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

    x = torch.rand((32, 197, 768)).to("cuda")
    msa_out = multihead_self_attention_block(x)
    print("output_size", msa_out.shape)  # noqa: T201
