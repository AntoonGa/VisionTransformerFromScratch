"""Created by agarc the 27/12/2023.
Features:
"""
import torch
from torchinfo import summary
from vit_pytorch import SimpleViT


class SimpleViTConfigurator:
    """ Generate SimpleVit models and hyperparameters
    Ref: https://github.com/lucidrains/vit-pytorch """
    @staticmethod
    def get_vit_config(batch_size: int = 1024,
                       image_size: tuple = (1, 28, 28),
                       channels: int = 1,
                       patch_size: int = 7,
                       num_classes: int = 10,
                       dim: int = 256,
                       depth: int = 6,
                       heads: int = 8,
                       mlp_dim: int = 512,
                       lr: float = 1e-4
                       ) -> tuple[SimpleViT, torch.optim, torch.nn.CrossEntropyLoss, torch.device]:
        # prepare Vit model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = SimpleViT(
            image_size=image_size[-1],
            channels=channels,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim
        ).to(device)

        input_size = (batch_size,) + image_size
        summary(model=model,
                input_size=input_size,
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"])

        # Setting up training parameters
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=lr
                                     )
        loss_fn = torch.nn.CrossEntropyLoss()

        return model, optimizer, loss_fn, device
