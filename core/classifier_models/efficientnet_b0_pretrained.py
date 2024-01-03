"""
Author: {author} || Date: {date}
Features:
efficient net b0 pretrained model
"""

import timm
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from core.dataloaders.playing_cards import PlayingCardDataset


class SimpleCardClassifierEfNetb0(nn.Module):
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


# %%
if __name__ == "__main__":
    # generate dataset
    train_data_dir = r"./datasets/playing_cards/train"

    # setting up transformation
    transformation = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    type(transformation)

    # creating dataset
    dataset = PlayingCardDataset(
        data_dir=train_data_dir,
        transform=transformation
    )

    # creating dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    # get a batch of data
    images, labels = next(iter(dataloader))
    # get input size
    input_size = images.shape
    print(input_size)

    # generate model
    model = SimpleCardClassifierEfNetb0(num_classes=53)
    summary(model=model,
            input_size=input_size,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

    # single forward pass
    preds = model(images.to("cuda"))
    print(preds.shape)
    print(preds[0])

    #classification
    cls = model.classify(preds)
    print(cls.shape)
    print(cls[0])
    #accuracy
    acc = model.accuracy(preds, labels.to("cuda"))
    print(acc)
