"""
Author: {author} || Date: {date}
Features:
PlayingCardDataset creator.
Classes are generated using the ImageFolder pytorch class.
Overriding the __getitem__ method allows for the return of the image and label.
Overriding the __len__ method allows for the len() function to be used on the dataset.
"""
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


class PlayingCardDataset(Dataset):
    """ generator for the playing card dataset"""

    def __init__(self,
                 data_dir: str = "",
                 transform: transforms.Compose = None) -> None:
        # check if data_dir points towards an existing directory
        if not os.path.isdir(data_dir):
            error_string = f"Directory '{data_dir}' not found."
            raise FileNotFoundError(error_string)

        # set default transform (transforms.ToTensor())
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

        # generate dataset
        try:
            self.data = ImageFolder(data_dir, transform=transform)
        except Exception as e:
            raise e

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        return self.data[idx]

    @property
    def classes(self) -> list[str]:
        return self.data.classes


# %%
if __name__ == "__main__":
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
    classes = dataset.classes
    print(classes)

    # creating dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    print(len(dataloader.dataset))
    # iterating over the dataset
    for _, (image, label) in enumerate(dataloader):
        print(image.shape)
        print(label.shape)
        break
