"""
Author: {author} || Date: {date}
Features:
"""
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST

from shared_modules.dir_handler import DirectoryHandler


class MnistDataset(Dataset):
    """ generator for the MNIST dataset"""

    def __init__(self,
                 data_dir: str = "",
                 train: bool = True,
                 transform: transforms.Compose = None) -> None:
        # set default transform (transforms.ToTensor())
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

        # generate dataset
        if train:
            DirectoryHandler.create_folder_if_not_exists(data_dir)
            self.data = MNIST(root=data_dir,
                              train=True,
                              download=True,
                              transform=transform
                              )
        else:
            DirectoryHandler.create_folder_if_not_exists(data_dir)
            self.data = MNIST(root=data_dir,
                              train=False,
                              download=True,
                              transform=transform
                              )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        return self.data[idx]

    @property
    def classes(self) -> list[str]:
        return self.data.classes


# %%
if __name__ == "__main__":
    train_data_dir = r"./datasets/mnist/test"

    # setting up transformation
    transformation = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    type(transformation)

    # creating dataset
    dataset = MnistDataset(
        data_dir=train_data_dir,
        train=False,
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
