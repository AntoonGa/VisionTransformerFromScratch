"""
Author: {author} || Date: {date}
Features:
"""
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from tqdm import tqdm

from shared_modules.dir_handler import DirectoryHandler


class MnistDataset(Dataset):
    """ generator for the MNIST dataset"""

    def __init__(self,
                 data_dir: str = "",
                 train: bool = True,
                 transform: transforms.Compose = None,
                 preload: bool = True) -> None:
        # set default transform (transforms.ToTensor())
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

        # generate dataset
        if train:
            DirectoryHandler.create_folder_if_not_exists(data_dir)
            self.data_ = MNIST(root=data_dir,
                               train=True,
                               download=True,
                               transform=transform
                               )
        else:
            DirectoryHandler.create_folder_if_not_exists(data_dir)
            self.data_ = MNIST(root=data_dir,
                               train=False,
                               download=True,
                               transform=transform
                               )
        # set classes
        self.cls = self.data_.classes
        # preload data in a list if specified
        if preload:
            self.data = [self.data_[i] for i in range(len(self.data_))]
        else:
            self.data = self.data_

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        return self.data[idx]

    @property
    def classes(self) -> list[str]:
        return self.cls


# %%
if __name__ == "__main__":
    import time
    train_data_dir = r"./datasets/mnist/train"

    batch_size = 32 * 4
    preload = False
    if preload:
        num_workers = 0
        prefetch_factor = None
        persistent_workers = False
    else:
        num_workers = 6
        prefetch_factor = 10
        persistent_workers = True

    # setting up transformation
    transformation = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # creating dataset
    dataset = MnistDataset(
        data_dir=train_data_dir,
        train=True,
        transform=transformation,
        preload=preload
    )
    classes = dataset.classes
    print(classes)

    # creating dataloader
    dataloader = DataLoader(dataset,
                            num_workers=num_workers,
                            persistent_workers=persistent_workers,
                            prefetch_factor=prefetch_factor,
                            batch_size=batch_size,
                            shuffle=True)

    t = time.time()
    # iterating over the dataset
    for _ in range(2):
        for _, data in tqdm(enumerate(dataloader)):
            images = data[0].to("cuda")
            labels = data[1].to("cuda")
    print(time.time() - t)
