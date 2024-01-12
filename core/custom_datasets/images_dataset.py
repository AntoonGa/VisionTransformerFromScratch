"""
Author: {author} || Date: {date}
Features:
"""
import os

import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class ImageDataset(Dataset):
    """ generator for the an image-based dataset
    Images and labels are stored in a folder structure
    Use "preload=True" to load the entire dataset in memory"""

    def __init__(self,
                 data_dir: str = "",
                 transform: transforms.Compose = None,
                 preload: bool = False) -> None:

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
        self.data_ = ImageFolder(data_dir, transform=transform)
        # set classes
        self.cls = self.data_.classes
        # preload data in a list if specified, this can take up a lot of memory
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

    train_data_dir = r"./datasets/birds/train"

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
        transforms.ToImage(),
        # transforms.ToDtype(torch.float32, scale=True),
        # transforms.Resize((128, 128), antialias=True),
    ])

    # creating dataset
    dataset = ImageDataset(
        data_dir=train_data_dir,
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
