"""
Author: {author} || Date: {date}
Features:
"""
import h5py
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm


class Hdf5Dataset(Dataset):
    """ Custom dataset for loading images from an hdf5 file.
    This system lazy loads the images from the hdf5 file."""

    def __init__(self, hdf5_file: str, transform: transforms = None) -> None:
        self.hdf5_file = hdf5_file
        self.transform = transform
        self.dataset = None

        with h5py.File(hdf5_file, "r") as file:
            self.length = len(file["images"])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        """ Returns a tuple of image and label."""
        if self.dataset is None:
            self.dataset = h5py.File(self.hdf5_file, mode="r", swmr=True)

        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# %%
if __name__ == "__main__":
    """ Test the hdf5 dataset. vs the image dataset"""

    root_folder = r"datasets/birds/train"
    hdf5_file = r"datasets/birds/train.hdf5"

    batch_size = 32 * 4
    epochs = 4
    num_workers = 6
    prefetch_factor = 10
    persistent_workers = True

    ### loading dataset from hdf5 ###
    print("With hdf5")
    dataset = Hdf5Dataset(hdf5_file)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             persistent_workers=persistent_workers,
                                             prefetch_factor=prefetch_factor,
                                             pin_memory=False)

    for _ in range(epochs):
        for _, (images, labels) in tqdm(enumerate(dataloader)):
            images = images.to("cuda")
            labels = labels.to("cuda")
