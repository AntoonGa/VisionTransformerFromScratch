"""
Author: {author} || Date: {date}
Features:
"""
import os

import h5py
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset


class Hdf5Dataset(Dataset):
    """ Custom dataset for loading images from an hdf5 file.
    This system lazy loads the images from the hdf5 file."""

    def __init__(self,
                 data_file: str,
                 transform: transforms.Compose = None,
                 preload: bool = False) -> None:
        self.preload = preload
        self.data_file = data_file

        # check if data_dir points towards an existing directory
        if not os.path.isfile(data_file):
            error_string = f"Directory '{data_file}' not found."
            raise FileNotFoundError(error_string)

        # set default transform (transforms.ToTensor())
        if transform is None:
            self.transform = transforms.Compose([transforms.ToImage(),
                                                 transforms.ToDtype(torch.float32, scale=True)])
        else:
            self.transform = transform

        # open the hdf5 file
        with h5py.File(self.data_file, "r") as file:
            # set iterator length
            self.length = len(file["images"])
            # Read the class and decode the byte strings to regular strings
            self.cls = [item.decode("utf-8") for item in file["classes"][:]]
            # start preloading if specified
            if self.preload:
                self.dataset = file
                # load image as a tensor
                self.imgs = [self.transform(self.dataset["images"][ii]) for ii in
                             range(self.length)]
                self.lbls = [self.dataset["labels"][ii] for ii in range(self.length)]
            else:
                self.dataset = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        """ Returns a tuple of image and label."""
        # use preloaded data if specified
        if self.preload:
            imgs = self.imgs[idx]
            lbls = self.lbls[idx]
            return imgs, lbls

        # if not preloaded, use lazy load
        if self.dataset is None:
            self.dataset = h5py.File(self.data_file, mode="r", swmr=True)
        # load image as a tensor
        imgs = self.transform(self.dataset["images"][idx])
        lbls = self.dataset["labels"][idx]
        return imgs, lbls

    @property
    def classes(self) -> list[str]:
        return self.cls


# %%
if __name__ == "__main__":
    pass
