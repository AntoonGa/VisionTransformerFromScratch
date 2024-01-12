"""
Author: {author} || Date: {date}
Features:
"""
import os

import h5py
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def create_hdf5_dataset(images_path: str,
                        hdf5_save_path: str,
                        target_size: tuple[int, int] = (128, 128)) -> None:
    """ Create an hdf5 database file from a folder containing images.
    The images are resized to the target_size and stored in the hdf5 file.
    This is useful when your dataset cannot fit in RAM and the data consists of many images"""
    transformation = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize(target_size, antialias=True)
    ])

    # check if data_dir points towards an existing directory
    if not os.path.isdir(images_path):
        error_string = f"Directory '{images_path}' not found."
        raise FileNotFoundError(error_string)

    # generate dataset
    try:
        data = ImageFolder(root=images_path, transform=transformation)
    except Exception as e:
        raise e
    # metadata
    channels = data[0][0].shape[0]
    classes = data.classes
    num_images = len(data)
    # set chunk size for loading data and appending to hdf5 file
    chunk_size = 1000

    # open the hdf5 file
    with h5py.File(hdf5_save_path, "w") as file:
        # Create datasets with chunks for efficient storage
        img_dataset = file.create_dataset("images", shape=(num_images,
                                                           target_size[0],
                                                           target_size[1],
                                                           channels),
                                          dtype="float32", chunks=None)
        lbl_dataset = file.create_dataset("labels", shape=num_images,
                                          dtype="int64", chunks=None)

        clc_dataset = file.create_dataset("classes",  # noqa: F841
                                          data=np.array(classes, dtype="S"))

        loader = DataLoader(data, batch_size=chunk_size, shuffle=False)

        curr_index = 0
        # start batch processing (to avoid RAM overflow)
        for images, labels in tqdm(loader):
            chunk_size = images.size(0)

            # This permute is necessary to invert the default conversion from PIL to Tensor
            imgs = images.clone().detach()
            imgs = imgs.permute(0, 2, 3, 1)

            lbls = labels.clone().detach()

            img_dataset[curr_index:curr_index + chunk_size] = imgs
            lbl_dataset[curr_index:curr_index + chunk_size] = lbls

            curr_index += chunk_size


if __name__ == "__main__":
    # root_folder = r"datasets/birds/train"
    # hdf5_file = r"datasets/birds/train/data_128x128.hdf5"
    # create_hdf5_dataset(root_folder, hdf5_file)
    #
    # root_folder = r"datasets/birds/test"
    # hdf5_file = r"datasets/birds/test/data_128x128.hdf5"
    # create_hdf5_dataset(root_folder, hdf5_file)
    #
    # root_folder = r"datasets/birds/valid"
    # hdf5_file = r"datasets/birds/valid/data_128x128.hdf5"
    # create_hdf5_dataset(root_folder, hdf5_file)
    #
    # root_folder = r"datasets/playing_cards/train"
    # hdf5_file = r"datasets/playing_cards/train/data_128x128.hdf5"
    # create_hdf5_dataset(root_folder, hdf5_file)
    #
    # root_folder = r"datasets/playing_cards/test"
    # hdf5_file = r"datasets/playing_cards/test/data_128x128.hdf5"
    # create_hdf5_dataset(root_folder, hdf5_file)
    #
    # root_folder = r"datasets/playing_cards/valid"
    # hdf5_file = r"datasets/playing_cards/valid/data_128x128.hdf5"
    # create_hdf5_dataset(root_folder, hdf5_file)
    pass
