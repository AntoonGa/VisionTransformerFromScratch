"""
Author: {author} || Date: {date}
Features:
"""
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def create_hdf5_dataset(images_path: str,
                        hdf5_save_path: str,
                        target_size: tuple[int, int] = (128, 128),
                        channels: int = 3) -> None:
    """ Create an hdf5 database file from a folder containing images.
    The images are resized to the target_size and stored in the hdf5 file.
    This is useful when your dataset cannot fit in RAM and the data consits of many images"""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])

    # check if data_dir points towards an existing directory
    if not os.path.isdir(images_path):
        error_string = f"Directory '{images_path}' not found."
        raise FileNotFoundError(error_string)

    # generate dataset
    try:
        data = ImageFolder(root=images_path, transform=transform)
        # set chunk size for loading data and appending to hdf5 file
        chunk_size = min(int(len(data) / 10), 1000)
    except Exception as e:
        raise e

    num_images = len(data)

    # open the hdf5 file
    with h5py.File(hdf5_save_path, "w") as file:
        # Create datasets with chunks for efficient storage
        img_dataset = file.create_dataset("images", shape=(num_images,
                                                           channels,
                                                           target_size[0],
                                                           target_size[1]),
                                          dtype="float32", chunks=None)
        lbl_dataset = file.create_dataset("labels", shape=(num_images,),
                                          dtype="int64", chunks=None)

        loader = DataLoader(data, batch_size=chunk_size, shuffle=False)

        current_index = 0
        # start batch processing (to avoid RAM overflow)
        for images, labels in tqdm(loader):
            chunk_size = images.size(0)

            images = np.array(images)
            labels = np.array(labels)

            img = torch.tensor(images, dtype=torch.float32)
            lbl = torch.tensor(labels, dtype=torch.long)

            img_dataset[current_index:current_index + chunk_size] = img
            lbl_dataset[current_index:current_index + chunk_size] = lbl

            current_index += chunk_size


# %%
if __name__ == "__main__":
    # Example usage
    root_folder = r"datasets/birds/train"
    hdf5_file = r"datasets/birds/train.hdf5"
    create_hdf5_dataset(root_folder, hdf5_file)

    root_folder = r"datasets/birds/test"
    hdf5_file = r"datasets/birds/test.hdf5"
    create_hdf5_dataset(root_folder, hdf5_file)

    root_folder = r"datasets/birds/valid"
    hdf5_file = r"datasets/birds/valid.hdf5"
    create_hdf5_dataset(root_folder, hdf5_file)

    root_folder = r"datasets/playing_cards/train"
    hdf5_file = r"datasets/playing_cards/train.hdf5"
    create_hdf5_dataset(root_folder, hdf5_file)

    root_folder = r"datasets/playing_cards/test"
    hdf5_file = r"datasets/playing_cards/test.hdf5"
    create_hdf5_dataset(root_folder, hdf5_file)

    root_folder = r"datasets/playing_cards/valid"
    hdf5_file = r"datasets/playing_cards/valid.hdf5"
    create_hdf5_dataset(root_folder, hdf5_file)

    # Use CustomDataset with the created HDF5 file
    # dataset = Hdf5Dataset(hdf5_save_path)
    # print(len(dataset))
    # print(dataset[0])
