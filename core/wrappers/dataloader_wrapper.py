"""
Author: {author} || Date: {date}
Features:
"""
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from core.custom_dataloaders.multiepoch_dataloader import MultiEpochsDataLoader
from core.custom_datasets.hdf5_dataset import Hdf5Dataset
from core.custom_datasets.images_dataset import ImageDataset


def dataloader_wrapper(data_dir: str,
                       data_file: str,
                       dataset_type: str = "image",
                       transformation: transforms.Compose = None,
                       preload: bool = False,
                       dataloader_type: str = "multiepoch",
                       batch_size: int = 32,
                       shuffle: bool = False,
                       num_workers: int = 0,
                       prefetch_factor: int = None,
                       persistent_workers: bool = False) -> DataLoader:
    """ Create a dataloader from a dataset using the specified:
    with the specified parameters
    This enables creating the loaders in the same processes as the wrapper fit() method

    This loader enables you to choose multiple types of datasets (image, hdf5, mnist)
    and dataloaders (dataloader, multiepoch).

    Note: preload is always fastest (regardless of database format or dataloaders)
    hdf5 is faster than image
    multiepoch is faster than dataloader
    ! preloading means the data is PRE TRANSFORMED and will remain identical for each epoch !

    args:
        data_dir: str
            The path to the directory containing the dataset
        data_file: str
            The name of the data file (for hdf5 datasets)
        dataset_type: str
            The type of dataset object to create
        is_train: bool
            Whether the dataset is for training or not (for mnist)
        transformation: transforms.Compose
            The torch.compose transformation to apply to the dataset
        preload: bool
            Whether to preload the dataset or not (for ImageDataset)
        dataloader_type: str
            The type of dataloader to create
        batch_size: int
            The batch size
        shuffle: bool
            Whether to shuffle the dataset or not
        num_workers: int
            The number of workers to use
        prefetch_factor: int
            The prefetch factor
        persistent_workers: bool
            Whether to use persistent workers or not"""
    # sanity checks
    if preload:
        num_workers = 0
    if num_workers == 0:
        prefetch_factor = None
        persistent_workers = False

    allowed_ld = {
        "dataloader": DataLoader,
        "multiepoch": MultiEpochsDataLoader,
    }
    if dataloader_type not in allowed_ld:
        msg = (f"Dataloader type <{dataloader_type}> not supported "
               f"please use 'dataloader' or 'multiepoch'")
        raise ValueError(msg)

    # creating dataset from available datasets in core/custom_datasets
    if dataset_type == "hdf5":
        data_file = data_dir + "/" + data_file
        dataset = Hdf5Dataset(data_file=data_file,
                              transform=transformation,
                              preload=preload)
    elif dataset_type == "image":
        dataset = ImageDataset(data_dir=data_dir,
                               transform=transformation,
                               preload=preload)
    else:
        msg = (f"Dataset type <{dataset_type}> not supported "
               f"please use 'hdf5' or 'image'")
        raise ValueError(msg)

    # creating loader from available dataloaders in core/custom_dataloaders
    loader = allowed_ld[dataloader_type](dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=num_workers,
                                         prefetch_factor=prefetch_factor,
                                         persistent_workers=persistent_workers)

    # creating loader
    return loader


# %%
if __name__ == "__main__":
    pass
