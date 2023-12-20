"""Created by agarc the 19/12/2023.
Features:
"""
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, ImageFolder

from core.image_preprocessing.image_transform import ImageTransformer


class DataPreparer:
    """ Class to prepare the data loaders from the data folder"""

    def __init__(self, data_path: str = "data", batch_size: int = 32) -> None:
        self.datadir = Path(data_path)
        self.image_transformer = ImageTransformer()
        self.batch_size = batch_size

    def transformed_dataloader(self, train_test_tag: str = "train") -> DataLoader | None:
        """ create a dataset loader from a given directory"""
        dataloader = None
        data = self._create_transformed_datafolder(train_test_tag)
        if train_test_tag == "train":
            # Create the training dataloader using DataLoader
            dataloader = DataLoader(
                dataset=data,
                shuffle=True,
                batch_size=self.batch_size,
                num_workers=0
            )
        elif train_test_tag == "test":
            # Create the test dataloader using DataLoader
            dataloader = DataLoader(
                dataset=data,
                shuffle=False,
                batch_size=self.batch_size,
                num_workers=0
            )

        return dataloader

    def _create_transformed_datafolder(self, train_test_tag: str = "train") -> DatasetFolder | None:
        """ create a data folder from a given directory
        transformation are applied here for now. This is not the best place to do it"""
        if train_test_tag != "train" and train_test_tag != "test":
            log_string = "train_test_tag must be either train or test"
            raise ValueError(log_string)

        data_dir = str(self.datadir / train_test_tag)
        try:
            training_dataset = ImageFolder(root=data_dir, transform=self.image_transformer.resize)
            return training_dataset
        except FileNotFoundError as e:
            log_string = f"Directory {data_dir} not found"
            raise log_string from e
        except Exception as e:
            log_string = f"Error while creating dataset from {data_dir}"
            raise log_string from e


if __name__ == "__main__":
    data_prepare = DataPreparer("data", 32)
    loader = data_prepare.transformed_dataloader("test")
