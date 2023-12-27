"""Created by agarc the 27/12/2023.
Features:
"""
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from shared_modules.dir_handler import DirectoryHandler


class MnistDataLoader:

    @staticmethod
    def get_dataloaders(batch_size: int,
                        save_path: str) -> tuple[DataLoader, DataLoader]:
        """ generate dataloaders from MNIST dataset"""
        # Create the folder if it doesn't exist
        DirectoryHandler.create_folder_if_not_exists(save_path)

        # prepare dataset and dataloader
        transform = ToTensor()
        train_set = MNIST(root=save_path,
                          train=True,
                          download=True,
                          transform=transform
                          )
        test_set = MNIST(root=save_path,
                         train=False,
                         download=True,
                         transform=transform
                         )

        train_dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
        test_dataloader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

        # Display some numbers about the dataset
        image_size = train_set[0][0].shape
        print(
            f"Dimensions of the images: {image_size} \n"
            f"Number of samples in the training set: {len(train_set)} \n"
            f"Number of batches during training: {(len(train_set) / batch_size):.1f} \n"
            f"Number of samples in the test set: {len(test_set)}"
        )
        classes = train_set.classes
        print("Classes of the dataset:", classes)
        return train_dataloader, test_dataloader
