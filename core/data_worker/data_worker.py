"""Created by agarc the 19/12/2023.
Features:
"""

from core.data_fetcher.data_preparer import DataPreparer


class DataWorker:
    """ DataWorker is a class that returns a data iterator and sends the data to the device"""
    def __init__(self, train_test_tag: str = "train", device="cuda") -> None:
        if device != "cuda" and device != "cpu":
            log_string = "device must be either cuda or cpu"
            raise ValueError(log_string)

        data_prepare = DataPreparer()
        self.data_iterator = iter(data_prepare.create_transformed_dataloader(train_test_tag))
        self.device = device

    def __iter__(self):  # noqa: ANN204
        return self

    def __next__(self):  # noqa: ANN204
        batch = next(self.data_iterator)
        random_images, random_labels = batch
        return random_images.to(self.device), random_labels.to(self.device)
