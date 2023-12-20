"""Created by agarc the 19/12/2023.
Features:
"""

from core.data_fetcher.data_preparer import DataPreparer


class DataWorker:
    """ DataWorker is a class that returns a data iterator and sends the data to the device"""

    def __init__(self, config: dict) -> None:
        """
        :param config:
        data_worker_config = {"data_path": "data",
                              "train_test_tag": "train"/"test",
                              "batch_size": 32,
                              "device": "cuda"/"cpu"}
        """
        #place config input to self variables
        self.data_path = config.get("data_path")
        self.train_test_tag = config.get("train_test_tag")
        self.batch_size = config.get("batch_size")
        self.device = config.get("device")

        if self.device != "cuda" and self.device != "cpu":
            log_string = "device must be either cuda or cpu"
            raise ValueError(log_string)

        # data preparation
        data_prepare = DataPreparer(data_path=self.data_path,
                                    batch_size=self.batch_size)
        # create data iterator
        self.data_iterator = iter(data_prepare.transformed_dataloader(self.train_test_tag))

    def __iter__(self):  # noqa: ANN204
        return self

    def __next__(self):  # noqa: ANN204
        # grab next batch of data when calling the next() method
        batch = next(self.data_iterator)
        random_images, random_labels = batch
        # send to device
        return random_images.to(self.device), random_labels.to(self.device)
