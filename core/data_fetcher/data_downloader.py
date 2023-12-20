"""Created by agarc the 19/12/2023.
Features:
Data download
"""
import logging
import os
from zipfile import ZipFile

import requests
from requests import Response


class DataDownloader:
    """ Class to download the data from the url and write it to the data folder"""
    def __init__(self, data_url: str = None, data_path: str = None) -> None:
        if data_url is None:
            self.data_url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"

        self.data_set_name = self.data_url.split("/")[-1]

        if data_path is None:
            self.data_path = "data"

    def download_n_write(self) -> None:
        """ download the data and write all datasets to the data folder"""
        # check if the data already exists in the data folder
        if not self._check_data_exits():
            # create data folder
            self._create_data_folder()
            # download data
            response = self._download_data()
            # write downloaded content to a file
            self._write_downloaded_content(response)
            # unzip the data
            self._unzip_data()
        return

    def _create_data_folder(self) -> None:
        """ if the data folder does not exist, create it"""
        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)

    def _download_data(self) -> Response:
        """ get data from url """
        response = requests.get(url=self.data_url, timeout=5)
        logging.info("Data downloaded successfully")
        return response

    def _check_data_exits(self) -> bool:
        """ check if the data exists"""
        boolean = os.path.isfile(os.path.join(self.data_path, self.data_set_name))
        if boolean:
            logging.info("Data already exists")
        return boolean

    def _write_downloaded_content(self, response: Response) -> None:
        """ write the downloaded content to a file"""
        with open(os.path.join(self.data_path, self.data_set_name), "wb") as f:
            f.write(response.content)
        logging.info("Data downloaded successfully")

    def _unzip_data(self) -> None:
        """ unzip the data"""
        with ZipFile(os.path.join(self.data_path, self.data_set_name),
                     "r") as zip:  # noqa: A001
            zip.extractall(self.data_path)
        logging.info("Data unzipped successfully")


if __name__ == "__main__":
    data_downloader = DataDownloader()
    data_downloader.download_n_write()
