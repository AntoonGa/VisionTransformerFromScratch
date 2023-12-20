"""Created by agarc the 03/12/2023.
Features:
loads .env file and sets environment variables. .env file should contain secrets only
load config.yaml and sets environment variables. config.yaml should contain public config only
TODO: changing to a class based on dictionnary
"""
import os

from config.load_config import load_config


class Image:
    @property
    def width(self) -> int:
        return int(os.environ.get("IMAGE_WIDTH"))

    @property
    def height(self) -> int:
        return int(os.environ.get("IMAGE_HEIGHT"))

    @property
    def channels(self) -> int:
        return int(os.environ.get("IMAGE_CHANNELS"))


class Training:
    @property
    def batch_size(self) -> int:
        return int(os.environ.get("TRAINING_BATCH_SIZE"))


class Encoder:

    @property
    def patch_size(self) -> int:
        return int(os.environ.get("ENCODER_PATCH_SIZE"))

    @property
    def embedding_dims(self) -> int:
        c = int(os.environ.get("IMAGE_CHANNELS"))
        p = int(os.environ.get("ENCODER_PATCH_SIZE"))
        a = int(c * p ** 2)
        return a

    @property
    def num_patches(self) -> int:
        h = int(os.environ.get("IMAGE_HEIGHT"))
        w = int(os.environ.get("IMAGE_WIDTH"))
        p = int(os.environ.get("ENCODER_PATCH_SIZE"))
        a = int((h * w) / p ** 2)
        return a


class Settings:
    def __init__(
            self, _env_path: str = None, _config_path: str = "./config/config.YAML"
    ) -> None:
        # load .env (for local secrets only, connection strings, db passwords etc.)
        # load .yaml config and push to env variables
        load_config(_config_path)

    @property
    def image(self) -> Image:
        return Image()

    @property
    def training(self) -> Training:
        return Training()

    @property
    def encoder(self) -> Encoder:
        return Encoder()


if __name__ == "__main__":
    settings = Settings()
    print(settings.image.width)
    print(settings.image.height)
    print(settings.image.channels)
    print(settings.training.batch_size)
    print(settings.encoder.patch_size)
    print(settings.encoder.embedding_dims)
    print(settings.encoder.num_patches)
