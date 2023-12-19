import logging
import os

import yaml


def load_config(config_file_path: str = "./config/config.YAML") -> dict:
    """
    Loads config YAML file and pushes the data to environment variables.
    """

    def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
        """
        Helper function to flatten nested dictionaries.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    try:
        with open(config_file_path, encoding="utf-8") as file:
            config = yaml.safe_load(file)

        flat_config = flatten_dict(config)

        # Push all signal to os.environ variables
        for key, value in flat_config.items():
            os.environ[key] = str(value)

        logging.info("config.YAML loaded and pushed to os.environ variables.")
    except FileNotFoundError as e:
        logging.info(
            "config.YAML does not exist. Nothing pushed to os.environ variables."
        )
        msg = f"{config_file_path} does not exist."
        raise FileNotFoundError(msg) from e
    except Exception as e:
        logging.exception("Error while loading config.YAML")
        msg = "Error while loading config.YAML"
        raise msg from e

    return config
