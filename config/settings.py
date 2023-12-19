"""Created by agarc the 03/12/2023.
Features:
loads .env file and sets environment variables. .env file should contain secrets only
load config.yaml and sets environment variables. config.yaml should contain public config only
"""
import os

from dotenv import load_dotenv

from config.load_config import load_config


class AzureStorageSettings:
    @property
    def connection_string(self) -> str:
        return os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

    @property
    def container_name(self) -> str:
        return os.environ.get("AZURE_STORAGE_CONTAINER_NAME")


class DBSettings:
    @property
    def db_name(self) -> str:
        return os.environ.get("DB_NAME")

    @property
    def user_name(self) -> str:
        return os.environ.get("DB_USER_NAME")

    @property
    def password(self) -> str:
        return os.environ.get("DB_PWD")

    @property
    def host(self) -> str:
        return os.environ.get("DB_HOST")

    @property
    def port(self) -> str:
        return os.environ.get("DB_PORT")

    @property
    def migration_path(self) -> str:
        return os.environ.get("DB_MIGRATION_PATH")


class StreamlitSettings:
    @property
    def welcome_message(self) -> str:
        return os.environ.get("STREAMLIT_OPTIONS_WELCOME_MESSAGE")

    @property
    def color_theme(self) -> str:
        return os.environ.get("STREAMLIT_OPTIONS_COLOR_THEME")

    @property
    def logo_path(self) -> str:
        return os.environ.get("STREAMLIT_IMAGES_LOGO_PATH")

    @property
    def favicon(self) -> str:
        return os.environ.get("STREAMLIT_IMAGES_FAVICON")


class Settings:
    def __init__(
        self, _env_path: str = None, _config_path: str = "./config/config.YAML"
    ) -> None:
        # load .env (for local secrets only, connection strings, db passwords etc.)
        load_dotenv(_env_path)
        # load .yaml config and push to env variables
        load_config(_config_path)

    @property
    def azure_storage(self) -> AzureStorageSettings:
        return AzureStorageSettings()

    @property
    def db(self) -> DBSettings:
        return DBSettings()

    @property
    def streamlit(self) -> StreamlitSettings:
        return StreamlitSettings()


if __name__ == "__main__":  # pragma: no cover
    env_path = "tests/test_data/.env_tests"
    config_path = "tests/test_data/config_tests.YAML"
    settings = Settings(env_path, config_path)
    print(settings.azure_storage.connection_string)  # noqa: T201
    print(settings.azure_storage.container_name)  # noqa: T201
    print(settings.db.db_name)  # noqa: T201
    print(settings.db.user_name)  # noqa: T201
    print(settings.db.password)  # noqa: T201
    print(settings.db.host)  # noqa: T201
    print(settings.db.port)  # noqa: T201
    print(settings.db.migration_path)  # noqa: T201
    print(settings.streamlit.welcome_message)  # noqa: T201
    print(settings.streamlit.color_theme)  # noqa: T201
    print(settings.streamlit.logo_path)  # noqa: T201
    print(settings.streamlit.favicon)  # noqa: T201
