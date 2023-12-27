"""Created by agarc the 27/12/2023.
Features:
"""
import os


class DirectoryHandler:
    @staticmethod
    def create_folder_if_not_exists(dir_path: str) -> None:
        """Create the folder if it doesn't exist"""

        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
                print(f"Folder '{dir_path}' created successfully.")
            except OSError as e:
                print(f"Error creating folder '{dir_path}': {e}")
        else:
            print(f"Folder '{dir_path}' already exists.")
