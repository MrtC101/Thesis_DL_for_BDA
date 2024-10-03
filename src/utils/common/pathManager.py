import os
import json
import yaml
import threading
import shutil
import numpy as np


class FilePath(str):
    """This class is a wrapper for os.path methods to manipulate paths during
    data access."""
    lock = threading.Lock()

    def basename(self):
        return os.path.basename(self)

    def remove(self):
        if self.is_dir():
            shutil.rmtree(self)
        elif self.is_file():
            os.remove(self)
        else:
            raise FileNotFoundError(f"{self} does not exist.")

    def must_be_file(self) -> None:
        """Check if the path is a valid file."""
        if not os.path.exists(self):
            raise FileNotFoundError(f"{self} does not exist.")
        if not os.path.isfile(self):
            raise IsADirectoryError(f"{self} is not a file.")

    def is_file(self) -> bool:
        """Check if the path is a file."""
        return os.path.isfile(self)

    def must_be_json(self) -> None:
        """Check if the file is a JSON file."""
        self.must_be_file()  # Ensure the path is a file
        if not self.is_json():
            raise ValueError(f"{self} must be a JSON file.")

    def is_json(self) -> bool:
        """Check if the file has a .json extension."""
        return self.lower().endswith(".json")

    def save_json(self, dict_obj: dict) -> None:
        """Save a dictionary to a JSON file."""
        if not self.is_json():
            raise ValueError(f"{self} is not a JSON file.")
        with open(self, 'w') as file:
            json.dump(dict_obj, file, indent=4)

    def read_json(self) -> dict:
        """Read a JSON file into a dictionary."""
        if not self.is_json():
            raise ValueError(f"{self} is not a JSON file.")
        with self.lock:
            with open(self) as file:
                return json.load(file)

    def must_be_npy(self) -> None:
        """Check if the file is a .npy file."""
        self.must_be_file()  # Ensure the path is a file
        if not self.is_npy():
            raise ValueError(f"{self} must be a .npy file.")

    def is_npy(self) -> bool:
        """Check if the file has a .npy extension."""
        return self.lower().endswith(".npy")

    def save_npy(self, array: np.ndarray) -> None:
        """Save a NumPy array to a .npy file."""
        if not self.is_npy():
            raise ValueError(f"{self} is not a .npy file.")
        np.save(self, array)

    def read_npy(self) -> np.ndarray:
        """Read a NumPy array from a .npy file."""
        if not self.is_npy():
            raise ValueError(f"{self} is not a .npy file.")
        return np.load(self)

    def must_be_yaml(self) -> None:
        """Check if the file is a YAML file."""
        self.must_be_file()  # Ensure the path is a file
        if not self.is_yaml():
            raise ValueError(f"{self} must be a YAML file.")

    def is_yaml(self) -> bool:
        """Check if the file has a .yaml extension."""
        return self.lower().endswith((".yaml", ".yml"))

    def save_yaml(self, dict_obj: dict) -> None:
        """Save a dictionary to a YAML file."""
        if not self.is_yaml():
            raise ValueError(f"{self} is not a JSON file.")
        with open(self, 'w') as file:
            yaml.dump(dict_obj, file, indent=4)

    def read_yaml(self) -> dict:
        """Read a YAML file into a dictionary."""
        if not self.is_yaml():
            raise ValueError(f"{self} is not a YAML file.")
        with self.lock:
            try:
                with open(self, 'r') as file:
                    return yaml.safe_load(file)
            except Exception as e:
                raise RuntimeError(f"Failed to read {self}: {e}")

    def must_be_dir(self):
        path = self
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist.")
        if not os.path.isdir(path):
            raise NotADirectoryError(f"{path} is not a directory.")

    def get_folder_paths(self):
        return [self.join(name) for name in os.listdir(self)
                if self.join(name).is_dir()]

    def get_folder_names(self):
        return [name for name in os.listdir(self)
                if self.join(name).is_dir()]

    def get_file_paths(self):
        return [self.join(name) for name in os.listdir(self)
                if self.join(name).is_file()]

    def get_files_names(self):
        return [name for name in os.listdir(self)
                if self.join(name).is_file()]

    def is_dir(self) -> bool:
        return os.path.isdir(self)

    def create_folder(self, delete_if_exist=False) -> 'FilePath':
        if self.is_dir() and delete_if_exist:
            self.remove()
        os.makedirs(self, exist_ok=True)
        return self

    def join(self, *kargs) -> 'FilePath':
        path = os.path.join(self, *kargs)
        return FilePath(path)

    def clean_folder(self, split_name: str) -> None:
        """
        Deletes previous data in the folder and creates a new folder for new
        data created by this pipeline iteration.

        Args:
            split_name (str): The name of the folder to be created or cleaned.
        """
        split_folder = os.path.join(self, split_name)
        if os.path.exists(split_folder) and os.path.isdir(split_folder):
            shutil.rmtree(split_folder)
        os.makedirs(split_folder, exist_ok=True)
