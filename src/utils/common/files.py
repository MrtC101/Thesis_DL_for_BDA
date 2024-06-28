import os
import json
import threading

lock = threading.Lock()

def read_json(json_path: str) -> dict:
    with lock:
        with open(json_path) as f:
            j = json.load(f)
    return j


def dump_json(json_filepath: str, dict_obj: dict) -> None:
    with open(json_filepath, 'w') as file:
        json.dump(dict_obj, file, indent=4)


def is_dir(path) -> bool:
    assert os.path.exists(path), f"{path} do not exist."
    assert os.path.isdir(path), f"{path} is not a directory."
    return True


def is_file(path: str) -> bool:
    assert os.path.exists(path), f"{path} do not exist."
    assert os.path.isfile(path), f"{path} is not a file."
    return True


def is_json(path) -> bool:
    is_file(path)
    assert path.split(".")[1] == "json", f"{path} must be a json file."
    return True


def is_npy(path) -> bool:
    is_file(path)
    assert path.split(".")[1] == "npy", f"{path} must be a npy file."
    return True


def clean_folder(output_path: str, split_name: str) -> None:
    """
        Deletes previous data in folder and a new creates folder for new \
        data created by this pipeline iteration.
    """
    split_folder = os.path.join(output_path, split_name)
    if (os.path.exists(split_folder) and os.path.isdir(split_folder)):
        os.system(f"rm -rf {split_folder}")
    os.makedirs(split_folder, exist_ok=True)
