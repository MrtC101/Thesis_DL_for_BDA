import os
import json

def read_json(json_path : str) -> dict:
    with open(json_path) as f:
        j = json.load(f)
    return j

def dump_json(json_filepath : str, dict_obj : dict) -> None:
    with open(json_filepath, 'w') as file:
        json.dump(dict_obj, file, indent=4)

def is_dir(path) -> bool:
    assert os.path.exists(path), f"{path} do not exist."
    assert os.path.isdir(path), f"{path} is not a directory." 
    return True;

def is_file(path) -> bool:
    assert os.path.exists(path), f"{path} do not exist."
    assert os.path.isfile(path), f"{path} is not a file." 
    return True 

def is_json(path) -> bool:
    assert path.split(".")[1] == "json",f"{path} must be a json file."
    return True