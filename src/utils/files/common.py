import json

def read_json(json_path : str) -> dict:
    with open(json_path) as f:
        j = json.load(f)
    return j

def dump_json(json_filepath : str, dict_obj : dict) -> None:
    with open(json_filepath, 'w') as file:
        json.dump(dict_obj, file, indent=4)