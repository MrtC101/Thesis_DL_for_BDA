# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
from collections import defaultdict
import os
import sys

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from os.path import join
from utils.common.files import is_dir, is_file, read_json
from utils.common.defaultDictFactory import nested_defaultdict

class PredictedPathManager:

    @staticmethod
    def load_paths(data_path: str) -> dict:
        """
            Creates a DisasterDict that stores each file path.

            This function loads file paths from a given directory structure and a JSON file.
            It verifies the existence of the paths, reads the JSON file to get the splits,
            and then organizes the file paths into a nested dictionary.

            Args:
                sliced_path (str): Path to the directory containing the sliced data.
                split_json_path (str): Path to the JSON file that contains the data splits.

            Returns:
                dict: A nested dictionary (DisasterDict) where each key represents a subset
                    (train, val, test) and contains the file paths organized by patch and file type.
        """
        is_dir(data_path)
        predicted_dict = nested_defaultdict(3)
        for file_name in os.listdir(data_path):
            parts = file_name.split("_")
            dis_id = parts[0]
            tile_id = parts[1]
            patch_id = parts[2]
            file_path = os.path.join(data_path, file_name)
            predicted_dict[dis_id][tile_id][patch_id]=file_path 
        return dict(predicted_dict)
