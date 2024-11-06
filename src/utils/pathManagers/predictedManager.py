# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
from utils.common.pathManager import FilePath
from utils.pathManagers.defaultDictFactory import nested_defaultdict


class PredictedPathManager:

    @staticmethod
    def load_paths(data_path: FilePath) -> dict:
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
        data_path.is_dir()
        predicted_dict = nested_defaultdict(3)
        for file_name in data_path.get_files_names():
            parts = file_name.split("_")
            dis_id = parts[0]
            tile_id = parts[1]
            patch_id = parts[2]
            file_path = data_path.join(file_name)
            predicted_dict[dis_id][tile_id][patch_id] = file_path
        return dict(predicted_dict)
