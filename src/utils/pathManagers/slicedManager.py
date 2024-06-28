# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from os.path import join
from utils.common.files import is_dir, is_file, read_json
from utils.common.defaultDictFactory import nested_defaultdict


class SlicedPathManager:

    def _add_original_images(self, subset, patch_dict, patch, split_dict):
        dis_id, tile_id, patch_id = patch.split("_")
        for time in ["pre", "post"]:
            img_path = split_dict[subset][dis_id][tile_id][time]["image"]
            patch_dict[subset][dis_id][tile_id][patch_id][f"org_{time}"] = \
                img_path

    def _add_patch_files(self, subset, sliced_dict, file, file_path):
        file_name: str = file.split(".")[0]
        dis_id, tile_id, patch_id, type = file_name.split("_")
        sliced_dict[subset][dis_id][tile_id][patch_id][type.replace("-","_")] = file_path

    def load_paths(self, sliced_path: str, split_json_path: str) -> dict:
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
        is_dir(sliced_path)
        is_file(split_json_path)
        split_dict = read_json(split_json_path)
        splits = list(split_dict.keys())
        sliced_dict = nested_defaultdict(5, str)
        for subset in splits:
            subset_path = join(sliced_path, subset)
            dataset_patches = sorted(os.listdir(subset_path))
            for patch in dataset_patches:
                # assert para los datos
                patch_path = join(subset_path, patch)
                files = sorted(os.listdir(patch_path))
                for file in files:
                    file_path = join(patch_path, file)
                    is_file(file_path)
                    self._add_patch_files(subset, sliced_dict, file, file_path)
                self._add_original_images(subset, sliced_dict, patch, split_dict)
        return sliced_dict
