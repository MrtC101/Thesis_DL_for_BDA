# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
import sys

from tqdm import tqdm

from utils.common.pathManager import FilePath

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

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
        sliced_dict[subset][dis_id][tile_id][patch_id][type.replace(
            "-", "_")] = file_path

    def load_paths(self, sliced_path: FilePath,
                   split_json_path: FilePath) -> dict:
        """
            Creates a DisasterDict that stores each file path.

            This function loads file paths from a given directory structure
            and a JSON file. It verifies the existence of the paths, reads
            the JSON file to get the splits, and then organizes the file
            paths into a nested dictionary.

            Args:
                sliced_path (str): Path to the directory containing the
                sliced data.
                split_json_path (str): Path to the JSON file that contains the
                data splits.

            Returns:
                dict: A nested dictionary (DisasterDict) where each key
                represents a subset (train, val, test) and contains the file
                paths organized by patch and file type.
        """
        sliced_path.must_be_dir()
        split_json_path.must_be_json()
        split_dict = split_json_path.read_json()
        splits = list(split_dict.keys())
        sliced_dict = nested_defaultdict(5, str)
        for subset in tqdm(splits):
            subset_path = sliced_path.join(subset)
            for patch in sorted(subset_path.get_folder_names()):
                patch_folder = subset_path.join(patch)
                patch_folder.must_be_dir()
                for file in sorted(patch_folder.get_files_names()):
                    file_path = patch_folder.join(file)
                    file_path.must_be_file()
                    self._add_patch_files(subset, sliced_dict, file, file_path)
                self._add_original_images(
                    subset, sliced_dict, patch, split_dict)
        return sliced_dict
