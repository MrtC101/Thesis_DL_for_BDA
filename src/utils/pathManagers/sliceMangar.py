import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from os.path import join
from utils.common.files import is_dir, is_file, read_json
from utils.common.defaultDictFactory import nested_defaultdict

class SlicePathManager:

    def __init__(self,folder_path):
        is_dir(folder_path)
        self.folder_path = folder_path

    def _add_original_images(self,subset,patch_dict,patch,split_dict):
        dis_id,tile_id,patch_id = patch.split("_")[0]
        for time in ["pre","post"]:
            img_path = split_dict[subset][dis_id][tile_id][time]["img"]
            patch_dict[subset][dis_id][tile_id][patch_id]["org_pre"] = img_path
    
    def _add_patch_files(self,subset,sliced_dict,file,file_path):
        file_name : str = file.split(".")[0]
        dis_id,tile_id,patch_id,type = file_name.split("_")
        sliced_dict[subset][dis_id][tile_id][patch_id][type] = file_path
    
    def load_paths(self,sliced_path: str,split_json_path: str) -> dict:
        """
            Creates a DisasterDict that stores each file path
        """
        is_dir(sliced_path)
        is_file(split_json_path)
        split_dict = read_json(split_json_path)

        dataset_subsets = os.listdir(sliced_path)
        sliced_dict = nested_defaultdict(4,str)
        for subset in dataset_subsets:
            subset_path = join(sliced_path,subset)
            dataset_patches = os.listdir(subset_path)
            for patch in dataset_patches:
                # assert para los datos
                patch_path = join(subset_path,patch)
                files = os.listdir(patch_path)
                for file in files:
                    file_path = join(patch_path,file)
                    is_file(file_path)
                    self._add_patch_files(subset,sliced_dict,file,file_path)
                self._add_original_images(subset,sliced_dict,patch,split_dict)
        return sliced_dict
    