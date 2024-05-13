import os
import sys

from pandas import read_json

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from os.path import join
from utils.common.files import is_dir, is_json, is_npy
from utils.common.defaultDictFactory import nested_defaultdict

class ShardPathManager:

    def _add_original_images(self,subset,patch_dict,patch,split_dict):
        dis_id,tile_id,patch_id = patch.split("_")
        for time in ["pre","post"]:
            img_path = split_dict[subset][dis_id][tile_id][time]["image"]
            patch_dict[subset][dis_id][tile_id][patch_id][f"org_{time}"] = img_path
    
    def _add_patch_files(self,subset,sliced_dict,file,file_path):
        file_name : str = file.split(".")[0]
        dis_id,tile_id,patch_id,type = file_name.split("_")
        sliced_dict[subset][dis_id][tile_id][patch_id][type] = file_path
    
    def load_paths(self,shard_dir_path: str) -> dict:
        """
            Creates a DisasterDict that stores each file path
        """
        is_dir(shard_dir_path) 
        shards_dict = nested_defaultdict(5,str)
        for subset in ["train","val"]:
            subset_path = join(shard_dir_path,subset)
            dataset_shards = sorted(os.listdir(subset_path))
            for shard in dataset_shards:
                # assert para los datos
                shard_path = join(subset_path,shard)
                is_npy(shard_path)
                shard_name = shard.split(".")[0]
                split_id, type_id, shard_id = shard_name.split("_")
                shards_dict[split_id][type_id][shard_id] = shard_path
            
            idx_json = join(shard_dir_path,f"{subset}_shard_idxs.json")
            is_json(idx_json)
            idx_json = read_json(idx_json)
            shards_dict[subset]["idx"] = list(idx_json["shard_idxs"])
        return shards_dict