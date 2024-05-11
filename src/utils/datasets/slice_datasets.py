import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.logger import get_logger
l = get_logger("delete_extra")

import cv2
from torch.utils.data import Dataset
from utils.pathManagers.rawManager import RawPathManager
from utils.common.files import read_json, is_dir, is_json

class SlicedDataset(Dataset):
   
    def __init__(self,split_name: str,splits_json_path: str):
        self.split_name = split_name
        self.splits_json_path = splits_json_path
        
        is_json(splits_json_path)
        splits_all_disasters = read_json(splits_json_path)
        self.split_name = split_name
        data = splits_all_disasters[split_name]
        
        self.tile_list = [(dis_id, tile_id, tile)
                          for dis_id in data.keys()
                          for tile_id, tile in data[dis_id].items()]

    def same_shape(self,dis_id,tile_id,img1, img2) -> bool:
        assert img1.shape == img2.shape, \
        f'Images from {dis_id}_{tile_id} should be the same size, {img1.shape} != {img2.shape}.'
        return True
    
    def __len__(self):
        return len(self.tile_list)

    def load_images(self, disaster_id, tile_id, tile):
        data = {}
        data["pre_image"] = cv2.imread(tile["pre"]["image"], cv2.COLOR_BGR2RGB)
        data["post_image"] = cv2.imread(tile["post"]["image"], cv2.COLOR_BGR2RGB)        
        pre_mask = cv2.imread(tile["pre"]["mask"], cv2.COLOR_BGR2RGB)
        post_mask = cv2.imread(tile["post"]["mask"], cv2.COLOR_BGR2RGB)
        data["pre_mask"]  = cv2.merge([pre_mask] * 3)
        data["post_mask"]  = cv2.merge([post_mask] * 3)

        self.same_shape(disaster_id,tile_id,data["pre_image"], data["post_image"])
        self.same_shape(disaster_id,tile_id,data["post_image"], data["pre_mask"])
        self.same_shape(disaster_id,tile_id,data["post_image"], data["post_mask"])
        return data
    
    def __getitem__(self, i):
        disaster_id, tile_id, tile = self.tile_list[i]
        data = self.load_images(disaster_id, tile_id, tile)
        return disaster_id, tile_id, data
    
class PatchDataset(Dataset):
   
    def __init__(self,split_name: str,splits_json_path: str):
        self.split_name = split_name
        self.splits_json_path = splits_json_path
        
        is_json(splits_json_path)
        splits_all_disasters = read_json(splits_json_path)
        self.split_name = split_name
        data = splits_all_disasters[split_name]
        
        self.tile_list = [(dis_id, tile_id, patch_id, files)
                          for dis_id in data.keys()
                          for tile_id in data[dis_id].keys()
                          for patch_id, files in data[dis_id][tile_id].items()]
   
    def __len__(self):
        return len(self.tile_list)
    
    def __getitem__(self, i):
        disaster_id, tile_id, patch_id, patch = self.tile_list[i]
        data = {}
        data["pre_image"] = cv2.imread(patch["pre-image"], cv2.COLOR_BGR2RGB)
        data["post_image"] = cv2.imread(patch["post-image"], cv2.COLOR_BGR2RGB)        
        data["pre_mask"] = cv2.imread(patch["semantic-mask"], cv2.COLOR_BGR2RGB)
        data["post_mask"] = cv2.imread(patch["class-mask"], cv2.COLOR_BGR2RGB)
        return disaster_id, tile_id, patch_id, data