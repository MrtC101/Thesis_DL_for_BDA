import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.logger import get_logger
l = get_logger("delete_extra")

import cv2
import torch
from torch.utils.data import Dataset
from utils.pathManagers.rawManager import RawPathManager
from utils.common.files import read_json, is_dir, is_json
from typing import Optional
    
class RawDataset(Dataset):
    
    def __init__(self, raw_path : Optional[str] = None,
                 split_name: Optional[str] = None,
                 splits_json_path: Optional[str] = None):
        assert \
        raw_path is not None and split_name is None and splits_json_path is None \
        or \
        raw_path is None and split_name is not None and splits_json_path is not None, \
        f"Constructor overloading, (raw_path) or (split_name,split_json_path) only allowed."

        if(raw_path is not None):
            data = RawPathManager.load_paths(raw_path)
        else:
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

    def __getitem__(self, i):

        disaster_id, tile_id, tile = self.tile_list[i]

        pre_img = cv2.imread(tile["pre"]["image"], cv2.COLOR_BGR2RGB)
        post_img = cv2.imread(tile["post"]["image"], cv2.COLOR_BGR2RGB)
        post_json = read_json(tile["post"]["json"])
        pre_json = read_json(tile["pre"]["json"])
        pre_mask = cv2.imread(tile["pre"]["mask"], cv2.COLOR_BGR2RGB)
        post_mask = cv2.imread(tile["post"]["mask"], cv2.COLOR_BGR2RGB)
        pre_mask = cv2.merge([pre_mask] * 3)
        post_mask = cv2.merge([post_mask] * 3)

        self.same_shape(disaster_id,tile_id,pre_img, post_img)
        self.same_shape(disaster_id,tile_id,pre_img, pre_mask)
        self.same_shape(disaster_id,tile_id,post_img, post_mask)
        
        return disaster_id, tile_id,{
                'pre_image': pre_img,
                'post_image': post_img,
                'semantic_mask': pre_mask,
                'class_mask': post_mask,
                "pre_json": pre_json,
                "post_json": post_json
            }