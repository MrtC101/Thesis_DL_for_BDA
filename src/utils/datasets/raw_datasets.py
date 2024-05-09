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

class TilesDataset(Dataset):
    """
        Dataset that uses RawPathManager to read all raw Dataset files.
    """

    def __init__(self, raw_data_path: str):
        data = RawPathManager.load_paths(raw_data_path)
        self.tile_list = [(dis_id, tile_id, tile)
                          for dis_id in data.keys()
                          for tile_id, tile in data[dis_id].items()]

    def __len__(self):
        return len(self.tile_list)

    def __getitem__(self, i):

        disaster_id, tile_id, tile = self.tile_list[i]

        pre_img = cv2.imread(tile["pre"]["image"], cv2.COLOR_BGR2RGB)
        post_img = cv2.imread(tile["post"]["image"], cv2.COLOR_BGR2RGB)
        post_json = read_json(tile["post"]["json"])
        pre_json = read_json(tile["pre"]["json"])

        return {
            "dis_id": disaster_id,
            "tile_id": tile_id,
            "pre_img": pre_img,
            "post_img": post_img,
            "pre_json": pre_json,
            "post_json": post_json
        }
    
def same_shape(dis_id,tile_id,img1, img2) -> bool:
    assert img1.shape[:2] == img2.shape[:2], \
        f'Images from {dis_id}_{tile_id} should be the same size, {img1.shape} != {img2.shape}.'
    return True

class RawDataset(Dataset):
    """
        Access Data using raw_splits.json file.
    """
    
    def __init__(self, split_name: str, splits_json_path: str):
        is_dir(splits_json_path)
        is_json(splits_json_path)
        splits_all_disasters = read_json(splits_json_path)
        # Tiles num count
        split_dict = splits_all_disasters[split_name]
        set_length = sum(len(tile) for tile in split_dict.values())
        self.split_name = split_name
        self.tile_list = [(dis_id, tile_id, tile)
                          for dis_id in split_dict.keys()
                          for tile_id, tile in split_dict[dis_id].items()]

    def __len__(self):
        return len(self.tile_list)

    def __getitem__(self, i):

        disaster_id, tile_id, tile = self.tile_list[i]

        pre_img = cv2.imread(tile["pre"]["image"], cv2.COLOR_BGR2RGB)
        post_img = cv2.imread(tile["post"]["image"], cv2.COLOR_BGR2RGB)
        # post_json = read_json(tile["post"]["json"])
        # pre_json = read_json(tile["pre"]["json"])
        pre_mask = cv2.imread(tile["pre"]["mask"], cv2.COLOR_BGR2RGB)
        post_mask = cv2.imread(tile["post"]["mask"], cv2.COLOR_BGR2RGB)

        same_shape(disaster_id,tile_id,pre_img, post_img)
        same_shape(disaster_id,tile_id,pre_img, pre_mask)
        same_shape(disaster_id,tile_id,post_img, post_mask)

        return (disaster_id, tile_id,{
            'pre_image': TF.to_tensor(pre_img),
            'post_image': TF.to_tensor(post_img),
            'semantic_mask': TF.to_tensor(pre_mask),
            'class_mask': TF.to_tensor(post_mask)
        })