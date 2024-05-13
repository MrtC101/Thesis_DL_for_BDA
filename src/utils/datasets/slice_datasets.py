import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.logger import get_logger
l = get_logger("delete_extra")

import cv2
from torch.utils.data import Dataset
from utils.common.files import read_json, is_json
    
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
        data = self.load_patches(disaster_id, tile_id, patch_id, patch)
        return disaster_id, tile_id, patch_id, data
    
    def load_patches(self,disaster_id, tile_id, patch_id, patch):
        data = {}
        data["pre_image"] = cv2.cvtColor(cv2.imread(patch["pre-image"]),cv2.COLOR_BGR2RGB)
        data["post_image"] = cv2.cvtColor(cv2.imread(patch["post-image"]),cv2.COLOR_BGR2RGB)
        data["pre_mask"] = cv2.imread(patch["semantic-mask"])[:,:,0]
        data["post_mask"] = cv2.imread(patch["class-mask"])[:,:,0]
        return data    
    
    @staticmethod
    def save_patches(disaster_id, tile_id, patch_list, split_folder):
        for i, patch in enumerate(patch_list):
            patch_id = f"{disaster_id}_{tile_id}_{str(i).zfill(3)}"
            patch_folder = os.path.join(split_folder, patch_id)
            os.makedirs(patch_folder, exist_ok=True)
            for key in patch.keys():
                img_name = f"{patch_id}_{key}.png"
                path = os.path.join(patch_folder, img_name)
                if(key in ['pre-image','post-img',]):
                    new_patch = cv2.cvtColor(patch[key],cv2.COLOR_RGB2BGR)
                elif(key in ['semantic-mask','class-mask']):
                    new_patch = patch[key]
                cv2.imwrite(path, new_patch)