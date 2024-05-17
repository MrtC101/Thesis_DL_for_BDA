import math
import numpy as np
from numpy import memmap
from utils.common.files import read_json
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import torch
import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

class ShardDataset(Dataset):
    """
    Accesses data using shard_splits.json file.

    This dataset class provides access to data stored in shards, as defined by a shard_splits.json file. 
    It enables loading and retrieval of images and corresponding masks from shard files efficiently. 
    """

    def __init__(self, split_name, split_json_name):
        self.paths = read_json(split_json_name)[split_name]
        self.shard_i = "000"
        self.shard_size = self.paths["idx"][0][1] - self.paths["idx"][0][0]
        self.refer_to_shard()

    def __len__(self):
        return self.shard_size

    def refer_to_shard(self):
        """
        Load shard data and set up memory-mapped arrays.
        """
        shape_rgb = (self.shard_size, 256, 256, 3)
        shape_gray = (self.shard_size, 256, 256)
        mode = "r+"
        sh_i = self.shard_i
        self.shard = {
            "pre_img": memmap(self.paths["pre-image"][sh_i], dtype='float64', mode=mode, shape=shape_rgb , offset=128),
            "post_img": memmap(self.paths["post-image"][sh_i], dtype='float64', mode=mode, shape=shape_rgb, offset=128),
            "bld_mask": memmap(self.paths["semantic-mask"][sh_i], dtype='uint8', mode=mode, shape=shape_gray, offset=128),
            "dmg_mask": memmap(self.paths["class-mask"][sh_i], dtype='uint8', mode=mode, shape=shape_gray, offset=128),
            "pre_org": memmap(self.paths["pre-orig"][sh_i], dtype='uint8', mode=mode, shape=shape_rgb, offset=128),
            "post_org": memmap(self.paths["post-orig"][sh_i], dtype='uint8', mode=mode, shape=shape_rgb, offset=128),
        }

    def get_shard(self, i):
        req_shard = 0
        j = 0
        size = 0
        for shard_i, segment in enumerate(self.paths["idx"]):
            start, end = segment
            if (start <= i and i < end):
                req_shard = shard_i
                j -= start
                size = end-start
                break
        req_shard = str(req_shard).zfill(3)
        if (req_shard != self.shard_i):
            self.shard_i = req_shard
            self.shard_size = size
            self.refer_to_shard(req_shard)
        return self.shard, j

    def __getitem__(self, i):

        shard, j = self.get_shard(i)
        pre_img = shard["pre_img"][j, :, :, :]
        post_img = shard["post_img"][j, :, :, :]
        bld_mask = shard["bld_mask"][j, :, :]
        dmg_mask = shard["dmg_mask"][j, :, :]
        pre_org = shard["pre_org"][j, :, :, :]
        post_org = shard["post_org"][j, :, :, :]

        pre_img = torch.from_numpy(pre_img).permute(dims=(2,0,1)).type(torch.FloatTensor)
        post_img = torch.from_numpy(post_img).permute(dims=(2,0,1)).type(torch.FloatTensor)
        bld_mask = torch.from_numpy(bld_mask).type(torch.LongTensor)
        dmg_mask = torch.from_numpy(dmg_mask).type(torch.LongTensor)
        pre_org = transforms.ToTensor()(pre_org)
        post_org = transforms.ToTensor()(post_org)

        return {
            'pre_image': pre_img,
            'post_image': post_img,
            'building_mask': bld_mask,
            'damage_mask': dmg_mask,
            'pre_image_orig': pre_org,
            'post_image_orig': post_org
        }

    def get_sample_images(self, num_chips_to_viz):
        """
        Get a deterministic set of images in the specified set (train or val) by using the dataset and
        not the dataloader. Only works if the dataset is not IterableDataset.

        Returns:
            samples: a list of 'num_chips_to_viz` for visualization.
        """
        num_to_skip = 1  # first few chips might be mostly blank
        assert len(self) > num_to_skip + num_chips_to_viz

        keep_every = math.floor((len(self) - num_to_skip) / num_chips_to_viz)
        samples_idx_list = []

        for sample_idx in range(num_to_skip, len(self), keep_every):
            samples_idx_list.append(sample_idx)

        return samples_idx_list
    
    @staticmethod
    def save_shard(image_patches,out_path,log,split_name,shard_i):
        for file_id, patch_list in image_patches.items():
            shard = np.stack(patch_list, axis=0)
            shard_path = os.path.join(out_path, f'{split_name}_{file_id}_{str(shard_i).zfill(3)}.npy')
            np.save(shard_path, shard, allow_pickle=False)
            log.info(f'Shape of last added shard {file_id} to {f"{split_name}_shard"} list is {shard.shape}, dtype is {shard.dtype}.')
        del shard