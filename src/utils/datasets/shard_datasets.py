import math
from numpy import memmap
from utils.common.files import read_json
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import torch
import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.logger import get_logger
l = get_logger("delete_extra")


class ShardDataset(Dataset):
    """
        Access Data using shard_splits.json file.
    """

    def __init__(self, split_name, split_json_name):
        self.paths = read_json(split_json_name)[split_name]
        self.shard_i = "000"
        self.shard_size = self.paths["idx"][0][1] - self.paths["idx"][0][0]
        self.refer_to_shard("000")

    def __len__(self):
        return len(self.pre_image_chip_shard)

    def refer_to_shard(self, shard_n):
        shard_i = str(shard_n).zfill(3)
        shape = (self.shard_size, 256, 256, 3)
        mode = "r"
        self.shard = {
            "pre_image_patches": memmap(self.paths["pre-image"][shard_i], dtype='float64', mode=mode, shape=shape),
            "post_image_patches": memmap(self.paths["post-image"][shard_i], dtype='float64', mode=mode, shape=shape),
            "bld_mask_patches": memmap(self.paths["semantic-mask"][shard_i], dtype='unit8', mode=mode, shape=shape),
            "dmg_mask_patches": memmap(self.paths["class-mask"][shard_i], dtype='unit8', mode=mode, shape=shape),
            "pre_img_orig": memmap(self.paths["pre-orig"][shard_i], dtype='unit8', mode=mode, shape=shape),
            "post_img_orig": memmap(self.paths["post-orig"][shard_i], dtype='unit8', mode=mode, shape=shape),
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
        if (req_shard != self.curr_shard):
            self.curr_shard = req_shard
            self.shard_size = size
            self.refer_to_shard(req_shard)
        return self.shard, j

    def __getitem__(self, i):

        shard, j = self.get_shard(i)
        pre_img = shard["pre_image_chips"][j, :, :, :]
        post_img = shard["post_image_chips"][j, :, :, :]
        bld_mask = shard["bld_mask_chips"][j, :, :, :]
        dmg_class = shard["dmg_mask_chips"][j, :, :, :]
        pre_image_orig = shard["pre_img_orig"][j, :, :, :]
        post_image_orig = shard["post_img_orig"][j, :, :, :]

        pre_img = torch.from_numpy(pre_img).type(torch.FloatTensor)
        post_img = torch.from_numpy(post_img).type(torch.FloatTensor)
        bld_mask = torch.from_numpy(bld_mask).type(torch.FloatTensor)
        dmg_class = torch.from_numpy(dmg_class).type(torch.FloatTensor)
        pre_image_orig = transforms.ToTensor()(pre_image_orig)
        post_image_orig = transforms.ToTensor()(post_image_orig)

        return {
            'pre_image': pre_img,
            'post_image': post_img,
            'building_mask': bld_mask,
            'damage_mask': dmg_class,
            'pre_image_orig': pre_image_orig,
            'post_image_orig': post_image_orig
        }

    def get_sample_images(self, num_chips_to_viz):
        """
        Get a deterministic set of images in the specified set (train or val) by using the dataset and
        not the dataloader. Only works if the dataset is not IterableDataset.

        Returns:
            samples: a list with `num_chips_to_viz` to visualize
        """
        num_to_skip = 1  # first few chips might be mostly blank
        assert len(self) > num_to_skip + num_chips_to_viz

        keep_every = math.floor((len(self) - num_to_skip) / num_chips_to_viz)
        samples_idx_list = []

        for sample_idx in range(num_to_skip, len(self), keep_every):
            samples_idx_list.append(sample_idx)

        return samples_idx_list
