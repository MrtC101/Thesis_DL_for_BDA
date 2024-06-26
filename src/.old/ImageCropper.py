import math
import torch
from torch import Tensor
import numpy as np

class ImageCropper:

    """
        Class that implements an iterator that crops an image into 16 256x256 patches.
    """
    pre_img : Tensor;
    post_img : Tensor;
    dmg_mask : Tensor;
    bld_mask : Tensor;
    
    last_patch : int;
    last_i : int = 0;
    last_j : int = 0;

    def __init__(self, pre_img : Tensor, post_img : Tensor,
                  dmg_mask : Tensor, bld_mask : Tensor):
        self.pre_img = pre_img
        self.post_img = post_img
        self.dmg_mask = dmg_mask
        self.bld_mask = bld_mask
        self.last_patch = 0;
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if(self.last_patch < 16):
            x = self.last_i
            y = self.last_j
            hight = y + 256
            width = x + 256
            patches = {}
            patches["pre_img"] =  self.pre_img[:,y:hight,x:width]
            patches["post_img"] = self.post_img[:,y:hight,x:width]
            patches["dmg_mask"] = self.dmg_mask[y:hight,x:width]
            patches["bld_mask"] = self.bld_mask[y:hight,x:width]
            if(self.last_patch % 4 == 3):
                self.last_i = hight;
                self.last_j = 0;
            else:
                self.last_j = width;
            self.last_patch += 1;
            return patches
        else:
            raise StopIteration

    @staticmethod
    def mergeCrops(mask_patches : list) -> np.array:
        rows = []
        row = []
        for i, patch in enumerate(mask_patches):
            row.append(patch)
            if(i%4==3):
                line = np.hstack(row)
                rows.append(line)
                row = []
        mask = np.vstack(rows)
        return mask
