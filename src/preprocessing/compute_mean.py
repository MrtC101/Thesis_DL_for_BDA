import os
import sys
if(os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.logger import get_logger
l = get_logger("delete_extra")


import json
from PIL import Image
from glob import glob
import numpy as np
import tqdm
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from skimage import transform
from torchvision import transforms
import random
from torch.utils.data import DataLoader
import argparse

class AverageMeter(object):
    """Computes and stores the average and current value
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """

        Args:
            val: mini-batch loss or accuracy value
            n: mini-batch size
        """
        self.val = val
        self.sum += val 
        self.count += n
        self.avg = self.sum / self.count

class DisasterDataset_img(Dataset):
    def __init__(self, data_dir, data_dir_ls, transform:bool, scale=1, mask_suffix=''):
        
        self.data_dir = data_dir
        self.dataset_sub_dir = data_dir_ls
        self.scale = scale
        self.transform = transform
        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

    def __len__(self):
        return len(self.dataset_sub_dir)
        
        return img_trans
    
    
    def __getitem__(self, i):
        
        imgs_dir = self.data_dir + self.dataset_sub_dir[i].replace('labels', 'images')

        idx = imgs_dir
        
        pre_img_file_name = imgs_dir + '_pre_disaster'
        pre_img_file = glob(pre_img_file_name + '.*')

        post_img_file_name = pre_img_file_name.replace('pre', 'post')
        post_img_file = glob(post_img_file_name + '.*')

        assert len(pre_img_file) == 1, \
            f'{len(pre_img_file)}? Either no image or multiple images found for the ID {idx}: {pre_img_file}'
        assert len(post_img_file) == 1, \
            f'{len(post_img_file)}? Either no post disaster image or multiple images found for the ID {idx}: {post_img_file}'

        pre_img = np.array(Image.open(pre_img_file[0]))
        post_img = np.array(Image.open(post_img_file[0]))

        assert pre_img.size == post_img.size, \
            f'Pre_ & _post disaster Images {idx} should be the same size, but are {pre_img.size} and {post_img.size}'

        return {'pre_image': pre_img, 'post_image': post_img}, {'pre_img_file_name': pre_img_file_name, 'post_img_file_name':post_img_file_name}
    
parser = argparse.ArgumentParser(
        description='Create masks for each label json file for disasters specified at the top of the script.')
parser.add_argument(
    'data_dir',
    help=('Path to the directory that contains both the `images` and `labels` folders. '
            'The `targets_border{border_width}` folder will be created if it does not already exist.')
)

args = parser.parse_args()

data_dir = args.data_dir

all_disaster_splits_json_filename = './constants/splits/final_mdl_all_disaster_splits.json'

def load_json_files(json_filename):
    with open(json_filename) as f:
        file_content = json.load(f)
    return file_content

splits = load_json_files(all_disaster_splits_json_filename)

# get list of train, val, test images
splits.keys()
all_images_ls = [] 
for item, val in splits.items():
    all_images_ls += val['train'] 
    all_images_ls += val['test'] 
    all_images_ls += val['val'] 

len(all_images_ls)
all_images_ls[1:20]

# compute mean [0,1]

mean_std_tile = {}
eps = np.finfo(float).eps
xBD_all = DisasterDataset_img(data_dir, all_images_ls, transform=True)
print('xBD_disaster_dataset length: {}'.format(len(xBD_all)))

for batch_idx, data in enumerate(xBD_all):
    print(f"{batch_idx+1}/{len(xBD_all)}",end="\r"if(batch_idx+1<len(xBD_all))else"\n")
    x_pre = data[0]['pre_image']/255.0
    x_post = data[0]['post_image']/255.0
    x_pre_filename = data[1]['pre_img_file_name'].replace('xBD', 'xBD/final_mdl_all_disaster_splits')
    x_post_filename = data[1]['post_img_file_name'].replace('xBD', 'xBD/final_mdl_all_disaster_splits')
    
    mean_std_tile[x_pre_filename]=[(x_pre[:,:,0].mean(), x_pre[:,:,1].mean(), x_pre[:,:,2].mean()), (max(eps, x_pre[:,:,0].std()), max(eps, x_pre[:,:,1].std()), max(eps, x_pre[:,:,2].std()))]
    mean_std_tile[x_post_filename]= [(x_post[:,:,0].mean(), x_post[:,:,1].mean(), x_post[:,:,2].mean()), (x_post[:,:,0].std(), x_post[:,:,1].std(), x_post[:,:,2].std())]

with open('./constants/splits/all_disaster_mean_stddev_tiles_0_1.json', 'w') as f:
    json.dump(mean_std_tile, f, indent=4)