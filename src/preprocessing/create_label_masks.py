# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
create_label_mask.py

For each label json file, flat in one directory, outputs a 2D raster of labels.
Have to run this for different root directories containing `labels` and `images` folders.

Manually fill out the disaster name (prefix to file names) in DISASTERS_OF_INTEREST at the top of the script.
Masks will be generated for these disasters only.

Sample invocation:
```
python data/create_label_masks.py /home/lynx/data -b 2
```

This script borrows code and functions from
https://github.com/DIUx-xView/xView2_baseline/blob/master/utils/mask_polygons.py
Below is their copyright statement:
"""
#####################################################################################################################################################################
# xView2                                                                                                                                                            #
# Copyright 2019 Carnegie Mellon University.                                                                                                                        #
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO    #
# WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY,          #
# EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, #
# TRADEMARK, OR COPYRIGHT INFRINGEMENT.                                                                                                                             #
# Released under a MIT (SEI)-style license, please see LICENSE.md or contact permission@sei.cmu.edu for full terms.                                                 #
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use  #
# and distribution.                                                                                                                                                 #
# This Software includes and/or makes use of the following Third-Party Software subject to its own license:                                                         #
# 1. SpaceNet (https://github.com/motokimura/spacenet_building_detection/blob/master/LICENSE) Copyright 2017 Motoki Kimura.                                         #
# DM19-0988                                                                                                                                                         #
#####################################################################################################################################################################

import os
import sys
if(os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.logger import get_logger
l = get_logger("delete_extra")


import argparse
from os.path import join
from shapely import wkt
from shapely.geometry import mapping, Polygon
from tqdm import tqdm
from utils.files.common import read_json, dump_json

import numpy as np
from skimage.io import imread
from cv2 import fillPoly, imwrite


# keep as a tuple, not a list
# add a _ at the end of the disaster-name so they are prefix-free

#DISASTERS_OF_INTEREST = ('guatemala-volcano_', 'hurricane-florence_', 'hurricane-harvey_', 'mexico-earthquake_', 'midwest-flooding_', 'palu-tsunami_', 'santa-rosa-wildfire_', 'socal-fire_', 'lower-puna-volcano_', 'nepal-flooding_', 'pinery-bushfire_', 'portugal-wildfire_', 'sunda-tsunami_', 'woolsey-fire_')
DISASTERS_OF_INTEREST = ('mexico-earthquake_', 'palu-tsunami_', 'sunda-tsunami_')

# running from repo root
LABEL_NAME_TO_NUM = read_json('constants/class_lists/xBD_label_map.json')['label_name_to_num']

def get_dimensions(file_path):
    """
        Returns (width, height, channels) of the image at file_path
    """
    pil_img = imread(file_path)
    img = np.array(pil_img)
    w, h, c = img.shape
    return (w, h, c)

def get_feature_info(feature):
    """Reading coordinate and category information from the label json file
    Args:
        feature: a python dictionary of json labels
    Returns a dict mapping the uid of the polygon to a tuple
        (numpy array of coords, numerical category of the building)
    """
    props = {}

    for feat in feature['features']['xy']:
        # read the coordinates
        feat_shape = wkt.loads(feat['wkt'])
        coords = list(mapping(feat_shape)['coordinates'][0])  # a new, independent geometry with coordinates copied

        # determine the damage type
        if 'subtype' in feat['properties']:
            damage_class = feat['properties']['subtype']
        else:
            damage_class = 'no-damage'  # usually for pre images - assign them to the no-damage class

        damage_class_num = LABEL_NAME_TO_NUM[damage_class]  # get the numerical label

        # maps to (numpy array of coords, numerical category of the building)
        props[feat['properties']['uid']] = (np.array(coords, np.int32), damage_class_num)
    return props


def mask_polygons_together_with_border(size, polys, border):
    """

    Args:
        size: A tuple of (width, height, channels)
        polys: A dict of feature uid: (numpy array of coords, numerical category of the building), from
            get_feature_info()
        border: Pixel width to shrink each shape by to create some space between adjacent shapes

    Returns:
        a dict of masked polygons with the shapes filled in from cv2.fillPoly
    """

    # For each WKT polygon, read the WKT format and fill the polygon as an image
    mask_img = np.zeros(size, np.uint8)  # 0 is the background class

    for uid, tup in polys.items():
        # poly is a np.ndarray
        poly, damage_class_num = tup

        # blank = np.zeros(size, np.uint8)

        # Creating a shapely polygon object out of the numpy array
        polygon = Polygon(poly)

        # Getting the center points from the polygon and the polygon points
        (poly_center_x, poly_center_y) = polygon.centroid.coords[0]
        polygon_points = polygon.exterior.coords

        # Setting a new polygon with each X,Y manipulated based off the center point
        shrunk_polygon = []
        for (x, y) in polygon_points:
            if x < poly_center_x:
                x += border
            elif x > poly_center_x:
                x -= border

            if y < poly_center_y:
                y += border
            elif y > poly_center_y:
                y -= border

            shrunk_polygon.append([x, y])

        # Transforming the polygon back to a np.ndarray
        ns_poly = np.array(shrunk_polygon, np.int32)

        # Filling the shrunken polygon to add a border between close polygons
        # Assuming there is no overlap!
        fillPoly(mask_img, [ns_poly], (damage_class_num, damage_class_num, damage_class_num))

    mask_img = mask_img[:, :, 0].squeeze()
    #print(f'shape of final mask_img: {mask_img.shape}')
    return mask_img


def mask_tiles(images_dir, json_paths, targets_dir, border_width, overwrite_target):

    for json_path in tqdm(json_paths):

        tile_id = os.path.basename(json_path).split('.json')[0]  # just the file name without extension
        image_path = join(images_dir, f'{tile_id}.png')
        target_path = join(targets_dir, f'{tile_id}_b{border_width}.png')

        if os.path.exists(target_path) and not overwrite_target:
            continue

        # read the label json
        label_json = read_json(label_path)

        # read the image and get its size
        tile_size = get_dimensions(image_path)

        # read in the polygons from the json file
        polys = get_feature_info(label_json)

        mask_img = mask_polygons_together_with_border(tile_size, polys, border_width)

        imwrite(target_path, mask_img)


def create_masks(data_path : str,border_width : int, overwrite_target : bool):
    images_dir = join(data_path, 'images')
    labels_dir = join(data_path, 'labels')

    assert os.path.isdir(data_path), f'{data_path} is not a directory'
    assert os.path.isdir(images_dir), f'{data_path} does not contain the folder `images`'
    assert os.path.isdir(labels_dir), f'{data_path} does not contain the folder `labels`'
    assert border_width >= 0, 'border_width < 0'
    assert border_width < 5, 'specified border_width is > 4 pixels - are you sure?'
    assert isinstance(DISASTERS_OF_INTEREST, tuple)
    for i in DISASTERS_OF_INTEREST:
        assert i.endswith('_')

    l.info(f'Disasters to create the masks for: {DISASTERS_OF_INTEREST}')

    targets_dir = join(data_path, f'targets_border{border_width}')
    print(f'A targets directory is at {targets_dir}')
    os.makedirs(targets_dir, exist_ok=True)

    # list out label files for the disaster of interest
    json_files = [file for file in os.listdir(labels_dir) 
                  if file.endswith('.json')]
    json_paths = [join(labels_dir, file) for file in json_files 
                  if file.startswith(DISASTERS_OF_INTEREST)]
    l.info(f'{len(json_files)} label jsons found in labels_dir, ' \
           f'{len(json_paths)} are for the disasters of interest.')
    
    mask_tiles(images_dir, json_paths, targets_dir, border_width, overwrite_target)
    l.info("Mask for disasters created.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create masks for each label json file for disasters specified at the top of the script.')
    parser.add_argument(
        'data_path',
        help=('Path to the directory that contains both the `images` and `labels` folders. '
              'The `targets_border{border_width}` folder will be created if it does not already exist.')
    )
    parser.add_argument(
        '-b', '--border_width',
        type=int,
        default=1
    )
    parser.add_argument(
        '-o', '--overwrite_target',
        help='flag if we want to generate all targets anew',
        action='store_true'
    )
    args = parser.parse_args()
    create_masks(*args)
