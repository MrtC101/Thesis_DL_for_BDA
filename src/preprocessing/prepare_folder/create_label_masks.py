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
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
from utils.common.logger import get_logger
l = get_logger("create_label_masks")


import argparse
import numpy as np
from tqdm import tqdm
from os.path import join
from shapely import wkt
from shapely.geometry import mapping, Polygon
from cv2 import fillPoly, imread, imwrite
from utils.common.files import read_json, is_dir

path = join(os.environ.get("DATA_PATH"),'constants/xBD_label_map.json')
LABEL_NAME_TO_NUM = read_json(path)['label_name_to_num']

def get_shape(image_path):
    """ Opens the image and get it's size"""
    image = imread(image_path)
    (h,w,c) = image.shape
    return (w,h,c)

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
        coords = list(mapping(feat_shape)['coordinates'][0]) 
        # a new, independent geometry with coordinates copied

        # determine the damage type
        if 'subtype' in feat['properties']:
            damage_class = feat['properties']['subtype']
        else:
            damage_class = 'no-damage'  # usually for pre images - assign them to the no-damage class

        damage_class_num = LABEL_NAME_TO_NUM[damage_class]  # get the numerical label

        # maps to (numpy array of coords, numerical category of the building)
        props[feat['properties']['uid']] = (np.array(coords, np.int8), damage_class_num)
    return props


def mask_polygons_together_with_border(size, polys, border):
    """

    Args:
        size: A tuple of (width, height, channels)
        polys: A dict of feature uid: (numpy array of coords, numerical category of the building), from
            get_feature_info()
        border: Pixel width to shrink each shape by to create some space between adjacent shapes

    Returns:
        a dict of masked polygons with mask_polygons_together_with_borderthe shapes filled in from cv2.fillPoly
    """

    # For each WKT polygon, read the WKT format and fill the polygon as an image
    mask_img = np.zeros(size, np.int8)  # 0 is the background class
    
    for tup in polys.values():
        # poly is a np.ndarray
        poly, damage_class_num = tup
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
        ns_poly = np.array([shrunk_polygon], np.int32)
        
        assert ns_poly.shape == (1,len(shrunk_polygon),2),f"{ns_poly.shape} wrong shape"
        
        # Filling the shrunken polygon to add a border between close polygons
        fillPoly(mask_img, ns_poly, (damage_class_num,)*3 )

    mask_img = mask_img[:, :, 0].squeeze()
    #print(f'shape of final mask_img: {mask_img.shape}')
    return mask_img

def mask_tiles(images_dir, labels_dir, targets_dir, border_width):
    
    # list out label files for the disaster of interest
    json_paths = [join(labels_dir,file) for file in os.listdir(labels_dir) if file.endswith('.json')]
    l.info(f'{len(json_paths)} json files found in labels directory.')

    for label_path in tqdm(json_paths):
        
        tile_id = os.path.basename(label_path).split('.json')[0]  # just the file name without extension
        image_path = join(images_dir, f'{tile_id}.png')
        target_path = join(targets_dir, f'{tile_id}_target.png')
        
        if(os.path.exists(target_path)):
            continue
        
        # read the label json
        label_json = read_json(label_path)

        # read the image and get its size
        tile_size = get_shape(image_path)
        # read in the polygons from the json file
        polys = get_feature_info(label_json)

        mask_img = mask_polygons_together_with_border(tile_size, polys, border_width)

        imwrite(target_path, mask_img)


def create_masks(raw_path : str,border_width : int):
    """
        Creates a new target mask for each image in the dataset folder.
    """
    assert border_width >= 0, 'border_width < 0'
    assert border_width < 5, 'specified border_width is > 4 pixels - are you sure?'
        
    for subset in tqdm(os.listdir(raw_path)):
        l.info(f"Creating masks for {subset}/ folder.")
        subset_path = join(raw_path,subset)
        images_dir = join(subset_path, 'images')
        labels_dir = join(subset_path, 'labels')
        is_dir(subset_path)
        is_dir(images_dir)
        is_dir(labels_dir)

        targets_dir = join(subset_path, 'targets')
        os.makedirs(targets_dir, exist_ok=True)
        mask_tiles(images_dir, labels_dir, targets_dir, border_width)

        l.info(f"Masks for {subset}/ folder created.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create masks for each label json file for disasters specified at the top of the script.')
    parser.add_argument(
        'raw_path',
        help=('Path to the directory that contains the content of the raw folder from xBD dataset.')
    )
    parser.add_argument(
        '-b', '--border_width',
        type=int,
        default=1
    )
    args = parser.parse_args()
    create_masks(*args)
