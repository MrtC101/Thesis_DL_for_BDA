# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Modificaciones (c) 2024 Martín Cogo Belver.
# Martín Cogo Belver has rights reserved over this modifications.
#
# Modification Notes:
# - Documentation added with docstrings for code clarity.
# - Re-implementation of methods to enhance readability and efficiency.
# - Re-implementation of features for improved functionality.
# - Changes in the logic of implementation for better performance.
# - Bug fixes in the code.
#
# See the LICENSE file in the root directory of this project for the full text of the MIT License.
#################################################################################
# xView2                                                                        #
# Copyright 2019 Carnegie Mellon University.                                    #
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING         #
# INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON          #
# UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS   #
# TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE  #
# OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL.#
# CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT#
# TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.                 #
# Released under a MIT (SEI)-style license, please see LICENSE.md or contact    #
# permission@sei.cmu.edu for full terms.                                        #
# [DISTRIBUTION STATEMENT A] This material has been approved for public release #
# and unlimited distribution. Please see Copyright notice for non-US Government #
# use and distribution.                                                         #
# DM19-0988                                                                     #
#################################################################################
import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

import shutil
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join
from shapely import wkt
from shapely.geometry import mapping, Polygon
from cv2 import fillPoly, imread, imwrite
from utils.common.files import read_json, is_dir
from utils.common.logger import LoggerSingleton
log = LoggerSingleton()

path = join(os.environ.get("DATA_PATH"), 'constants/xBD_label_map.json')
LABEL_NAME_TO_NUM = read_json(path)['label_name_to_num']


def get_shape(image_path: str) -> tuple:
    """Opens the image and retrieves its size."""
    image = imread(image_path)
    (h, w, c) = image.shape
    return (w, h, c)


def get_feature_info(feature: dict) -> dict:
    """Reads coordinate and category information from the label JSON file.

    Args:
        feature (dict): JSON labels.

    Returns:
        dict: Mapping the UID of the polygon to a tuple
        containing a NumPy array of coordinates and
        the numerical category of the building.
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
            # usually for pre images - assign them to the no-damage class
            damage_class = 'no-damage'

        damage_class_num = LABEL_NAME_TO_NUM[damage_class]
        # maps to (numpy array of coords, numerical category of the building)
        props[feat['properties']['uid']] = \
            (np.array(coords, np.uint8), damage_class_num)
    return props


def polygons_list_to_mask(size: tuple, polys: list, border: int) -> np.ndarray:
    """ Plots inside an np.ndarray all polygons inside polys.

    Args:
        size: A tuple of (width, height, channels) that represents the output
        image shape. 
        polys: A dict of feature uid:\
            (numpy array of coords, numerical category of the building),
            from get_feature_info()
        border: Pixel width to shrink each shape by to create some space
        between adjacent shapes

    Returns:
        np.ndarray: masked polygons with polygons_list_to_mask shapes filled in
        from cv2.fillPoly
    """

    mask_img = np.zeros(size, np.int32)  # 0 is the background class
    for tup in polys.values():
        # poly is a np.ndarray
        poly, damage_class_num = tup
        polygon = Polygon(poly)

        # Getting the center points from the polygon and the polygon points
        (poly_center_x, poly_center_y) = polygon.centroid.coords[0]
        polygon_points = polygon.exterior.coords

        # Setting a new polygon with each X,Y manipulated
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

            x = max(min(x, 255), 0)
            y = min(min(y, 255), 0)
            shrunk_polygon.append([x, y])

        # Transforming the polygon back to a np.ndarray
        ns_poly = np.array([shrunk_polygon], np.int32)

        assert ns_poly.shape == (1, len(shrunk_polygon), 2), \
            f"{ns_poly.shape} wrong shape"

        # Filling the shrunken polygon to add a border between close polygons
        mask_img = fillPoly(mask_img, ns_poly, (damage_class_num,)*3)

    mask_img = mask_img[:, :, 0].squeeze()
    # print(f'shape of final mask_img: {mask_img.shape}')
    return mask_img


def mask_tiles(images_dir: str, labels_dir: str, targets_dir: str,
               border_width: int) -> None:
    """Creates a new target mask for each image in the dataset folder.
    and stores it inside the new targets directory

    Args:
        images_dir: path to the images folder.

        labels_dir: path to the labels folder.

        targets_dir: path to the new targets folder.

        border_width: size of new polygons plotted inside new mask image.
    """
    # list out label files for the disaster of interest
    json_paths = [join(labels_dir, file)
                  for file in os.listdir(labels_dir) if file.endswith('.json')]
    log.info(f'{len(json_paths)} json files found in labels directory.')

    for label_path in tqdm(json_paths):

        # just the file name without extension
        tile_id = os.path.basename(label_path).split('.json')[0]
        image_path = join(images_dir, f'{tile_id}.png')
        target_path = join(targets_dir, f'{tile_id}_target.png')

        if (os.path.exists(target_path)):
            continue

        # read the label json
        label_json = read_json(label_path)

        # read the image and get its size
        tile_size = get_shape(image_path)
        # read in the polygons from the json file
        polys = get_feature_info(label_json)

        mask_img = polygons_list_to_mask(tile_size, polys, border_width)

        imwrite(target_path, mask_img)


def create_masks(raw_path: str, border_width: int) -> None:
    """Creates a target image mask for each label json file inside
    `raw_path/subset/label` folder and creates `raw_path/subset/target` folder
    for each subset inside raw_path.

        Args:
            raw_path: Path to the xBD dataset directory that contains `subset`
            folders.

        Raises:
            AssertionException: If Path is not a Folder

        Example:
            >>> delete_not_in("data/xBD/raw")
    """

    log.name = "Create Target Masks"
    assert border_width >= 0, 'border_width < 0'
    assert border_width < 5, \
        'specified border_width is > 4 pixels - are you sure?'

    for subset in tqdm(os.listdir(raw_path)):
        log.info(f"Creating masks for {subset}/ folder.")
        subset_path = join(raw_path, subset)
        images_dir = join(subset_path, 'images')
        labels_dir = join(subset_path, 'labels')
        is_dir(subset_path)
        if (not os.path.exists(images_dir) or not os.path.exists(labels_dir)):
            log.info(f"Skiping folder {subset_path}, there is no folder 'images' or 'labels'.")
            shutil.move(subset_path, join(raw_path, "..", subset))
            continue

        targets_dir = join(subset_path, 'targets')
        os.makedirs(targets_dir, exist_ok=True)
        mask_tiles(images_dir, labels_dir, targets_dir, border_width)

        log.info(f"Masks for {subset}/ folder created.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create masks for each label json file for disasters \
            specified at the top of the script.')
    parser.add_argument(
        'raw_path',
        help=('Path to the directory that contains the content of the raw \
              folder from xBD dataset.')
    )
    parser.add_argument(
        '-b', '--border_width',
        type=int,
        default=1
    )
    args = parser.parse_args()
    create_masks(*args)
