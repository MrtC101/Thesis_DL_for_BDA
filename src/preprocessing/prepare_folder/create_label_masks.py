# Modificaciones (c) 2024 Martín Cogo Belver.
# Martín Cogo Belver has rights reserved over this modifications.

import os
import sys
import math
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join
from shapely import wkt

from cv2 import fillPoly, imread, imwrite

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.files import read_json, is_dir
from utils.loggers.console_logger import LoggerSingleton
from utils.visualization.label_to_color import LabelDict

log = LoggerSingleton()
labels = LabelDict()

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

    props = []

    for building in feature['features']['xy']:
        # if build from bld_mask = no-damage
        damage_class = building.get('subtype','no-damage')
        
        dmg_label = labels.get_key_by_num(damage_class)

        # read the coordinates
        shape = wkt.loads(building['wkt'])
        point_list = list(shape.exterior.coords)
        point_list = list(map(lambda cord: (math.floor(cord[0]),\
                                            math.floor(cord[1])), point_list))

        centroid  = shape.centroid.coords[0]
        props.append((point_list, dmg_label, centroid))
    return props

def border_correction(x, y, c_x, c_y, b):
    """Manipurales c polygon's border point by b
    Args:
        (x,y) point coordinate
        (c_x,c_y) polygon's centroid point
        b border width
    """
    x += b if y < c_x else (-b)
    y += b if y < c_y else (-b)
    x = max(min(x, 1024), 0)
    y = max(min(y, 1024), 0)
    return x,y

def polygons_list_to_mask(image_path, label_path, border: int) -> np.ndarray:
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

    label_json = read_json(label_path)
    size = get_shape(image_path)
    polys = get_feature_info(label_json)
        
    mask_img = np.zeros(size, np.int32)  # 0 is the background class
    for point_list, dmg_label, centroid in polys:
        c_x, c_y = centroid
        poly = list(map(lambda cord: border_correction(cord[0], cord[1],
                                                        c_x, c_y, border), point_list))
        ns_poly = np.array([poly], np.int32)
        mask_img = fillPoly(mask_img, ns_poly, (dmg_label,)*3)

    return mask_img[:,:,0].squeeze()


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

        tile_id = os.path.basename(label_path).split('.json')[0]
        image_path = join(images_dir, f'{tile_id}.png')
        target_path = join(targets_dir, f'{tile_id}_target.png')

        # read the label json
        mask_img = polygons_list_to_mask(image_path, label_path, border_width)
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
    assert border_width < 5, 'specified border_width is > 4 pixels - are you sure?'

    for subset in tqdm(os.listdir(raw_path)):
        log.info(f"Creating masks for {subset}/ folder.")
        
        subset_path = join(raw_path, subset)
        is_dir(subset_path)

        images_dir = join(subset_path, 'images')
        labels_dir = join(subset_path, 'labels')
        targets_dir = join(subset_path, 'targets')

        if (not os.path.exists(images_dir) or not os.path.exists(labels_dir)):
            log.info(f"Skiping folder {subset_path}, there is no folder 'images' or 'labels'.")
            #Move folder to skip it
            shutil.move(subset_path, join(raw_path, "..", subset))

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
