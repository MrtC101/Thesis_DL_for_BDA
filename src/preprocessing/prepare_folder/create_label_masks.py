# Modificaciones (c) 2024 Martín Cogo Belver.
# Martín Cogo Belver has rights reserved over this modifications.
import shutil
import numpy as np
from tqdm import tqdm
from os.path import join
from shapely import wkt
from cv2 import imwrite
from rasterio.features import rasterize
from utils.common.pathManager import FilePath
from utils.loggers.console_logger import LoggerSingleton
from utils.visualization.label_to_color import LabelDict

log = LoggerSingleton()
labels = LabelDict()


def json_to_mask(label_path: FilePath,
                 size: tuple = (1024, 1024)) -> np.ndarray:
    """
    Converts a list of polygon shapes from a JSON label file into a mask image.

    Args:
        label_path (FilePath): Path to the JSON file containing the label data.
          Must be a valid JSON file.
        size (tuple, optional): Size of the output mask image (height, width).
          Default is (1024, 1024).

    Returns:
        np.ndarray: A 2D numpy array representing the mask image, where each
            pixel value corresponds to the numerical category of the building
            as specified in the JSON file.

    Raises:
        AssertionError: If the label_path is not a valid JSON file or if the
        file does not exist.
        ValueError: If the JSON file does not contain the expected structure.
    """
    label_path.must_be_json()
    label_json = label_path.read_json()
    shapes = []
    for building in label_json['features']['xy']:
        # Get damage class, defaulting to 'no-damage' if not present
        damage_class = building['properties'].get('subtype', 'no-damage')
        dmg_label = labels.get_num_by_key(damage_class)
        if(dmg_label == 5):
            dmg_label = 0
        # Read the coordinates
        shapes.append((wkt.loads(building['wkt']), dmg_label))
    mask_img = np.zeros(shape=size, dtype=np.uint8)
    if len(shapes) > 0:
        mask_img = rasterize(shapes, size, fill=0)
    return mask_img


def mask_tiles(labels_dir: FilePath, targets_dir: FilePath) -> None:
    """Creates a new target mask for each image in the dataset folder.
    and stores it inside the new targets directory

    Args:
        images_dir: path to the images folder.
        targets_dir: path to the new targets folder.
    """
    # list out label files for the disaster of interest
    log.info(f"{len(labels_dir.get_file_paths())} json files found in labels" +
             " directory.")
    for label_path in tqdm(labels_dir.get_file_paths()):
        file_name = label_path.basename().split('.json')[0]
        target_path = targets_dir.join(f'{file_name}_target.png')
        # read the label json
        mask_img = json_to_mask(label_path)
        imwrite(target_path, mask_img)


def create_masks(raw_path: FilePath) -> None:
    """Creates a target image mask for each label json file inside
    `raw_path/subset/label` folder and creates `raw_path/subset/target`
      folder for each subset inside raw_path.

        Args:
            raw_path: Path to the xBD dataset directory that contains `subset`
            folders.

        Raises:
            AssertionException: If Path is not a Folder

        Example:
            >>> delete_not_in("data/xBD/raw")
    """

    for subset in tqdm(raw_path.get_folder_names()):
        log.info(f"Creating masks for {subset}/ folder.")

        subset_path = raw_path.join(subset)
        labels_dir = subset_path.join('labels')
        targets_dir = subset_path.join('targets')

        if (not labels_dir.is_dir()):
            log.info(f"Skiping folder {subset_path}," +
                     "there is no folder 'images' or 'labels'.")
            # Move folder to skip it
            shutil.move(subset_path, join(raw_path, "..", subset))

        if not targets_dir.is_dir():
            targets_dir.create_folder()
            mask_tiles(labels_dir, targets_dir)

        log.info(f"Masks for {subset}/ folder created.")
