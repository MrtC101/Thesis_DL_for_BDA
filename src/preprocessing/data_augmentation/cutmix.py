# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
from collections import defaultdict
import random
from typing import Dict
import numpy as np
import pandas as pd
from shapely import Polygon, wkt
import shapely
from preprocessing.raw.split_raw_dataset import get_buildings, get_tiles_count
from utils.common.pathManager import FilePath
from rasterio.features import rasterize
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
from torchvision.io import read_image
from torchvision.utils import save_image
from utils.loggers.console_logger import LoggerSingleton
import torch


def save_new_tile(dis_id: str, tile_id: str,
                  images: dict[str, torch.Tensor],
                  tile_dict: dict, aug_path: FilePath, n) -> None:
    """
    Save augmented images and update the tile dictionary with new paths.

    Args:
        dis_id (str): The disaster ID for the image tiles.
        tile_id (str): The original tile ID.
        images (dict[str, torch.Tensor]): Dictionary of images to be saved.
        tile_dict (dict): Dictionary containing the dataset split information.
        aug_path (FilePath): Path to the directory where augmented images will
        be saved.

    Returns:
        None
    """
    # Generar nuevos identificadores de imagen
    new_tile_id = str(n) + 'a' + tile_id
    prefixes = ["pre", "post", "pre", "post"]
    names = ["disaster", "disaster", "disaster_target", "disaster_target"]
    types = ["image", "image", "mask", "mask"]

    # Guardar imágenes transformadas y actualizar el diccionario
    for (key, img), prefix, name, t in zip(images.items(), prefixes,
                                           names, types):
        folder = aug_path.join(f"{dis_id}_{new_tile_id}")
        folder.create_folder()
        img_path = folder.join(
            f"{dis_id}_{new_tile_id}_{prefix}_{name}.png"
        )
        save_image(img / 255, img_path)
        if new_tile_id not in tile_dict["train"][dis_id].keys():
            new_tile = defaultdict(lambda: defaultdict(lambda: str))
            tile_dict["train"][dis_id][new_tile_id] = new_tile
        tile_dict["train"][dis_id][new_tile_id][prefix][t] = img_path


def random_crop(img: torch.Tensor, max_pad: int = 50) -> torch.Tensor:
    """
    Replicates the `iaa.Crop` functionality from the imgaug library by
    applying random padding.

    Args:
        img (torch.Tensor): The input image tensor to be cropped.
        max_pad (int, optional): Maximum number of pixels to pad on each
        side of the image. Default is 50.

    Returns:
        torch.Tensor: The image tensor after applying random padding.
    """
    pad_left = torch.randint(0, max_pad + 1, (1,)).item()
    pad_right = torch.randint(0, max_pad + 1, (1,)).item()
    pad_top = torch.randint(0, max_pad + 1, (1,)).item()
    pad_bottom = torch.randint(0, max_pad + 1, (1,)).item()
    padding = (pad_left, pad_top, pad_right, pad_bottom)
    return transforms.Pad(padding)(img)


# Sequence of spatial transformations
spatial_transforms = transforms.Compose([
    # Random crop of 50 pixels on both axes
    transforms.Lambda(random_crop),
    # Random rotation between -25 and +25 degrees
    # Random scaling between 0.9 and 1.2 on both axes
    # Random translation between -5% and +5%
    transforms.RandomAffine(degrees=(-25, 25),
                            translate=(0.05, 0.05),
                            scale=(0.9, 1.2)),
    transforms.Resize((1024, 1024)),
    # Horizontal, Vertical flip with a probability of 50%
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5)
])

# Secuencia de transformaciones de color
# Multiplicar valores de matiz y saturación
# Cambio aleatorio de contraste
# Cambio aleatorio de brillo
color_transforms = transforms.ColorJitter(
    hue=(-0.4, 0.4),
    saturation=(0.6, 1.4),
    contrast=(0.75, 1.25),
    brightness=(0.75, 1.25))


def augment_imgs(pre_img: torch.Tensor, post_img: torch.Tensor, bld_mask: torch.Tensor,
                 dmg_mask: torch.Tensor, **kargs) -> dict:
    """Applies spatial and color transformations to the input images and masks.

    This function takes in a set of images and corresponding masks, applies spatial
    transformations to all images, and then applies color transformations to the
    pre- and post-images. The resulting transformed images and masks are returned in a
    dictionary.

    Args:
        pre_img : The pre-disaster image tensor.
        post_img : The post-disaster image tensor.
        bld_mask : The building mask tensor.
        dmg_mask : The damage mask tensor.

    Returns:
        dict: A dictionary containing the transformed images and masks
    """
    # Aplicar transformaciones espaciales
    stacked_imgs = torch.stack([pre_img, post_img, bld_mask, dmg_mask],
                               dim=0)
    stacked_imgs = spatial_transforms(stacked_imgs)
    pre_img, post_img, bld_mask, dmg_mask = torch.unbind(stacked_imgs,
                                                         dim=0)

    # Aplicar transformaciones de color
    stacked_imgs = torch.stack([pre_img, post_img], dim=0)
    stacked_imgs = color_transforms(stacked_imgs)
    pre_img, post_img = torch.unbind(stacked_imgs, dim=0)

    images = {
        "pre_img": pre_img,
        "post_img": post_img,
        "bld_mask": bld_mask[0],  # Extraer el canal
        "dmg_mask": dmg_mask[0]   # Extraer el canal
    }
    return images


def read_tile(tile_dict: dict) -> dict:
    """Loads images from a tile dictionary and returns a dictionary with all of them."""
    img_dict = {
        "pre_img": read_image(tile_dict["pre"]["image"]),
        "post_img": read_image(tile_dict["post"]["image"]),
        "bld_mask": read_image(tile_dict["pre"]["mask"]).expand(3, 1024, 1024),
        "dmg_mask": read_image(tile_dict["post"]["mask"]).expand(3, 1024, 1024),
        "pre_json": FilePath(tile_dict["pre"]["json"]).read_json(),
        "post_json": FilePath(tile_dict["post"]["json"]).read_json(),
    }
    return img_dict


def get_square(tile: dict, poly: shapely.Polygon) -> dict:
    """Returns the corresponding content inside the bounding box of the `poly` in
    a dictionary for each pre and post image and bld,damage mask.
    """
    x1, y1, x2, y2 = [round(e) for e in poly.bounds]
    mask_matrix = rasterize([poly], out_shape=(
        1024, 1024), fill=0, dtype=np.uint8)
    mask_matrix = torch.from_numpy(mask_matrix).to(torch.bool)
    mask_matrix = mask_matrix.expand_as(tile["pre_img"])
    reg_dict = {
        "pre_img": (tile["pre_img"] * mask_matrix)[:, y1:y2, x1:x2],
        "bld_mask": (tile["bld_mask"] * mask_matrix)[:, y1:y2, x1:x2],
        "post_img": (tile["post_img"] * mask_matrix)[:, y1:y2, x1:x2],
        "dmg_mask": (tile["dmg_mask"] * mask_matrix)[:, y1:y2, x1:x2]
    }
    return reg_dict


def get_poly(tile, list_id) -> shapely.Polygon:
    """Loads the polygon with wkt format"""
    return wkt.loads(tile["post_json"]["features"]["xy"][list_id]["wkt"])

# Función para calcular el momento de inercia


def moment_of_inertia(polygon):
    centroid = polygon.centroid
    coords = np.array(polygon.exterior.coords)
    x_coords = coords[:, 0] - centroid.x
    y_coords = coords[:, 1] - centroid.y
    I_xx = np.sum(y_coords**2)
    I_yy = np.sum(x_coords**2)
    return I_xx, I_yy


def farthest_points(polygon: Polygon):
    """
    Encuentra los puntos más alejados entre sí en el borde de un polígono.

    Args:
        polygon (Polygon): El polígono de entrada.

    Returns:
        tuple: Dos puntos (como tuplas de coordenadas) más alejados entre sí en el
        borde del polígono.
    """
    exterior_coords = np.array(polygon.exterior.coords)
    max_distance = 0
    farthest_pair = (None, None)

    for i in range(len(exterior_coords)):
        for j in range(i + 1, len(exterior_coords)):
            distance = np.linalg.norm(
                exterior_coords[i] - exterior_coords[j])
            if distance > max_distance:
                max_distance = distance
                farthest_pair = (
                    tuple(exterior_coords[i]), tuple(exterior_coords[j]))

    return farthest_pair


def vector_between_points(point1, point2):
    return np.array(point2) - np.array(point1)


def calculate_scale(new_poly: Polygon, org_poly: Polygon):
    """
    Calculate the scale factors (width and height) required to resize
    a new polygon to match the bounds of an original polygon.

    Args:
        new_poly (Polygon): The new polygon to be scaled.
        org_poly (Polygon): The original polygon to match.

    Returns:
        tuple: A tuple containing the width and height scale factors.
    """
    nx1, ny1, nx2, ny2 = new_poly.bounds
    ox1, oy1, ox2, oy2 = org_poly.bounds
    dx = (ox2 - ox1) / (nx2 - nx1)
    w = round((nx2 - nx1) * dx)
    dy = (oy2 - oy1) / (ny2 - ny1)
    h = round((ny2 - ny1) * dy)
    return w, h


def vector_from_center(poly: Polygon):
    x1, y1, x2, y2 = poly.bounds
    b1 = (round(x1), round(y1))
    b2 = (round(x2), round(y2))
    bc = (round((b2[0]-b1[0])/2) + b1[0], round((b2[1]-b1[1])/2) + b1[1])

    p1, p2 = farthest_points(poly)
    vec1 = vector_between_points(bc, p1)
    vec2 = vector_between_points(bc, p2)
    vec_c = vec1 if np.linalg.norm(vec1) >= np.linalg.norm(vec2) else vec2
    return vec_c


def calculate_rotation_angle(vec1, vec2):
    unit_vec1 = vec1 / np.linalg.norm(vec1)
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(unit_vec1, unit_vec2)
    angle_radians = np.arccos(dot_product)
    angle_degrees = np.degrees(angle_radians)
    # Determinar el signo del ángulo
    return angle_degrees if np.cross(unit_vec1, unit_vec2) > 0 else - angle_degrees


def modify_image_based_on_polygon(replacement_region: Dict[str, torch.Tensor],
                                  new_polygon: Polygon, original_polygon: Polygon):
    # Apilar las imágenes
    pre_img = replacement_region['pre_img']
    post_img = replacement_region['post_img']
    bld_mask = replacement_region['bld_mask']
    dmg_mask = replacement_region['dmg_mask']

    stacked_imgs = torch.stack([pre_img, post_img, bld_mask, dmg_mask],
                               dim=0)

    # Calculate rotation degree
    vec1 = vector_from_center(new_polygon)
    vec2 = vector_from_center(original_polygon)
    rotation_angle = calculate_rotation_angle(vec1, vec2)
    w, h = calculate_scale(new_polygon, original_polygon)

    resized_reg = F.resize(stacked_imgs, size=(h, w))
    rotated_reg = F.rotate(resized_reg, -rotation_angle, expand=False)
    # Separar las imágenes
    pre_img, post_img, bld_mask, dmg_mask = torch.unbind(
        rotated_reg, dim=0)
    # Actualizar el diccionario con las imágenes modificadas
    replacement_region['pre_img'] = pre_img
    replacement_region['post_img'] = post_img
    replacement_region['bld_mask'] = bld_mask
    replacement_region['dmg_mask'] = dmg_mask

    return replacement_region


def replace_region(new_tile: Dict[str, torch.Tensor], modified_region: Dict[str, torch.Tensor],
                   new_poly: Polygon, org_poly: Polygon) -> Dict[str, torch.Tensor]:
    """
    Replaces a region in the new_tile based on the modified_region and aligns it according to
    the polygons.

    Args:
        new_tile (Dict[str, torch.Tensor]): The tile to be modified.
        modified_region (Dict[str, torch.Tensor]): The region to be inserted into the tile.
        new_poly (Polygon): The polygon representing the region to be moved.
        org_poly (Polygon): The target polygon representing the position to move the region to.

    Returns:
        Dict: The updated tile with the replaced region.
    """
    # Get the bounds of the new polygon and adjust them based on the translation
    x1, y1, x2, y2 = org_poly.bounds
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    # Replace the region in the new_tile
    for key in modified_region.keys():
        _, w, z = modified_region[key].shape
        w = 1023-y1 if y1 + w > 1023 else w
        z = 1023-x1 if x1+z > 1023 else z
        # Insertar el parche en la imagen destino
        mask = (modified_region[key][:, 0:w, 0:z] > 0).any(axis=0)
        mask = mask.expand(3, w, z)
        curr_tile = new_tile[key].clone()
        new_region = torch.where(~mask,
                                 curr_tile[:, y1:y1+w, x1:x1+z],
                                 modified_region[key][:, 0:w, 0:z])
        curr_tile[:, y1:y1+w, x1:x1+z] = new_region
        new_tile[key] = curr_tile
    return new_tile


def make_replacement(new_list_id: list, rep_tile, org_list_id: list, new_tile: dict):
    """Adds each building to cover up its corresponding replaceable building"""
    new_poly = get_poly(rep_tile, new_list_id)
    org_poly = get_poly(new_tile, org_list_id)
    replacement_region: tuple = get_square(rep_tile, new_poly)
    modified_region = modify_image_based_on_polygon(replacement_region, new_poly, org_poly)
    new_tile = replace_region(new_tile, modified_region, new_poly, org_poly)
    return new_tile


def create_new_tile(tile_dict, replaceable_blds):
    """Creates a new tile by replacing specified buildings in an original tile.

    This function reads an original tile and replaces buildings based on the provided
    `replaceable_blds` DataFrame, which contains information about the buildings to
    be replaced and their corresponding replacements. The resulting tile is then augmented
    to enhance its features.

    Args:
        tile_dict (dict): A dictionary containing tiles indexed by their IDs.
        replaceable_blds (pd.DataFrame): A DataFrame containing information about
                                          buildings to be replaced and their replacements,
                                          with columns for original and new tile IDs.

    Returns:
        aug_tile: The augmented tile after applying building replacements.
    """

    org_tile_id = list(replaceable_blds["org_tile_id"].unique())[0]
    new_tile = read_tile(tile_dict[org_tile_id])
    for bld_tile_id in replaceable_blds['new_tile_id'].unique():
        curr_tile_rep = replaceable_blds[replaceable_blds["new_tile_id"] == bld_tile_id]
        rep_tile = read_tile(tile_dict[bld_tile_id])
        for id, row in curr_tile_rep.iterrows():
            new_tile = make_replacement(
                row['new_list_id'], rep_tile, row['org_list_id'], new_tile)
    aug_tile = augment_imgs(**new_tile)
    return aug_tile


def new_tile_count(replaceable_blds: pd.DataFrame, tile: dict, label: list) -> list:
    """Returns the count of buildings for each tile in a list."""
    rep_count = get_tiles_count(replaceable_blds)
    new_tile_count = tile - rep_count
    new_tile_count[label] = tile[label] + len(replaceable_blds)
    return new_tile_count


def get_replaceable_buildings(tile_id: str, bld_x_tile_df: pd.DataFrame, label: list) -> list:
    """Builds a dataframe with all buildings used for replacement.
    Buildings are selected randomly."""
    tile_blds = bld_x_tile_df.loc[tile_id]
    not_lab_blds = tile_blds[tile_blds["label"] != label]
    not_lab_blds = not_lab_blds.reset_index()
    rep_rate = random.randrange(5, 10, 1) / 10
    rep_bld_ids = random.sample(list(not_lab_blds.index), round(len(not_lab_blds) * rep_rate))
    replaceable_blds = not_lab_blds.loc[rep_bld_ids]
    return replaceable_blds


def combination_of_replacement(label_blds_df: pd.DataFrame,
                               replace_blds_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a relation between replacement and replaceable buildings."""
    true_index = replace_blds_df.index
    label_blds_df = label_blds_df.sample(len(replace_blds_df), replace=True)
    label_blds_df.reset_index(inplace=True)
    label_blds_df.rename(columns={key: "new_"+key for key in label_blds_df.columns},
                         inplace=True)
    replace_blds_df.rename(columns={key: "org_"+key for key in replace_blds_df.columns},
                           inplace=True)
    replace_blds_df.reset_index(drop=True, inplace=True)
    rep = pd.concat([label_blds_df, replace_blds_df], axis=1)
    rep.set_index(true_index, drop=True, inplace=True)
    return rep


def augment_label(tile_dict: dict, aug_path: FilePath, out_path: FilePath) -> dict:
    """Sequential augmentation of images for the minority class to match the majority class.

    This function increases the number of buildings in the minority class while minimizing
    the sampling of other buildings to maintain balance in the dataset. It processes the
    training tiles for each disaster, extracting building and tile counts, and then applies
    augmentation strategies to create new tiles.

    Args:
        tile_dict (dict): A dictionary containing training tiles organized by disaster IDs.
        aug_path (FilePath): The path where augmented tiles will be saved.
        out_path (FilePath): The path where the training weights will be saved.

    Returns:
        dict: The updated `tile_dict` with augmented tiles for the minority classes.

    Notes:
        - The function logs the balanced dataset summary for each disaster.
        - The output includes a JSON file with weights for specific labels based on the
          augmented dataset.
    """
    log = LoggerSingleton()
    n = 0
    label_sum = None
    for dis_id, tiles in tile_dict["train"].items():
        bld_x_tile_df = get_buildings(tiles)
        lab_x_tile_df = get_tiles_count(bld_x_tile_df)
        new_df = lab_x_tile_df.copy()
        curr_num = lab_x_tile_df.sum()
        for label in curr_num.sort_values().index:
            if label not in ["no-buildings", "un-classified"]:
                lab_tiles = lab_x_tile_df[lab_x_tile_df[label] > 0]
                tile_id = random.choice(lab_tiles.index)
                lab_blds = bld_x_tile_df[bld_x_tile_df["label"] == label]
                while (new_df.sum()[label] < lab_x_tile_df.sum().max()):
                    tile = lab_tiles.loc[[tile_id]]
                    tile_not_label_blds = get_replaceable_buildings(
                        tile_id, bld_x_tile_df, label)
                    new_tile = new_tile_count(
                        tile_not_label_blds, tile, label)
                    new_df = pd.concat([new_df, new_tile])
                    replaceable_blds = combination_of_replacement(
                        lab_blds, tile_not_label_blds)
                    aug_tile = create_new_tile(tiles, replaceable_blds)
                    save_new_tile(dis_id, tile_id, aug_tile,
                                  tile_dict, aug_path, n)
                    n += 1
        log.info(f"Balanced dataset for disaster: {new_df.sum()}")
        if label_sum is None:
            label_sum = new_df.sum()
        else:
            label_sum += new_df.sum()
    weights_df = label_sum.sum() / label_sum
    weights_df[label_sum.isna()] = 0.0
    weights = {label: w for label, w in weights_df.items()
               if label in ['destroyed', 'major-damage', 'minor-damage', 'no-damage']}
    out_path.join("train_weights.json").save_json(weights)
    return tile_dict


def do_cutmix(tile_json_path: FilePath, data_path: FilePath, out_path: FilePath) -> FilePath:
    """
    1. For each disaster, extract the buildings and the count of each tile.
    2. For each class, ordered from the minority to the majority, filter the tiles
    that contain that class.
    3. Calculate the total number of new buildings needed.
    4. Randomly determine the total number of buildings to replace from each image.

    5. The number of images will increase until one class is balanced.
    6. The idea is to take the buildings from one image, take the buildings from other classes,
    and randomly replace them with buildings from other images, but with the same damage level.
    7. Transformations are applied to the cropped patch so that it better fits the building
      being replaced.
    Since they are buildings from the same disaster and from images containing buildings of the
      same class, we can say they share part of the context.
    8. The entire image is subjected to transformations to ensure it is not identical to the
      other images. (Perhaps we'll make sure they are not too different.)
    9. How do I determine the number of buildings to replace in each image?
    Assuming I can determine the number of new buildings needed and
    I can determine how many to replace per image

    Args:
        tile_json_path: Path to the already split file.
        data_path: Path to the data folder.
        out_path: Path to the path where to generate the cutmixed split json file.
    """

    data_path.must_be_dir()
    aug_path = data_path.join("cutmixed")
    aug_path.create_folder()
    tile_json_path.must_be_json()
    tile_dict = tile_json_path.read_json()

    aug_split_json_path = augment_label(tile_dict, aug_path, out_path)

    aug_split_json_path = tile_json_path.replace("raw", "aug")
    aug_split_json_path = FilePath(aug_split_json_path)
    aug_split_json_path.save_json(tile_dict)
    return aug_split_json_path
