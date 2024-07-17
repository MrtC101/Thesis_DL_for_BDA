# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import torch
import random
from tqdm import tqdm
from collections import defaultdict
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.transforms import transforms
from utils.common.pathManager import FilePath
from utils.loggers.console_logger import LoggerSingleton

log = LoggerSingleton()


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


def save_augmentations(dis_id: str, tile_id: str,
                       images: dict[str, torch.Tensor],
                       tile_dict: dict, aug_path: FilePath) -> None:
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
    new_tile_id = 'a' + tile_id
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


def make_augmentations(splits_json_path: FilePath, xbd_path: FilePath,
                       n: int) -> FilePath:
    """
    Create augmented images by applying spatial and color transformations to
    a subset of tiles.

    Args:
        splits_json_path (FilePath): Path to the JSON file containing
        the dataset splits.
        xbd_path (FilePath): Path to the directory where augmented images
        will be saved.
        n (int): Number of images to randomly select and augment.

    Returns:
        FilePath: Path to the updated JSON file containing augmented
        image information.
    """
    xbd_path.must_be_dir()
    aug_path = xbd_path.join("augmented")
    aug_path.create_folder()
    splits_json_path.must_be_json()
    tile_dict = splits_json_path.read_json()

    tile_list = [(dis_id, tile_id, tile)
                 for dis_id, tiles in tile_dict["train"].items()
                 for tile_id, tile in tiles.items()]
    ids = random.choices(range(len(tile_list)), k=n)
    log.info(f"Creating {n} new augmented images.")
    for id in tqdm(ids):
        dis_id, tile_id, tiles = tile_list[id]

        # Leer imágenes y máscaras
        pre_img = read_image(tiles["pre"]["image"])
        post_img = read_image(tiles["post"]["image"])
        bld_mask = read_image(tiles["pre"]["mask"]).expand(3, 1024, 1024)
        dmg_mask = read_image(tiles["post"]["mask"]).expand(3, 1024, 1024)

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
        save_augmentations(dis_id, tile_id, images, tile_dict, aug_path)

    aug_split_json_path = splits_json_path.replace("raw", "aug")
    aug_split_json_path = FilePath(aug_split_json_path)
    aug_split_json_path.save_json(tile_dict)
    log.info("Done")
    return aug_split_json_path
