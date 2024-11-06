# Copyright (c) 2024 Martín Cogo Belver.
# Martín Cogo Belver has rights reserved over this modifications.
import os
import random
import sys
import math
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.io import read_image
from torchvision.transforms import Normalize
from torchvision.utils import draw_bounding_boxes
from postprocessing.bbs.bounding_boxes import get_bbs_from_mask

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.visualization.label_mask_visualizer import LabelMaskVisualizer
from models.siam_unet_model import SiamUnet
from utils.visualization.label_to_color import LabelDict


class DeployModel(SiamUnet):
    """Class that impelments all methods used during deployment."""

    label_dict = LabelDict()

    def load_weights(self, weights_path: str):
        """
            Load weights saved with `torch.save()`. Because this is a deployment Model all layers
            are freeze. It means that all layers are requires_grad = False.

        """
        device = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
        assert os.path.isfile(weights_path), \
            f"{weights_path} is not a file."
        checkpoint = torch.load(weights_path, map_location=device)
        self.load_state_dict(checkpoint['state_dict'])
        self.freeze_model_params()
        self.device = device
        print("MODEL LOADED")

    def _normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        """Applys color normalization for input image"""
        if img.shape[0] > 3:
            img = img[0:3, :, :]
        img = img.to(torch.float32) / 255.0
        mean_rgb = [img[0].mean(), img[1].mean(), img[2].mean()]
        std_rgb = [img[0].std(), img[1].std(), img[2].std()]
        norm_img = Normalize(mean=mean_rgb, std=std_rgb)(img)
        return norm_img

    def _crop_image(self, image: torch.Tensor) -> list:
        """Corps image of 1024x1024 to 16 patches of 256x256
        Args:
            image: `torch.Tensor` with shape (1024,1024)
        """
        patch_list = []
        points = [math.floor(1024*p) for p in torch.arange(0, 1, 0.25)]
        for x in points:
            for y in points:
                patch = image[:, x:x+256, y:y+256]
                patch_list.append(patch)
        return patch_list

    def _preprocess(self, pre_img: torch.Tensor, post_img: torch.Tensor) -> list[torch.Tensor]:
        """Preprocessing pipeline for both pre and post disaster images.
            Args:
             pre_img: Predisaster image with shape 1024x1024
             post_img: Postdisaster image with shape 1024x1024
        """
        pre_norm = self._normalize_image(pre_img)
        post_norm = self._normalize_image(post_img)
        pre_patches_list = self._crop_image(pre_norm)
        post_patches_list = self._crop_image(post_norm)
        # to batch
        pre_patches = torch.stack(pre_patches_list, dim=0)
        post_patches = torch.stack(post_patches_list, dim=0)
        return pre_patches, post_patches

    def _merge_patches(self, patch_batch: torch.Tensor) -> torch.Tensor:
        """Merge all 16 patches of 256x256 into one image of 1024x1024
        Args:
            patch_batch: a torch.Tensor with shape (16,256,256)
        Returns:
            torch.Tensor: damage mask tensor with shape (1024,1024)
        """
        rows = []
        row = []
        for i, patch in enumerate(torch.unbind(patch_batch)):
            row.append(patch)
            if (i % 4 == 3):
                line = torch.cat(row, dim=1)
                row = []
                rows.append(line)
        mask = torch.cat(rows, dim=0)
        img = mask.to(torch.uint8)
        return img

    color = {
        "no-damage": 'mediumturquoise',
        "minor-damage": 'violet',
        "major-damage": 'blue',
        "destroyed": 'lime',
        "un-classified": 'black'
    }

    def bbs_imgs(self, bbs_df: pd.DataFrame, dir_path: str, n: int):
        """Generates a png transparent background image for each class of bounding boxes.

            Args:
                bbs_df: `pd.Dataframe` with all predicted building bounding boxes.
                dir_path: Path to store each bounding box image.
                n: class label number. (Used in frontend)
        """
        print("bbs:", len(bbs_df))
        for cls in self.label_dict.keys_list:
            if cls not in ["background", "un-classified"]:
                # Filtrar las bounding boxes para la clase actual
                cur_df = bbs_df[bbs_df["label"] == cls]
                boxes = torch.tensor(
                    cur_df[['x1', 'y1', 'x2', 'y2']].values, dtype=torch.float)
                color = self.color[cls]

                # Crear una imagen en blanco para dibujar las bounding boxes
                image_with_boxes = torch.zeros(
                    (3, 1024, 1024), dtype=torch.uint8)

                # Dibujar las bounding boxes en la imagen
                if len(cur_df) > 0:
                    image_with_boxes = draw_bounding_boxes(image_with_boxes,
                                                           boxes, colors=color,
                                                           width=2)

                # Convertir el tensor a una imagen numpy
                img_np = image_with_boxes.permute(1, 2, 0).numpy()
                # Crear una máscara alfa para el fondo transparente
                alpha = np.any(img_np > 0, axis=2).astype(np.uint8) * 255
                # Añadir el canal alfa a la imagen
                img = np.dstack((img_np[:, :, 2::-1], alpha))

                # Guardar la imagen usando cv2.imwrite
                i = self.label_dict.get_num_by_key(cls)
                file_path = os.path.join(dir_path, f"bb_{i}_{n}.png")
                cv2.imwrite(file_path, img)

    def _postproccess(self, patch_batch: torch.Tensor, dir_path: str) -> str:
        """Pipeline for postprocessing damage mask patches

            Args:
                patch_batch: batch of 16 256x256 damage mask patches predicted.
                dir_path: Path to store the predicted 1024x1024 damge mask image.
        """
        n = random.randint(0, 10**10)
        pred_img = self._merge_patches(patch_batch)
        dmg_path = os.path.join(dir_path, f"dmg_img_{n}.png")
        img = LabelMaskVisualizer.draw_label_img(pred_img)
        LabelMaskVisualizer.save_tensor_img(img, dmg_path)

        bbs_df = get_bbs_from_mask(pred_img)
        pd_values = list(bbs_df.value_counts(subset=["label"]).items())
        pd_table = pd.DataFrame([(val[0][0], val[1]) for val in pd_values],
                                columns=["Level", "Count"])
        table_path = os.path.join(dir_path, f"pred_table_{n}.csv")
        pd_table.to_csv(table_path, index=False)

        self.bbs_imgs(bbs_df, dir_path, n)

    def make_prediction(self, pre_path: str, post_path: str, pred_dir) -> None:
        """Pipeline for pre and post image prediction of a damage mask.

            Args:
                pre_path: Path to the pre-disaster 1024x1024 image.
                post_path: Path to the post-disaster 1024x1024 image.
                pred_dir: Path to the directory where predicted damage mask should be stored.
        """
        if os.path.isfile(pre_path):
            pre_img = read_image(pre_path)
        else:
            Exception(f"{pre_path} is not a file.")

        if os.path.isfile(post_path):
            post_img = read_image(post_path)
        else:
            Exception(f"{post_path} is not a file.")

        pre_patches, post_patches = self._preprocess(pre_img, post_img)

        pre_patches.to(device=self.device)
        post_patches.to(device=self.device)
        logit_masks = self(pre_patches, post_patches)
        pred_patches = self.compute_predictions(logit_masks)

        os.makedirs(pred_dir, exist_ok=True)
        self._postproccess(pred_patches[2], pred_dir)
