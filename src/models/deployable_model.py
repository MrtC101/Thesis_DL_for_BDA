import os
import sys
import math
from matplotlib import patches, pyplot as plt
import numpy as np
import pandas as pd
import torch
import json
from torchvision.io import read_image
from torchvision.transforms import Normalize

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.visualization.plot_results import prediction_with_label_map

with open("/home/mrtc101/Desktop/tesina/repo/hiper_siames/data/constants/xBD_label_map.json") as f:
    labels_d = json.load(f)  
       
from postprocessing.ploting.bounding_boxes import get_bbs_form_mask
from models.saimunte_model import SiamUnet

class DeployModel(SiamUnet):
    
    def load_weights(self, weights_path):
        device = torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")
        assert os.path.isfile(weights_path),\
              f"{weights_path} is not a file."
        checkpoint = torch.load(weights_path)
        self.load_state_dict(checkpoint['state_dict'])
        self.freeze_model_param()
        self.device = device
        print("MODEL LOADED")
    
    def _merge_patches(self, patch_batch : torch.Tensor) -> torch.Tensor:
        rows = []
        row = []
        for i, patch in enumerate(torch.unbind(patch_batch)):
            row.append(patch)
            if(i % 4 == 3):
                line = torch.cat(row, dim=1)
                row = []
                rows.append(line)
        mask = torch.cat(rows, dim=0)
        img = mask.to(torch.uint8)
        img = np.array(img)
        return img

    color = {
        "no-damage":'mediumturquoise',
        "minor-damage":'violet',
        "major-damage":'aqua',
        "destroyed":'lime',
        "un-classified":'black'
    }
    
    def bbs_imgs(self, bbs_df : pd.DataFrame, dir_path : str):
        for i, label in enumerate(self.color.keys()):
            fig, ax = plt.subplots(figsize=(10.24, 10.24), dpi=100, facecolor='none')
            bounding_boxes = bbs_df[bbs_df["label"] == label]
            # Dibujar cada bounding box y etiqueta en la imagen
            for _, row in bounding_boxes.iterrows():
                x, y, w, h, l = row["x"], row["y"], row["w"], row["h"], row["label"]
                rect = patches.Rectangle((x, y), width=w, height=h, linewidth=2,
                                        edgecolor=self.color[l], facecolor='none')
                ax.add_patch(rect)
                ax.set_xlim(0, 1024)
                ax.set_ylim(1024, 0)
                ax.axis('off')
            file_path = os.path.join(dir_path, f"bb_{i}.png")
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Eliminar márgenes
            plt.savefig(file_path, transparent=True, format='png', pad_inches=0)
            plt.close()
           
    def _postproccess(self, patch_batch : torch.Tensor, dir_path : str) -> str:
        """Implements the postprocessing pipeline"""
        dmg_labels = [1, 2, 3, 4]
        pred_img = self._merge_patches(patch_batch)
        dmg_path = os.path.join(dir_path, "dmg_img.png")
        color_pred_img = prediction_with_label_map(pred_img, labels_d)
        plt.imsave(dmg_path, color_pred_img)
        plt.close()
        
        bbs_df = get_bbs_form_mask(pred_img, dmg_labels)
        pd_values = list(bbs_df.value_counts(subset=["label"]).items())
        pd_table = pd.DataFrame([(val[0][0], val[1]) for val in pd_values],
                                 columns=["Level", "Count"])
        table_path = os.path.join(dir_path, "pred_table.csv")
        pd_table.to_csv(table_path, index=False)

        self.bbs_imgs(bbs_df, dir_path)

    def _normalize_image(self, img : torch.Tensor) -> torch.Tensor:
        if img.shape[0] > 3:
            img = img[0:3,:,:]
        img = img.to(torch.float32) / 255.0
        mean_rgb = [img[0].mean(), img[1].mean(), img[2].mean()]
        std_rgb = [img[0].std(), img[1].std(), img[2].std()]
        norm_img = Normalize(mean=mean_rgb, std=std_rgb)(img)
        return norm_img
    
    def _crop_image(self, image : torch.Tensor) -> list:
        patch_list = []
        points = [math.floor(1024*p) for p in torch.arange(0, 1, 0.25)]
        for x in points:
            for y in points:
                patch = image[:, x:x+256 ,y:y+256]
                patch_list.append(patch)
        return patch_list
        
    def _preprocess(self, pre_img : torch.Tensor, post_img : torch.Tensor) -> list[torch.Tensor]:
        pre_norm = self._normalize_image(pre_img)
        post_norm = self._normalize_image(post_img)
        pre_patches_list = self._crop_image(pre_norm)
        post_patches_list = self._crop_image(post_norm)
        #to batch
        pre_patches = torch.stack(pre_patches_list, dim=0) 
        post_patches = torch.stack(post_patches_list, dim=0)
        return pre_patches, post_patches

    def make_prediction(self, pre_path: str, post_path : str, pred_dir) -> str:
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
