import os
import sys
import math
import pandas as pd
import torch
from torchvision.io import read_image
from torchvision.utils import save_image, draw_bounding_boxes
from matplotlib import transforms

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

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
    
    def _merge_patches(self, patch_list : list) -> torch.Tensor:
        rows = []
        row = []
        for i, patch in enumerate(patch_list):
            row.append(patch)
            if(i % 4 == 3):
                line = torch.cat(row, dim=1)
                row = []
                rows.append(line)
        mask = torch.cat(line, dim=0)
        return mask
    
    def _postproccess(self, pred_patch_list : list, dir_path : str) -> str:
        """Implements the postprocessing pipeline"""
        dmg_labels = [1, 2, 3, 4]
        pred_img = self._merge_patches(pred_patch_list)
        dmg_path = os.path.join(dir_path, "pred_img.png")
        save_image(pred_img, dmg_path)
        
        bbs_df = get_bbs_form_mask(pred_img, dmg_labels)
        pd_values = list(bbs_df.value_counts(subset=["label"]).items())
        pd_table = pd.DataFrame([(val[0][0], val[1]) for val in pd_values],
                                 columns=["Level", "Count"])
        table_path = os.path.join(dir_path, "pred_table.csv")
        pd_table.to_csv(table_path, index=False)

        color = {
            "no-damage":'mediumturquoise',
            "minor-damage":'violet',
            "major-damage":'aqua',
            "destroyed":'lime',
            "un-classified":'black'
        }
        for label in bbs_df["label"].unique():
            curr_bbs_df = bbs_df[bbs_df["label"] == label]
            bbs = [(r["x"], r["y"],r["x"]+r["w"],r["y"]+r["h"]) for r in curr_bbs_df.iterrows()]
            l = torch.tensor([*bbs])
            labels = bbs_df["label"]
            colors = [color[l] for l in labels]
            bbs_img = draw_bounding_boxes(image=torch.zeros(size=(1024,1024)), boxes=l,
                                           colors=colors)
            dir_path = os.path.join(dir_path, f"{label}.png")
            save_image(bbs_img, dmg_path)

    def _normalize_image(img : torch.Tensor) -> torch.Tensor:
        img = img.to(torch.float32) / 255.0
        mean_rgb = [img[0].mean(), img[1].mean(), img[2].mean()]
        std_rgb = [img[0].std(), img[1].std(), img[2].std()]
        norm_img = transforms.Normalize(mean=mean_rgb, std=std_rgb)(img)
        return norm_img
    
    def _crop_image(self, image : torch.Tensor) -> list:
        patch_list = []
        points = [math.floor(1024*p) for p in torch.arange(0, 1, 0.25)]
        for y in points:
            for x in points:
                patch = image[x:x+256,y:y+256]
                patch_list.append(patch)
        return patch_list
        
    def _preprocess(self, pre_img : torch.Tensor, post_img : torch.Tensor) -> list[torch.Tensor]:
        pre_norm = self._normalize_image(pre_img)
        post_norm = self._normalize_image(post_img)
        pre_patches = self._crop_image(pre_norm)
        post_patches = self._crop_image(post_norm)
        patch_list = [(pre,post) for pre,post in zip(pre_patches, post_patches)]
        return patch_list

    def make_prediction(self, pre_path: str, post_path : str, pred_dir) -> str:
        if os.path.isfile(pre_path):
            pre_img = read_image(pre_path)
        else:
            Exception(f"{pre_path} is not a file.")
        if os.path.isfile(post_path):
            post_img = read_image(post_path)
        else:
            Exception(f"{post_path} is not a file.")
        patches = self._preprocess(pre_img, post_img)
        pred_patches = [] 
        for pre_patch, post_path in patches:
            pre_patch.to(device=self.device)
            post_path.to(device=self.device)
            logit_masks = self(pre_patch, post_path)
            pred_patch = self.compute_predictions(logit_masks)
            pred_patches.append(pred_patch[2])
        
        os.makedirs(pred_dir, exist_ok=True)
        self._postproccess(pred_patches, pred_dir)
