# Modificaciones (c) 2024 Martín Cogo Belver.
# Martín Cogo Belver has rights reserved over this modifications.
import cv2
import numpy as np
import matplotlib
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils.visualization.label_to_color import LabelDict

matplotlib.use("Agg")  # Usar backend 'Agg' para entornos sin display


class LabelMaskVisualizer:
    """Visualizes raster mask labels and predictions."""

    def __init__(self):
        """Constructs a raster label visualizer."""
        self.label_map = LabelDict()
        self.num_classes = len(self.label_map)
        required_colors = [mcolors.to_rgb(mcolors.CSS4_COLORS[color])
                           for color in self.label_map.color_list]
        self.colormap = mcolors.ListedColormap(required_colors)
        self.normalizer = mcolors.Normalize(vmin=0, vmax=self.num_classes - 1)

    def draw_label_img(self, label_tensor: torch.Tensor) -> torch.Tensor:
        """Visualizes a label mask or hardmax predictions of a model."""
        img: torch.Tensor
        if (len(label_tensor.shape) == 3) or (label_tensor.shape[0] == 1):
            label_tensor = label_tensor.squeeze(0)
        img = torch.zeros((label_tensor.shape[1],
                           label_tensor.shape[0], 3), dtype=torch.uint8)

        for label_i in label_tensor.unique():
            color = self.colormap(label_i)[:3]
            c_t = (torch.Tensor(color) * 255).to(torch.uint8)
            mask = (label_tensor == label_i)
            img[mask] = c_t
        return img

    @staticmethod
    def save_arr_img(img: np.ndarray, path: str):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path, img)

    @staticmethod
    def save_tensor_img(img: torch.Tensor, path: str):
        LabelMaskVisualizer.save_arr_img(img.numpy(), path)
