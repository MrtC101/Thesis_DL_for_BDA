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
        self.colormap = self.create_colormap(self.label_map)
        self.normalizer = mcolors.Normalize(vmin=0, vmax=self.num_classes - 1)

    @staticmethod
    def create_colormap(label_map: LabelDict) -> mcolors.ListedColormap:
        """Returns a matplotlib color map."""
        required_colors = [mcolors.to_rgb(mcolors.CSS4_COLORS[color])\
                            for color in label_map.color_list]
        return mcolors.ListedColormap(required_colors)

    @staticmethod
    def fig_to_array(fig: matplotlib.figure.Figure) -> np.ndarray:
        """Converts a matplotlib figure to a np.array."""
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        arr = buf.reshape(h, w, 3)
        return arr

    def draw_label_img(self, label_tensor: torch.Tensor) -> np.ndarray:
        """Visualizes a label mask or hardmax predictions of a model."""
        if len(label_tensor.shape) == 3 and label_tensor.shape[0] == 1:
            label_tensor = label_tensor.squeeze(0)

        assert len(label_tensor.shape) == 2,\
            "Only 2 dimensional tensor admitted."
        label_arr = label_tensor.numpy()

        # Ensure valid label values
        assert np.min(label_arr) >= 0,\
            f'Invalid value for class label: {np.min(label_arr)}'
        assert np.max(label_arr) <= self.num_classes,\
            f'Invalid value for class label: {np.max(label_arr)}'
        
        fig, ax = plt.subplots(figsize=(1024 / 100, 1024 / 100), dpi=100, facecolor='none')
        ax.imshow(label_arr, cmap=self.colormap, norm=self.normalizer, interpolation='none')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        img_arr = self.fig_to_array(fig)
        plt.close(fig)
        return img_arr
    
    @staticmethod
    def save_arr_img(img : np.ndarray, path : str):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path, img)
    
    @staticmethod
    def save_tensor_img(img : torch.Tensor, path : str):
        LabelMaskVisualizer.save_arr_img(img.numpy(), path)