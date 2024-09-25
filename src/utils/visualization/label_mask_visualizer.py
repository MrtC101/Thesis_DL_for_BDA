# Modificaciones (c) 2024 Martín Cogo Belver.
# Martín Cogo Belver has rights reserved over this modifications.
import cv2
import numpy as np
import matplotlib
import torch
import matplotlib.colors as mcolors
from utils.visualization.label_to_color import LabelDict

matplotlib.use("Agg")  # Usar backend 'Agg' para entornos sin display


class LabelMaskVisualizer:
    """Visualizes raster mask labels and predictions."""
    @staticmethod
    def draw_label_img(label_tensor: torch.Tensor) -> torch.Tensor:
        # Asumimos que LabelDict es una clase que define un mapeo de etiquetas a colores
        label_map = LabelDict()
        required_colors = [mcolors.to_rgb(mcolors.CSS4_COLORS[color])
                           for color in label_map.color_list]
        colormap = mcolors.ListedColormap(required_colors)

        # Asegurarse de que label_tensor tiene la forma correcta
        if len(label_tensor.shape) == 3 and label_tensor.shape[0] == 1:
            label_tensor = label_tensor.squeeze(0)

        # Crear una imagen en blanco con el tamaño adecuado
        img = torch.zeros(
            (label_tensor.shape[1], label_tensor.shape[0], 3), dtype=torch.uint8)

        # Aplicar colores a cada etiqueta única
        for label_i in label_tensor.unique():
            color = colormap(int(label_i))[:3]
            color_tensor = (torch.tensor(color) * 255).to(torch.uint8)
            img[label_tensor == label_i] = color_tensor
        return img

    @staticmethod
    def save_arr_img(img: np.ndarray, path: str):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path, img)

    @staticmethod
    def save_tensor_img(img: torch.Tensor, path: str):
        LabelMaskVisualizer.save_arr_img(img.numpy(), path)
