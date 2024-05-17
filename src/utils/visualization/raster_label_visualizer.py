# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from PIL import Image, ImageColor
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import os
from io import BytesIO
from typing import Union, Tuple
import matplotlib
import torch
matplotlib.use('Agg')


class RasterLabelVisualizer(object):
    """Visualizes raster mask labels and predictions."""

    def __init__(self, label_map: Union[str, dict]):
        """Constructs a raster label visualizer.

        Args:
            label_map: a path to a JSON file containing a dict, or a dict.
              The dict needs to have two fields:

            num_to_name {
                numerical category (str or int) : display name (str)
            }

            num_to_color {
                numerical category (str or int) : color representation
                (an object that matplotlib.colors recognizes
                as a color; additionally a (R, G, B) tuple or list with uint8
                  values will also be parsed)
            }
        """
        if isinstance(label_map, str):
            assert os.path.exists(label_map)
            with open(label_map) as f:
                label_map = json.load(f)

        assert 'num_to_name' in label_map
        assert isinstance(label_map['num_to_name'], dict)
        assert 'num_to_color' in label_map
        assert isinstance(label_map['num_to_color'], dict)

        self.num_to_name = RasterLabelVisualizer._dict_key_to_int(
            label_map['num_to_name'])
        self.num_to_color = RasterLabelVisualizer._dict_key_to_int(
            label_map['num_to_color'])

        assert len(self.num_to_color) == len(self.num_to_name)
        self.num_classes = len(self.num_to_name)

        # check for duplicate names or colors
        assert len(set(self.num_to_color.values())) == self.num_classes, \
            'There are duplicate colors in the colormap'
        assert len(set(self.num_to_name.values())) == self.num_classes, \
            'There are duplicate class names in the colormap'

        self.num_to_color = \
            RasterLabelVisualizer.standardize_colors(self.num_to_color)

        # create the custom colormap according to colors defined in label_map
        required_colors = []
        # key is originally a string
        # num already cast to int
        for num, color_name in sorted(self.num_to_color.items(),
                                      key=lambda x: x[0]):
            rgb = mcolors.to_rgb(mcolors.CSS4_COLORS[color_name])
            # mcolors.to_rgb is to [0, 1] values;
            # ImageColor.getrgb gets [1, 255] values
            required_colors.append(rgb)

        self.colormap = mcolors.ListedColormap(required_colors)
        # vmin and vmax appear to be inclusive,
        # so if there are a total of 34 classes,
        # class 0 to class 33 each maps to a color
        self.normalizer = mcolors.Normalize(vmin=0, vmax=self.num_classes - 1)

        self.color_matrix = self._make_color_matrix()

    @staticmethod
    def _dict_key_to_int(d: dict) -> dict:
        return {int(k): v for k, v in d.items()}

    def _make_color_matrix(self) -> np.ndarray:
        """Creates a color matrix of dims (num_classes, 3),
          where a row corresponds to the RGB values of each class.
        """
        matrix = []
        for num, color in sorted(self.num_to_color.items(),
                                 key=lambda x: x[0]):
            rgb = RasterLabelVisualizer.matplotlib_color_to_uint8_rgb(color)
            matrix.append(rgb)
        matrix = np.array(matrix)

        assert matrix.shape == (self.num_classes, 3)

        return matrix

    @staticmethod
    def standardize_colors(num_to_color: dict) -> dict:
        """Return a new dict num_to_color with colors verified.
          uint8 RGB tuples are converted to a hex string
        as matplotlib.colors do not accepted uint8 intensity values"""
        new = {}
        for num, color in num_to_color.items():
            if mcolors.is_color_like(color):
                new[num] = color
            else:
                # try to see if it's a (r, g, b) tuple or list of uint8 values
                assert len(color) == 3 or len(color) == 4, \
                    f'Color {color} is specified as a tuple or list but is \
                          not of length 3 or 4'
                for c in color:
                    assert isinstance(
                        c, int) and 0 < c < 256, f'RGB value {c} is out of \
                            range'

                new[num] = RasterLabelVisualizer.uint8_rgb_to_hex(
                    color[0], color[1], color[3])  # drop any alpha values
        assert len(new) == len(num_to_color)
        return new

    @staticmethod
    def uint8_rgb_to_hex(r: int, g: int, b: int) -> str:
        """Convert RGB values in uint8 to a hex color string

        Reference
        https://codereview.stackexchange.com/questions/229282/
        performance-for-simple-code-that-converts-a-rgb-tuple-to-hex-string
        """
        return f'#{r:02x}{g:02x}{b:02x}'

    @staticmethod
    def matplotlib_color_to_uint8_rgb(color: Union[str, tuple, list]
                                      ) -> Tuple[int, int, int]:
        """Converts any matplotlib recognized color representation to
        (R, G, B) uint intensity values

        Need to use matplotlib, which recognizes different color formats,
        to convert to hex,
        then use PIL to convert to uint8 RGB. matplotlib does not support
        the uint8 RGB format
        """
        color_hex = mcolors.to_hex(color)
        # '#DDA0DD' to (221, 160, 221); alpha silently dropped
        color_rgb = ImageColor.getcolor(color_hex, 'RGB')
        return color_rgb

    def show_label_raster(self, label_raster: Union[Image.Image, np.ndarray],
                          size: Tuple[int, int] = (10, 10)
                          ) -> Tuple[Image.Image, BytesIO]:
        """Visualizes a label mask or hardmax predictions of a model,
          according to the category color map
        provided when the class was initialized.

        The label_raster provided needs to contain values in [0, num_classes].

        Args:
            label_raster: 2D numpy array or PIL Image where each number
            indicates the pixel's class
            size: matplotlib size in inches (h, w)

        Returns:
            (im, buf) - PIL image of the matplotlib figure, and a BytesIO buf
              containing the matplotlib Figure
            saved as a PNG
        """
        if not isinstance(label_raster, np.ndarray):
            label_raster = np.asarray(label_raster)

        label_raster = label_raster.squeeze()
        assert len(label_raster.shape) == 2, \
            'label_raster provided has more than 2 dimensions after squeezing'

        label_raster.astype(np.uint8)

        # min of 0, which is usually empty / no label
        assert np.min(label_raster) >= 0, \
            f'Invalid value for class label: {np.min(label_raster)}'

        # non-empty, actual class labels start at 1
        assert np.max(label_raster) <= self.num_classes, \
            f'Invalid value for class label: {np.max(label_raster)}'

        _ = plt.figure(figsize=size)
        _ = plt.imshow(label_raster, cmap=self.colormap,
                       norm=self.normalizer, interpolation='none')

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        im = Image.open(buf)
        return im, buf

    def prepare_for_vis(self, logger, phase, dataset, sample_ids, model,
                        epoch, device):
        """This method creates logging image files for tensorboard visualization"""
        for item in sample_ids:
            data = dataset[item]

            c, h, w = data['pre_image'].size()
            pre = data['pre_image'].reshape(1, c, h, w)
            post = data['post_image'].reshape(1, c, h, w)

            scores = model(pre.to(device=device), post.to(device=device))

            if (epoch == 1):
                gt = {}
                gt['pre_img'] = data["pre_image_orig"]
                gt['post_img'] = data["post_image_orig"]
                gt['bld_mask'] = data["building_mask"].reshape(1, h, w)
                true_dmg_mask = data["damage_mask"]
                im, _ = self.show_label_raster(
                    np.array(true_dmg_mask), size=(5, 5))
                gt['dmg_mask'] = transforms.ToTensor()(
                    transforms.ToPILImage()(np.array(im)).convert("RGB"))
                for key, img in gt.items():
                    tag = f'{"gt"}_{key}_{phase}_id_{item}'
                    logger.add_image(tag, img, epoch, dataformats='CHW')

            tp = {}
            # compute predictions & confusion metrics
            softmax = torch.nn.Softmax(dim=1)
            tp['pred_seg_pre'] = torch.argmax(softmax(scores[0]), dim=1)
            tp['pred_seg_post'] = torch.argmax(softmax(scores[1]), dim=1)
            preds_cls = torch.argmax(softmax(scores[2]), dim=1)
            im, _ = self.show_label_raster(
                preds_cls.cpu().numpy(), size=(5, 5))
            tp['pred_dmg_cls'] = transforms.ToTensor()(
                transforms.ToPILImage()(np.array(im)).convert("RGB"))
            for key, img in tp.items():
                tag = f'{"tp"}_{key}_{phase}_id_{item}'
                logger.add_image(tag, img, epoch, dataformats='CHW')
