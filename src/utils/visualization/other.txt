# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

def get_tiff_colormap(self) -> dict:
    """Returns the object to pass to rasterio dataset object's write_colormap() function,
    which is a dict mapping int values to a tuple of (R, G, B)

    See https://rasterio.readthedocs.io/en/latest/topics/color.html for writing the TIFF colormap
    """
    colormap = {}
    for num, color in self.num_to_color.items():
        # uint8 RGB required by TIFF
        colormap[num] = RasterLabelVisualizer.matplotlib_color_to_uint8_rgb(
            color)
    return colormap


def get_tool_colormap(self) -> str:
    """Returns a string that is a JSON of a list of items specifying the name and color
    of classes. Example:
    "[
        {"name": "Water", "color": "#0000FF"},
        {"name": "Tree Canopy", "color": "#008000"},
        {"name": "Field", "color": "#80FF80"},
        {"name": "Built", "color": "#806060"}
    ]"
    """
    classes = []
    for num, name in sorted(self.num_to_name.items(), key=lambda x: int(x[0])):
        color = self.num_to_color[num]
        color_hex = mcolors.to_hex(color)
        classes.append({
            'name': name,
            'color': color_hex
        })
    classes = json.dumps(classes, indent=4)
    return classes


@staticmethod
def plot_colortable(name_to_color: dict, title: str, sort_colors: bool = False, emptycols: int = 0) -> plt.Figure:
    """
    function taken from https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    """

    cell_width = 212
    cell_height = 22
    swatch_width = 70
    margin = 12
    topmargin = 40

    # Sort name_to_color by hue, saturation, value and name.
    if sort_colors is True:
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                         name)
                        for name, color in name_to_color.items())
        names = [name for hsv, name in by_hsv]
    else:
        names = list(name_to_color)

    n = len(names)
    ncols = 4 - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 80  # other numbers don't seem to work well

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin / width, margin / height,
                        (width - margin) / width, (height - topmargin) / height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize=24, loc='left', pad=10)

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        swatch_end_x = cell_width * col + swatch_width
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.hlines(y, swatch_start_x, swatch_end_x,
                  color=name_to_color[name], linewidth=18)

    return fig


def plot_color_legend(self, legend_title: str = 'Categories') -> plt.Figure:
    """Builds a legend of color block, numerical categories and names of the categories.

    Returns:
        a matplotlib.pyplot Figure
    """
    label_map = {}
    for num, color in self.num_to_color.items():
        label_map['{} {}'.format(num, self.num_to_name[num])] = color

    fig = RasterLabelVisualizer.plot_colortable(
        label_map, legend_title, sort_colors=False, emptycols=3)
    return fig


def visualize_softmax_predictions(self, softmax_preds: np.ndarray) -> np.ndarray:
    """Visualizes softmax probabilities in RGB according to the class label's assigned colors

    Args:
        softmax_preds: numpy array of dimensions (batch_size, num_classes, H, W) or (num_classes, H, W)

    Returns:
        numpy array of size ((batch_size), H, W, 3). You may need to roll the last axis to in-front before
        writing to TIFF

    Raises:
        ValueError when the dimension of softmax_preds is not compliant
    """

    assert len(softmax_preds.shape) == 4 or len(softmax_preds.shape) == 3

    # row the num_classes dimension to the end
    if len(softmax_preds.shape) == 4:
        assert softmax_preds.shape[1] == self.num_classes
        softmax_preds_transposed = np.transpose(
            softmax_preds, axes=(0, 2, 3, 1))
    elif len(softmax_preds.shape) == 3:
        assert softmax_preds.shape[0] == self.num_classes
        softmax_preds_transposed = np.transpose(
            softmax_preds, axes=(1, 2, 0))
    else:
        raise ValueError(
            'softmax_preds does not have the required length in the dimension of the classes')

    # ((batch_size), H, W, num_classes) @ (num_classes * 3) = ((batch_size), H, W, 3)
    colored_view = softmax_preds_transposed @ self.color_matrix
    return colored_view


def visualize_matrix(matrix: np.ndarray) -> Image.Image:
    """Shows a 2D matrix of RGB or greyscale values as a PIL Image.

    Args:
        matrix: a (H, W, 3) or (H, W) numpy array, representing a colored or greyscale image

    Returns:
        a PIL Image object
    """
    assert len(matrix.shape) in [2, 3]

    image = Image.fromarray(matrix)
    return image
