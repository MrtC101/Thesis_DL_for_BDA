import os
import cv2
import pandas as pd
import numpy as np
import torch
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib import gridspec
from torchvision.utils import draw_bounding_boxes
from utils.common.pathManager import FilePath
from utils.visualization.label_to_color import LabelDict
from utils.visualization.label_mask_visualizer import LabelMaskVisualizer

matplotlib.use("Agg")

labels_dict = LabelDict()


def bbs_by_level_figures(dis_id: str, tile_id: str, bbs_df: pd.DataFrame, save_folder: str):
    """Generates a png transparent background image for each class of bounding boxes.
    Args:
        dis_id: disaster label.
        tile_id: tile label.
        bbs_df: `pd.Dataframe` with all bounding boxes of an image.
        save_folder: path to the folder where to save the image.
    """
    for cls in np.unique(bbs_df["label"]):
        # Filtrar las bounding boxes para la clase actual
        cur_df = bbs_df[bbs_df["label"] == cls]
        boxes = torch.tensor(
            cur_df[['x1', 'y1', 'x2', 'y2']].values, dtype=torch.float)

        # Crear una lista de etiquetas de la clase actual
        color = labels_dict.get_color_by_key(cls)

        # Crear una imagen en blanco para dibujar las bounding boxes
        # Asegurarse de que tenga 3 canales de color
        img_tensor = torch.zeros((3, 1024, 1024), dtype=torch.uint8)

        # Dibujar las bounding boxes en la imagen
        image_with_boxes = draw_bounding_boxes(img_tensor, boxes, colors=color, width=2)

        # Convertir el tensor a una imagen numpy
        img_np = image_with_boxes.permute(1, 2, 0).numpy()

        # Crear una máscara alfa para el fondo transparente
        alpha = np.any(img_np > 0, axis=2).astype(np.uint8) * 255
        # Añadir el canal alfa a la imagen
        img = np.dstack((img_np[:, :, 2::-1], alpha))

        # Guardar la imagen usando cv2.imwrite
        file_path = os.path.join(
            save_folder, f"{dis_id}_{tile_id}_{cls}_bbs.png")
        cv2.imwrite(file_path, img)


def addPlotTable(ax: matplotlib.axes, curr_table: pd.DataFrame, fontsize: int,
                 col_width: list, row_height: float) -> matplotlib.axes:
    """Plots the given table in the given matplotlib figure axis.
    Args:
        ax: `matplotlib.axes` where to plot the table.
        curr_table: `pd.Dataframe` the corresponding table content.
        fontsize: font size
        col_width: list of floats to use for the width of each row inside table.
        row_height: one float to use as the height for all rows inside the table.
    Returns:
        matplotlib.axes: the axes with the plotted table.
    """
    table = ax.table(
        colWidths=col_width,
        cellText=curr_table.values,
        colLabels=curr_table.columns,
        rowLabels=curr_table.index,
        cellLoc='center',
        loc='center'
    )
    table.set_fontsize(fontsize)
    curr_colors = ["darkgrey"]
    curr_colors.extend([labels_dict.get_color_by_key(key)
                        for key in curr_table["Level"]])
    # Aplicar estilos a la tabla
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Negritas para los encabezados
            cell.set_linewidth(2.0)
            cell.get_text().set_fontweight('bold')
        else:
            cell.set_facecolor(curr_colors[i])
        cell.set_height(row_height)
    return table


def comparative_figure(dis_id: str, tile_id: str, pre_img: torch.Tensor, post_img: torch.Tensor,
                       pred_mask: torch.Tensor, gt_table: pd.DataFrame, pd_table: pd.DataFrame,
                       save_path: str):
    """Plots few figures.
    1. save the pre-disaster image.
    2. save the post-disaster image.
    3. save the predicted damage mask image.
    4. plots the ground truth building count table.
    5. plots the predicted building count table.
    6. Plots a comparative figure with 1-5 thins inside it.

    Args:
        dis_id : disaster label.
        tile_id : tile label.
        pre_img : pre-disaster image as a `torch.Tensor`
        post_img : post-disaster image as a `torch.Tensor`
        pre_mask : predicted damage mask as a `torch.Tensor`
        gt_table : `pd.Dataframe` with the ground truth buildings.
        pd_table : `pd.Dataframe` with the predicted buildings.
        save_path : path to the directory where to output all figures.
    """
    title_size = 30
    subtitle_size = 25

    fig = plt.figure(figsize=(1524 / 100, 1024 / 100), dpi=100)
    gs = gridspec.GridSpec(2, 3, height_ratios=[0.7, 0.4])
    plt.suptitle(f"Disaster {dis_id} {tile_id}",
                 fontsize=title_size, fontweight="bold")

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(pre_img)
    ax1.axis('off')
    ax1.set_title('Pre-disaster Imagen', fontsize=subtitle_size)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(post_img)
    ax2.axis('off')
    ax2.set_title('Post-disaster Imagen', fontsize=subtitle_size)

    ax3 = fig.add_subplot(gs[0, 2])
    draw_mask = LabelMaskVisualizer.draw_label_img(pred_mask)
    ax3.imshow(draw_mask)
    ax3.axis('off')
    ax3.set_title('Predicted Damage Mask', fontsize=subtitle_size)

    ax4 = fig.add_subplot(gs[1, 0])  # Span all columns in the second row
    ax4.axis('off')
    addPlotTable(ax4, gt_table, subtitle_size, [
                 0.6]*len(gt_table.columns), 0.16)
    ax4.set_title('True building count', fontsize=subtitle_size)

    ax5 = fig.add_subplot(gs[1, 2])  # Span all columns in the second row
    ax5.axis('off')
    addPlotTable(ax5, pd_table, subtitle_size, [
                 0.6]*len(pd_table.columns), 0.16)
    ax5.set_title('Predicted building count', fontsize=subtitle_size)

    ax6 = fig.add_subplot(gs[1, 1])  # Span all columns in the second row
    ax6.axis('off')

    os.makedirs(save_path, exist_ok=True)
    files = os.path.join(
        save_path, f"{dis_id}-{tile_id}_predicted_dmg_mask.png")
    plt.tight_layout()
    plt.savefig(files, format='png', bbox_inches='tight', pad_inches=0.2)
    plt.close()


def superposed_img(dis_id: str, tile_id: str, pre_img: torch.Tensor, pred_img: torch.Tensor,
                   save_path: str):
    """Creates a matplotlib figure that shows 'pre_img','post_img and 'pred_img'
    in the first row and 'curr_table' in the second row.

    Args:
        dis_id : disaster label.
        tile_id : tile label.
        pre_img : pre-disaster image as a `torch.Tensor`
        post_img : post-disaster image as a `torch.Tensor`
        pre_mask : predicted damage mask as a `torch.Tensor`
        save_path : path to the directory where to output the figure.
    """
    superposed_image = cv2.addWeighted(
        pre_img.numpy(), 0.8, pred_img.numpy(), 0.5, 0)
    file_path = os.path.join(
        save_path, f"{dis_id}_{tile_id}_superposed.png")
    LabelMaskVisualizer.save_arr_img(superposed_image, file_path)


def plot_roc_curves(type, roc_curves, out):
    """Plots the ROC curve
    Args:
        type: Name of the analysis level. ("object" or "pixel").
        pr_curves: Precomputed roc curve for each class.
        out: Path to the directory where to save the last thing.
    """
    plt.figure(figsize=(10, 10), dpi=100)

    for label, (fpr, tpr, auc_value) in roc_curves.items():
        c = labels_dict.get_color_by_num(label)
        plt.plot(fpr, tpr, color=c,
                 label=f'Class {label} (AUC = {auc_value:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('False Positive Rate (FPR)', fontsize=18)
    plt.ylabel('True Positive Rate (TPR)', fontsize=18)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=20)
    plt.subplots_adjust(left=0.1, right=0.6, top=0.6, bottom=0.1)
    plt.legend(loc='lower right', fontsize=16, bbox_to_anchor=(1.7, 0))
    plt.grid(True)
    # especificiar si es parche o tile.
    plt.savefig(os.path.join(out, f"{type}_ROC_curves.png"))


def plot_pr_curves(type: str, pr_curves: dict, out: str):
    """Plot the PR curve plot
    Args:
        type: Name of the analysis level. ("object" or "pixel").
        pr_curves: Precomputed precision and recall curve for each class.
        out: path to the directory where to save the last thing.
    """
    plt.figure(figsize=(10, 10), dpi=100)

    for label, (r, p, ppd, pr_value) in pr_curves.items():
        c = labels_dict.get_color_by_num(label)
        plt.plot(r, p, color=c,
                 label=f'Class {label} (AUC = {pr_value:.2f})')
        plt.plot([0, 1], [ppd, ppd], color=c, linestyle='--',
                 label=f'Class {label} (BP = {ppd:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.title('Precision-Recall Curve', fontsize=18)
    plt.subplots_adjust(left=0.1, right=0.6, top=0.6, bottom=0.1)
    plt.legend(loc='lower right', fontsize=16, bbox_to_anchor=(1.7, 0))
    plt.grid(True)
    # especificiar si es parche o tile.
    plt.savefig(os.path.join(out, f"{type}_PR_curves.png"))


def plot_loss(tr_l: pd.DataFrame, vl_l: pd.DataFrame, metric_dir: FilePath):
    """Plots the loss curve over the train and validation loss over epochs

    Args:
        tr_l: Dataframe with training loss over epochs.
        vl_l: Dataframw with validation loss over epochs.
        metric_dir: path to the directory where to save the figure.
    """
    tr_l = tr_l.set_index("epoch")
    vl_l = vl_l.set_index("epoch")
    tr_l = tr_l.rename(columns={"loss": "train_loss"})
    vl_l = vl_l.rename(columns={"loss": "val_loss"})
    f_df = pd.concat([tr_l, vl_l], axis=1)

    ax: matplotlib.axes.Axes
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    ax.plot(f_df, label=f_df.columns)
    ax.set_title("Loss Over Epoch")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    fig.savefig(metric_dir.join("loss_plots.png"))


def plot_harmonic_mean(tr_m: pd.DataFrame, vl_m: pd.DataFrame, metric_dir: FilePath):
    """Plots a figure with the harmonic mean over epochs for validation and training.

    Args:
        tr_m: Dataframe with training metrics over the epochs.
        vl_m: Dataframe with validation metrics over the epochs.
        metric_dir: path to the folder where to save the figure.
    """
    tr = tr_m[tr_m["class"] == 0][["epoch", "f1_harmonic_mean"]]
    tr = tr.rename(columns={"f1_harmonic_mean": "f1_h_train"})
    tr = tr.set_index("epoch")
    vl = vl_m[vl_m["class"] == 0][["epoch", "f1_harmonic_mean"]]
    vl = vl.rename(columns={"f1_harmonic_mean": "f1_h_val"})
    vl = vl.set_index("epoch")
    metrics_df = pd.concat([tr, vl], axis=1)
    ax: matplotlib.axes.Axes
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    ax.plot(metrics_df, label=metrics_df.columns)
    ax.set_title('F1 Harmonic Mean Over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Harmonic Mean')
    ax.legend()
    ax.grid(True)
    fig.savefig(metric_dir.join("f1_h_mean_plots.png"))


def plot_metric_per_class(tr_m: pd.DataFrame, metric: str, prefix: str, metric_dir: FilePath):
    """Plot a figure of each class metric evolution over epochs.

    Args:
        tr_m: Dataframe with training metrics for each epoch.
        metric: name of the metric to be plot
        prefix: prefix to use for the save file. "dmg" or "bld"
        metric_dir: path to the directory where to save the metrics.
    """
    tr_m['class'] = tr_m['class'].apply(labels_dict.get_key_by_num)
    tr = tr_m.pivot(index='epoch', columns='class', values=metric)
    ax: matplotlib.axes.Axes
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    ax.plot(tr, label=tr.columns)
    ax.set_title(f'{metric.capitalize()} per Class over Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'{metric}')
    ax.legend()
    ax.grid(True)
    fig.savefig(metric_dir.join(f"{prefix}_{metric}_plots.png"))
