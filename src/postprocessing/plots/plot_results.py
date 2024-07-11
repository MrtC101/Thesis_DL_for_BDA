import os
import cv2
import matplotlib.pyplot as plt
import matplotlib
import torch
matplotlib.use("Agg")
from matplotlib import gridspec, patches
import numpy as np
import pandas as pd
from utils.visualization.label_mask_visualizer import LabelMaskVisualizer
from utils.visualization.label_to_color import LabelDict
from torchvision.utils import draw_bounding_boxes

labels_dict = LabelDict()

def bbs_by_level_figures(dis_id, tile_id, bbs_df: pd.DataFrame, save_folder):
    """
    Generates a png transparent background image for each class of bounding boxes.
    (La idea es tener una imagen con las bounding boxes de con la misma clase)
    """
    for cls in np.unique(bbs_df["label"]):
        # Filtrar las bounding boxes para la clase actual
        cur_df = bbs_df[bbs_df["label"] == cls]
        boxes = torch.tensor(cur_df[['x1', 'y1', 'x2', 'y2']].values, dtype=torch.float)
        labels = [cls] * len(cur_df)  # Crear una lista de etiquetas de la clase actual
        color = labels_dict.get_color_by_key(cls)

        # Crear una imagen en blanco para dibujar las bounding boxes
        img_tensor = torch.zeros((3, 1024, 1024), dtype=torch.uint8)  # Asegurarse de que tenga 3 canales de color

        # Dibujar las bounding boxes en la imagen
        image_with_boxes = draw_bounding_boxes(img_tensor, boxes, colors=color, width=2,
                                               #labels=labels
                                               )

        # Convertir el tensor a una imagen numpy
        img_np = image_with_boxes.permute(1, 2, 0).numpy()
        
        # Crear una máscara alfa para el fondo transparente
        alpha = np.any(img_np > 0, axis=2).astype(np.uint8) * 255
        img = np.dstack((img_np[:, :, 2::-1], alpha))  # Añadir el canal alfa a la imagen

        # Guardar la imagen usando cv2.imwrite
        file_path = os.path.join(save_folder, f"{dis_id}_{tile_id}_{cls}_bbs.png")
        cv2.imwrite(file_path, img)

def addPlotTable(ax, curr_table, fontsize, col_width, row_height):
    """Plots the given table in the given matplotlib figure axis."""    
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

def comparative_figure(dis_id,tile_id,pre_img,post_img, pred_mask, gt_table, pd_table, save_path):
    title_size = 30
    subtitle_size = 25
    
    viz = LabelMaskVisualizer()
    fig = plt.figure(figsize=(1524 / 100, 1024 / 100), dpi=100)
    gs = gridspec.GridSpec(2,3,height_ratios=[0.7, 0.4])
    plt.suptitle(f"Disaster {dis_id} {tile_id}", fontsize=title_size, fontweight="bold")

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(pre_img)
    ax1.axis('off')
    ax1.set_title('Pre-disaster Imagen', fontsize=subtitle_size)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(post_img)
    ax2.axis('off')
    ax2.set_title('Post-disaster Imagen', fontsize=subtitle_size)

    ax3 = fig.add_subplot(gs[0, 2])
    if(len(pred_mask.shape)>2):
        pred_mask = pred_mask.reshape(-1)
    ax3.imshow(pred_mask, cmap=viz.colormap, norm=viz.normalizer, interpolation='none')
    ax3.axis('off')
    ax3.set_title('Predicted Damage Mask', fontsize=subtitle_size)
    
    ax4 = fig.add_subplot(gs[1, 0])  # Span all columns in the second row
    ax4.axis('off')
    addPlotTable(ax4, gt_table, subtitle_size, [0.6]*len(gt_table.columns), 0.16)
    ax4.set_title('Predicted building count', fontsize=subtitle_size)

    ax5 = fig.add_subplot(gs[1, 2])  # Span all columns in the second row
    ax5.axis('off')
    addPlotTable(ax5, pd_table, subtitle_size, [0.6]*len(pd_table.columns), 0.16)
    ax5.set_title('Predicted building count', fontsize=subtitle_size)

    ax6 = fig.add_subplot(gs[1, 1])  # Span all columns in the second row
    ax6.axis('off')
    
    os.makedirs(save_path, exist_ok=True)
    files = os.path.join(save_path, f"{dis_id}-{tile_id}_predicted_dmg_mask.png")
    plt.tight_layout()
    plt.savefig(files, format='png', bbox_inches='tight', pad_inches=0.2)
    plt.close()

def generate_figures(dis_id, tile_id, tile_dict, pred_mask, gt_table, pd_table, save_path):
    """Creates a matplotlib figure that shows 'pre_img','post_img and 'pred_img'
        in the first row and 'curr_table' in the second row."""
    
    #Save images.
    os.makedirs(save_path, exist_ok=True)
    
    pre_img = tile_dict["pre_img"].numpy()
    pre_path = os.path.join(save_path, f"{dis_id}_{tile_id}_pre_disaster.png")
    LabelMaskVisualizer.save_arr_img(pre_img, pre_path)

    post_img = np.array(tile_dict["post_img"])
    post_path = os.path.join(save_path, f"{dis_id}_{tile_id}_post_disaster.png")
    LabelMaskVisualizer.save_arr_img(post_img, post_path)
    
    pred_path = os.path.join(save_path, f"{dis_id}_{tile_id}_pred_damage_mask.png")
    pred_img = LabelMaskVisualizer().draw_label_img(pred_mask)
    LabelMaskVisualizer.save_arr_img(pred_img, pred_path)
    
    comparative_figure(dis_id, tile_id, pre_img, post_img, pred_mask, gt_table[1],
                        pd_table[1], save_path)

    gt_folder = os.path.join(save_path, "gt_bbs")
    os.makedirs(gt_folder, exist_ok=True)
    bbs_by_level_figures(dis_id, tile_id, gt_table[0], gt_folder)

    pred_folder = os.path.join(save_path, "pd_bbs")
    os.makedirs(pred_folder, exist_ok=True)
    bbs_by_level_figures(dis_id, tile_id, pd_table[0], pred_folder)

    superposed_image = cv2.addWeighted(pre_img, 0.8, pred_img, 0.5, 0)
    file_path = os.path.join(save_path, f"{dis_id}_{tile_id}_superposed.png")
    LabelMaskVisualizer.save_arr_img(superposed_image, file_path)

def plot_roc_curves(type, roc_curves, out):
    plt.figure(figsize=(10, 8))
    
    for label, (fpr, tpr, auc_value) in roc_curves.items():
        plt.plot(fpr, tpr, label=f'Class {label} (AUC = {auc_value:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    #especificiar si es parche o tile.
    plt.savefig(os.path.join(out,f"{type}_ROC_curves.png"))

def plot_pr_curves(type, pr_curves, out):
    plt.figure(figsize=(10, 8))
    
    for label, (fpr, tpr, pr_value) in pr_curves.items():
        plt.plot(fpr, tpr, label=f'Class {label} (AP = {pr_value:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    #especificiar si es parche o tile.
    plt.savefig(os.path.join(out,f"{type}_PR_curves.png"))
