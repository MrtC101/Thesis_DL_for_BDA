import os
from matplotlib import gridspec
import matplotlib.pyplot as plt
import pandas as pd
from utils.visualization.raster_label_visualizer import RasterLabelVisualizer

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
    colors = ['darkgray','limegreen','orange','red','gray']
    table.set_fontsize(fontsize)
    # Aplicar estilos a la tabla
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Negritas para los encabezados
            cell.set_linewidth(2.0)
            cell.get_text().set_fontweight('bold')
        else:
            cell.set_facecolor(colors[i-1])
        cell.set_height(row_height)
    return table

def save_results(dis_id, tile_id, pre_img, post_img, pred_mask,
                  curr_table : pd.DataFrame, label_map_json : str, save_path:str):
    """Creates a matplotlib figure that shows 'pre_img','post_img and 'pred_img'
        in the first row and 'curr_table' in the second row."""
    
    viz = RasterLabelVisualizer(label_map_json)

    #fig = plt.figure(figsize=(30.72, 20.48), dpi=100) GOOD RESOLUTION
    
    fig = plt.figure(figsize=(15.36, 10.24), dpi=100)
    title_size = 30
    subtitle_size = 25
    gs = gridspec.GridSpec(2,3,height_ratios=[0.7, 0.4])
    plt.suptitle("1024x1024 Images Prediction Result", fontsize=title_size, fontweight="bold")

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

    ax4 = fig.add_subplot(gs[1, :])  # Span all columns in the second row
    ax4.axis('off')
    ax4.set_title(f"Disaster {dis_id} {tile_id}", fontsize=title_size, fontweight="bold")
    addPlotTable(ax4, curr_table, subtitle_size, [0.47]*len(curr_table.columns), 0.16)

    os.makedirs(save_path,exist_ok=True)
    files = os.path.join(save_path, f"{dis_id}-{tile_id}_predicted_dmg_mask.png")
    plt.tight_layout()
    plt.savefig(files, format='png', bbox_inches='tight', pad_inches=0.2)
    plt.close()

def PlotSuppersoed():
    # Plot 

def plotBBSOnly(bbs_list):
    """
        Generates a png transparent background image for each class of bounding boxes.
        (La idea es tener una iamgen con las bounding boxes de con la misma clase)
    """
    # Crear una figura y un eje
    fig, ax = plt.subplots()

    # Graficar los datos
    ax.plot(x, y)

    # Configurar el fondo transparente
    fig.patch.set_alpha(0.0)  # Fondo de la figura transparente
    ax.patch.set_alpha(0.0)   # Fondo del área de los ejes transparente

    bbs_img = plt.savefig('imagen_transparente.png', transparent=True)
    return bbs_img

def addPlotBBS(mask, bbs_img):
    """ Superposes a mask with a bb's image."""
    pass

def addPlotSuperposed(pre_img, dmg_mask):
    """Superposes the pre_img with the dmg_mask."""
    pass


# superposse images
def supp(img, target):
    img_disaster = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    target[target[:, :, 0] == 1] = [70, 70, 70]
    target[target[:, :, 0] == 2] = [178, 255, 102]
    target[target[:, :, 0] == 3] = [194, 255, 94]
    target[target[:, :, 0] == 4] = [255, 139, 51]
    target[target[:, :, 0] == 5] = [255, 19, 19]

    alpha = 0.7  # Ajustar la transparencia de la segunda imagen
    superposed_image = cv2.addWeighted(img_disaster, 0.8, target, alpha, 0)
    return superposed_image


def visualize_bounding_boxes(ax, bounding_boxes):
    damaged = bounding_boxes[bounding_boxes["label"] != "no-damage"]
    # Dibujar cada bounding box y etiqueta en la imagen
    for i in range(len(damaged)):
        row = damaged.iloc[i]
        x, y = (row["x1"], row["y1"])
        z, w = (row["x2"], row["y2"])
        rect = patches.Rectangle(
            (x, y), z-x, w-y, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(z, w, row["label"], color='white', fontsize=8,
                fontweight=750, fontfamily='sans-serif',
                verticalalignment='top',
                bbox=dict(facecolor='red', alpha=0.3,
                          edgecolor='none', pad=3,))


def showImages(img_pre, img_post, target_pre, target_post, bounding_boxes):
    fig, axes = plt.subplots(1, 2, figsize=(22, 22))

    target_pre_img = supp(img_pre, target_pre)
    target_post_img = supp(img_post, target_post)
    # Mostrar las imágenes

    axes[0].imshow(cv2.cvtColor(target_pre_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Mascaras de edificios antes del desastre',
                      fontsize=17, fontweight=800)
    axes[0].axis('off')
    axes[0].text(400, 20, '(Los colores son aleatorios)',
                 fontweight=400, bbox=dict(facecolor='white'))

    axes[1].imshow(target_post_img)
    visualize_bounding_boxes(axes[1], bounding_boxes)
    axes[1].set_title('Edificios dañados después del desastre',
                      fontsize=17, fontweight=800)
    axes[1].axis('off')

    plt.subplots_adjust(wspace=0.0)
    plt.show()
