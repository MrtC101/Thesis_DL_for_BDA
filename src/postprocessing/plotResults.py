import os
from matplotlib import gridspec
import matplotlib.pyplot as plt
import pandas as pd
from utils.visualization.raster_label_visualizer import RasterLabelVisualizer

def save_results(dis_id, tile_id, pre_img, post_img, pred_mask,
                  curr_table : pd.DataFrame, label_map_json : str, save_path:str):
    """
        Crea un matplot lib donde se ven la imagen pre post predicción
        Junto con su tabla de conteo.
    """
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
    table = ax4.table(
            colWidths=[0.47]*len(curr_table.columns),
            cellText=curr_table.values,
            colLabels=curr_table.columns,
            rowLabels=curr_table.index,
            cellLoc='center',
            loc='center'
            )
    colors = ['darkgray','limegreen','orange','red','gray']
    table.set_fontsize(subtitle_size)
    # Aplicar estilos a la tabla
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Negritas para los encabezados
            cell.set_linewidth(2.0)
            cell.get_text().set_fontweight('bold')
        else:
            cell.set_facecolor(colors[i-1])
        cell.set_height(0.16)

    os.makedirs(save_path,exist_ok=True)
    files = os.path.join(save_path, f"{dis_id}-{tile_id}_predicted_dmg_mask.png")
    plt.tight_layout()
    plt.savefig(files, format='png', bbox_inches='tight', pad_inches=0.2)
    plt.close()
        