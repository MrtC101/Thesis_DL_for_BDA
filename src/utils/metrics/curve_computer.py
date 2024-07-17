from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import trange
import torch

from models.trainable_model import TrainModel
from utils.datasets.train_dataset import TrainDataset
from utils.metrics.matrix_computer import MatrixComputer
from utils.metrics.metric_computer import MetricComputer
from postprocessing.plots.plot_results import plot_pr_curves, plot_roc_curves

def pixel_metric_curves(loader : DataLoader, model : TrainModel, metric_dir : str,
                        k : int = 10) -> None:
    """ Computes the metrics necessary for plotting the ROC curve and PR curve for the dataset. 
    """
    dataset : TrainDataset = loader.dataset
    n_class = 5
    model.eval()
    #columns=["tp","fp","fn","tn"]
    patch_conf_mtrx_dict = defaultdict(lambda : torch.zeros(size=(n_class,4),dtype=torch.int32))
    for i in trange(0,int(len(dataset)/16)-1):
        dis_id, tile_id, patch_dict = dataset.get_by_id(i)
        for patch_id, patch in patch_dict.items():
            x_pre = patch['pre_img']
            x_post = patch['post_img']
            y_cls = patch['dmg_mask']
            bin_true_tensor = model.make_binary(y_cls, [0,1,2,3,4])
            logit_masks = model(x_pre.unsqueeze(0), x_post.unsqueeze(0))
            
            for th in torch.arange(0,1+(1/k),1/k):
                bin_pred_tensor = model.compute_binary(logit_masks[2], th)
                bin_pred_tensor = bin_pred_tensor.squeeze(0)
                conf_matrices = MatrixComputer.\
                    compute_bin_matrices_px(bin_pred_tensor, bin_true_tensor)
                key = round(float(th),1)
                patch_conf_mtrx_dict[key] += conf_matrices
    roc_curve : tuple = MetricComputer.compute_ROC(patch_conf_mtrx_dict)
    plot_roc_curves("pixel", roc_curve, metric_dir)  
    pr_curve : tuple = MetricComputer.compute_PR(patch_conf_mtrx_dict)
    plot_pr_curves("pixel", pr_curve, metric_dir)

def merge_patches(patches_dict: dict) -> dict:
    tile_dict = {}
    for th, patches in patches_dict.items():
        rows_pd = []
        rows_gt = []
        row_pd = []
        row_gt = []
        for i, (pd_patch, gt_patch) in enumerate(patches):
            row_pd.append(pd_patch)
            row_gt.append(gt_patch)
            if i % 4 == 3:  # Cada 4 parches en una fila
                line_pd = torch.cat(row_pd, dim=2)
                line_gt = torch.cat(row_gt, dim=2)
                row_pd = []
                row_gt = []
                rows_pd.append(line_pd)
                rows_gt.append(line_gt)
        
        pd_tile = torch.cat(rows_pd, dim=1)
        gt_tile = torch.cat(rows_gt, dim=1)
        tile_dict[th] = {"pd_bin": pd_tile, "gt_bin": gt_tile} 
    return tile_dict


def object_metric_curves(loader : DataLoader, metric_dir : str,
                        k : int = 5) -> None:
    """
    Resumen de Métricas y Evaluaciones
    Métrica	Qué Mide	Fórmula o Método
    - IoU (Intersección sobre Unión)	Calidad del ajuste de las cajas delimitadoras	
    - mAP (Precisión Media)	Media de precisión en diferentes umbrales de IoU	
    - Curva PR (Precisión-Recuerdo)	Relación entre precisión y recall a diferentes umbrales	
    """
    for th in torch.arange(0,1+(1/k),1/k):
        key = round(float(th),1)
        bin_pred_tensor = tile_dict[key]["pd_bin"]
        bin_true_tensor = tile_dict[key]["gt_bin"]
        #INEFICIENTE
        #conf_matrices = MatrixComputer.\
        #    compute_bin_matrices_obj(bin_pred_tensor, bin_true_tensor)
        #conf_matrices = torch.transpose(conf_matrices,0,1)
        #obj_conf_mtrx_dict[key] += conf_matrices 

