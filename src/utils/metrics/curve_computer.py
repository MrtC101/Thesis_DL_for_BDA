from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import trange
import torch

from models.trainable_model import TrainModel
from utils.datasets.train_dataset import TrainDataset
from utils.metrics.matrix_computer import MatrixComputer
from utils.metrics.metric_computer import MetricComputer
from postprocessing.plots.plot_results import plot_pr_curves, plot_roc_curves



def pixel_metric_curves(loader: DataLoader, model: TrainModel, device : torch.device, metric_dir: str,
                        k: int = 200) -> None:
    """ Computes the metrics necessary for plotting the ROC curve and PR curve for the dataset. 
    """
    n_class = 5
    model.eval()
    # columns=["tp","fp","fn","tn"]
    patch_conf_mtrx_dict = defaultdict(lambda: torch.zeros(size=(n_class, 4)))

    for dis_id, tile_id, patch_id, patch in tqdm(loader):
        x_pre = patch['pre_img'].to(device=device)
        x_post = patch['post_img'].to(device=device)
        y_cls = patch['dmg_mask'].to(device=device)

        masks = [(y_cls == lab_i) for lab_i in range(5)]
        bin_true_tensor = torch.stack(masks, dim=0)
        
        logit_masks = model(x_pre, x_post)[2]

        for th in torch.linspace(0,1,k):
            bin_pred_tensor = model.softmax(logit_masks) >= th
            bin_pred_tensor = bin_pred_tensor.permute(1,0,2,3)
            conf_matrices = MatrixComputer.compute_bin_matrices_px(bin_pred_tensor, bin_true_tensor)
            key = round(float(th), 5)
            patch_conf_mtrx_dict[key] += conf_matrices.to(device='cpu')
    
    roc_curve: tuple = MetricComputer.compute_ROC(patch_conf_mtrx_dict)
    plot_roc_curves("pixel", roc_curve, metric_dir)
    pr_curve: tuple = MetricComputer.compute_PR(patch_conf_mtrx_dict)
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
