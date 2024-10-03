from collections import defaultdict
from sklearn.metrics import auc
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from models.trainable_model import TrainModel
from postprocessing.plots.plot_results import plot_pr_curves, plot_roc_curves

# BINARY MASKS


def compute_bin_matrices_px(bin_pred_tensor: torch.Tensor,
                            bin_true_tensor: torch.Tensor) -> torch.Tensor:
    axis = (-2, -1)
    tp = torch.sum(bin_true_tensor & bin_pred_tensor, axis).sum(axis=1)
    fn = torch.sum(bin_true_tensor & ~bin_pred_tensor, axis).sum(axis=1)
    fp = torch.sum(~bin_true_tensor & bin_pred_tensor, axis).sum(axis=1)
    tn = torch.sum(~bin_true_tensor & ~bin_pred_tensor, axis).sum(axis=1)
    batch = torch.stack([tp, fp, fn, tn], axis=1)
    return batch


def pixel_metric_curves(loader: DataLoader, model: TrainModel, device: torch.device,
                        metric_dir: str, k: int = 200) -> None:
    """ Computes the metrics necessary for plotting the ROC curve and PR curve for the dataset."""
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

        for th in torch.linspace(0, 1, k):
            bin_pred_tensor = model.softmax(logit_masks) >= th
            bin_pred_tensor = bin_pred_tensor.permute(1, 0, 2, 3)
            conf_matrices = compute_bin_matrices_px(bin_pred_tensor, bin_true_tensor)
            key = round(float(th), 5)
            patch_conf_mtrx_dict[key] += conf_matrices.to(device='cpu')

    roc_curve: tuple = compute_ROC(patch_conf_mtrx_dict)
    plot_roc_curves("pixel", roc_curve, metric_dir)
    pr_curve: tuple = compute_PR(patch_conf_mtrx_dict)
    plot_pr_curves("pixel", pr_curve, metric_dir)


def compute_ROC(conf_dict: dict[torch.Tensor]):
    # Almacenar las curvas ROC para cada clase y umbral
    curves = defaultdict(lambda: defaultdict(list))

    for th, conf_tensor in conf_dict.items():
        for i_label in range(conf_tensor.shape[0]):
            tp, fp, fn, tn = map(float, conf_tensor[i_label])
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (fp + tn) if (fp + tn) > 0 else 0
            curves[i_label][th] = (1 - specificity, sensitivity)  # FPR, TPR

    # Calcular el AUC para cada curva ROC
    roc_curves = {}
    for label, th_curves in curves.items():
        x, y = zip(*sorted(th_curves.values()))  # Ordenar por FPR
        auc_value = auc(x, y)
        roc_curves[label] = (x, y, auc_value)
    return roc_curves


def _compute_ap(precision, recall):
    ap = 0.0
    for i in range(len(recall)):
        if i == 0:
            ap += precision[i] * recall[i]
        else:
            ap += precision[i] * (recall[i] - recall[i-1])
    return ap


def compute_PR(conf_dict: dict[torch.Tensor]):
    pr_curves = {}
    conf = torch.stack([t for t in conf_dict.values()], dim=0)
    conf_list = torch.unbind(conf, dim=1)
    for i_label in range(conf.shape[1]):
        tp, fp, fn, tn = torch.unbind(conf_list[i_label], dim=1)
        precision = tp / (tp + fp)
        precision = torch.where(torch.isnan(
            precision), torch.tensor(1.0), precision)
        recall = tp / (tp + fn)
        recall = torch.where(torch.isnan(
            recall), torch.tensor(0.0), recall)
        base_precision = tp.max() / (tp.max() + tn.max())
        pr_curves[i_label] = (recall, precision, base_precision, _compute_ap(precision, recall))
    return pr_curves
