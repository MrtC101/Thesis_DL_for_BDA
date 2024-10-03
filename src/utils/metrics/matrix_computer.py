# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import random
import pandas as pd
import torch
from tqdm import tqdm
from postprocessing.bbs.bounding_boxes import BoundingBox
from utils.loggers.console_logger import LoggerSingleton

from postprocessing.bbs.polygon_manager import get_buildings, get_instance_mask
from utils.metrics.metric_manager import Level
from scipy.optimize import linear_sum_assignment

# matplotlib.use("TkAgg")


# PIXEL LEVEL


def patches_px_conf_mtrx(level, labels_set, gt_bld_mask, gt_dmg_mask, pd_bld_mask, pd_dmg_mask):
    """This method computes a confusión matrix for the hole batch of patches masks """
    conf_mtrx_list = []
    for label in labels_set:
        if level == Level.PX_BLD:
            conf_mtrx = conf_mtrx_px_by_cls(gt_bld_mask, pd_bld_mask, label)
        elif level == Level.PX_DMG:
            conf_mtrx = conf_mtrx_px_by_cls(gt_dmg_mask, pd_dmg_mask, label)
        conf_mtrx_list.append(conf_mtrx)
    return pd.DataFrame(conf_mtrx_list)


def conf_mtrx_px_by_cls(gt_mask: torch.Tensor, pd_mask: torch.Tensor, label: list) -> dict:
    """Computes a confusión matrix for a batch of images with a one vs all approach.

        Args:
            gt_mask : Ground truth mask for buildings or for damages
            pd_mask : Predicted mask for buildings or for damages

    """
    gt_mat = (gt_mask == label)
    pd_mat = (pd_mask == label)
    axis = (0, 1, 2)
    tp = torch.sum(gt_mat & pd_mat, axis)
    fn = torch.sum(gt_mat & ~pd_mat, axis)
    fp = torch.sum(~gt_mat & pd_mat, axis)
    tn = torch.sum(~gt_mat & ~pd_mat, axis)
    total_pixels = tp + fn + fp + tn
    return {'class': label,
            'true_pos': int(tp),
            'false_pos': int(fp),
            'false_neg': int(fn),
            'true_neg': int(tn),
            'total': int(total_pixels)}


def px_multiclass_conf_mtrx(dmg_mask: torch.Tensor, pred_mask: torch.Tensor,
                            dmg_labels: list[int]) -> pd.DataFrame:
    l_size = len(dmg_labels)
    mat = torch.zeros(size=(l_size + 1, l_size + 1), dtype=torch.int32)

    # Llenar la matriz de confusión
    for gt in range(l_size):
        for prd in range(l_size):
            bin_true = dmg_mask == dmg_labels[gt]
            bin_pred = pred_mask == dmg_labels[prd]
            mat[gt][prd] = torch.sum(bin_true & bin_pred).item()

    # Sumar los totales por fila y columna
    for gt in range(l_size):
        mat[gt][l_size] = torch.sum(mat[gt, :l_size]).item()
    for prd in range(l_size):
        mat[l_size][prd] = torch.sum(mat[:l_size, prd]).item()

    mat[l_size][l_size] = torch.sum(mat[:l_size, :l_size]).item()
    # Convertir a DataFrame de pandas
    mat_df = pd.DataFrame(mat.numpy(), columns=dmg_labels + ['Total'],
                          index=dmg_labels + ['Total'])
    return mat_df


def tile_px_conf_mtrx(gt_dmg_mask: torch.Tensor, pd_dmg_mask: torch.Tensor, labels_set: int):
    """Computes the confusion matrix for the predicted mask.(Pixel Level)"""
    conf_mtrx_list = []
    for label in labels_set:
        conf = conf_mtrx_px_by_cls(gt_dmg_mask.unsqueeze(0), pd_dmg_mask.unsqueeze(0), label)
        conf_mtrx_list.append(conf)
    return pd.DataFrame(conf_mtrx_list)


# OBJECT LEVEL


def match_polygons(gt_buildings: list, gt_label_matrix: torch.Tensor,
                   pd_buildings: list, pd_label_matrix: torch.Tensor, th: float):
    """
        Match each ground truth polygon with the max IoU predicted polygon,
        without repetition and only if its IoU is higher than the given
        threshold.
    """
    building_list = []
    if len(gt_buildings) > 0 and len(pd_buildings) > 0:
        relation = torch.zeros(size=(len(gt_buildings), len(pd_buildings)), dtype=torch.float)

        # Fill relation matrix
        for gid, (gt_poly, _) in enumerate(tqdm(gt_buildings)):
            gt_bb = BoundingBox.create(gt_poly)
            x1, y1, x2, y2 = gt_bb.get_components()
            candidates: torch.Tensor = pd_label_matrix[y1:y2, x1:x2].unique()
            candidates: list = candidates.tolist()
            try:
                candidates.remove(-1)
            except ValueError:
                pass

            for pid in candidates:
                pd_poly, _ = pd_buildings[pid]
                intersection = gt_poly.intersection(pd_poly).area
                union = gt_poly.union(pd_poly).area
                relation[gid][pid] = intersection / union if union > 0 else 0.0

        # Asignación hungara para transformar la relación en binaría.
        cost_matrix = 1 - relation
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        for gid, pid in zip(row_indices, col_indices):
            iou = relation[gid][pid]
            if iou > th:
                building_list.append({
                    'gid': gid,
                    'pid': pid,
                    'iou': iou,
                    'glabel': gt_buildings[gid][1],
                    'plabel': pd_buildings[pid][1],
                })

    return pd.DataFrame(building_list)


def evaluate_polygon_iou(target_mask, pred_mask, labels_set, iou_threshold: float = 0.5):
    """
    Evalúa la coincidencia entre polígonos en las máscaras ground truth y predichas
    utilizando un umbral de IoU.

    Parámetros:
    - target_mask: La máscara ground truth.
    - pred_mask: La máscara predicha.
    - labels_set: El conjunto de etiquetas de los polígonos.
    - iou_threshold: El umbral de IoU para considerar una coincidencia (por defecto 0.5).

    Retorna:
    - gt_labels: Series con las etiquetas de los polígonos ground truth.
    - pd_labels: Series con las etiquetas de los polígonos predichos.
    - IoU_df: DataFrame Unicamente con aquellos polígonos gt que coincidieron con alguno pd,
    y la label que se predijo.
    """
    log = LoggerSingleton()

    gt_buildings = get_buildings(target_mask)
    gt_label_matrix = get_instance_mask(gt_buildings)
    pd_buildings = get_buildings(pred_mask)
    pd_label_matrix = get_instance_mask(pd_buildings)
    log.info('\n' +
             f"{len(gt_buildings)} buildings found in ground truth mask" +
             '\n' +
             f"{len(pd_buildings)} buildings found in predicted mask")
    log.info("Matching labels for each building.")
    IoU_df = match_polygons(gt_buildings, gt_label_matrix,
                            pd_buildings, pd_label_matrix, iou_threshold)
    gt_labels = pd.Series([l for _, l in gt_buildings])
    pd_labels = pd.Series([l for _, l in pd_buildings])
    return IoU_df, gt_labels, pd_labels


def patches_obj_conf_mtrx(level, labels_set, n_masks, gt_bld_mask,
                          gt_dmg_mask, pd_bld_mask, pd_dmg_mask):
    """ Because computing a confusion matrix for the hole batch of patches
        is computationally expensive, we are randomly sampling n.
    The result is the sum of all matrices."""
    patch_ids = [random.randint(0, len(pd_bld_mask)-1)
                 for _ in range(n_masks)]
    # iterates over the shard
    conf_matrix = None
    for i in patch_ids:
        target_mask = gt_bld_mask[i] if level == Level.OBJ_BLD else gt_dmg_mask[i]
        pred_mask = pd_bld_mask[i] if level == Level.OBJ_BLD else pd_dmg_mask[i]
        # level == Level.OBJ_DMG
        gt_labels, pd_labels, IoU_df = \
            evaluate_polygon_iou(target_mask, pred_mask, labels_set)
        cls_conf_mtrx = \
            conf_mtrx_obj_by_cls(gt_labels, pd_labels, IoU_df, labels_set)

        if conf_matrix is None:
            conf_matrix = cls_conf_mtrx.copy()
        else:
            for key in conf_matrix.keys():
                conf_matrix[key] += cls_conf_mtrx[key]
    return conf_matrix


def conf_mtrx_obj_by_cls(gt_labels, pd_labels, IoU_df, labels_set):
    """Builds a confusion matrix for each class at object level."""
    conf_mrtx = []
    for c in labels_set:
        if (len(IoU_df) > 0):
            curr_df = IoU_df[IoU_df["glabel"] == c]
            tp = (curr_df["glabel"] == curr_df["plabel"]).sum()
        else:
            tp = 0
        gt_s = gt_labels[gt_labels == c]
        pd_s = pd_labels[pd_labels == c]
        fn = len(gt_s) - tp
        fp = len(pd_s) - tp
        tn = 0
        # Calculating true negatives is typically not done in object detection

        conf_mrtx.append({
            'class': c,
            'true_pos': tp,
            'false_pos': fp,
            'false_neg': fn,
            'true_neg': tn,
            'true_total': len(gt_s),
            'pred_total': len(pd_s)
        })

    return pd.DataFrame(conf_mrtx)


def obj_multiclass_conf_mtrx(gt_labels, pd_labels, IoU_df, labels_set) -> pd.DataFrame:
    """
    Calcula la matriz de confusión multiclass a nivel de objeto.

    Parámetros:
    - gt_labels: Series con las etiquetas de los polígonos ground truth.
    - pd_labels: Series con las etiquetas de los polígonos predichos.
    - IoU_df: DataFrame con los valores de IoU entre los polígonos coincidentes.
    - labels_set: Conjunto de etiquetas de los polígonos.

    Retorna:
    - mat_df: DataFrame con la matriz de confusión.
    """
    l_size = len(labels_set)
    size = l_size + 2  # +2 para 'Undetected' y 'Total'
    mat = torch.zeros(size=(size, size), dtype=torch.int32)

    if (len(IoU_df) > 0):
        for gt_lab in labels_set:
            for pd_lab in labels_set:
                gt_labxpd_lab = (IoU_df["glabel"] == gt_lab) & (IoU_df["plabel"] == pd_lab)
                mat[gt_lab-1][pd_lab-1] = gt_labxpd_lab.sum()

    # Adding Undetected column
    for gt_lab in labels_set:
        mat[gt_lab - 1, size - 2] = (gt_labels == gt_lab).sum()
        mat[gt_lab - 1, size - 1] = \
            mat[gt_lab - 1, size - 2] - mat[gt_lab - 1, 0:len(labels_set)].sum()

    # Adding Ghost row
    for pd_lab in labels_set:
        mat[size - 2, pd_lab - 1] = (pd_labels == pd_lab).sum()
        mat[size - 1, pd_lab - 1] = \
            mat[size - 2, pd_lab - 1] - mat[0:len(labels_set), pd_lab - 1].sum()

    # Total predicted
    mat[size - 2, size - 2] = mat[:, size - 2].sum()
    mat[size - 2, size - 1] = mat[:, size - 1].sum()
    mat[size - 1, size - 2] = mat[size - 1, :].sum()

    # Convertir a DataFrame de pandas
    mat_df = pd.DataFrame(mat.numpy(),
                          columns=labels_set + ['Total', 'Undetected'],
                          index=labels_set + ['Total', 'Ghost'])
    return mat_df


def tile_obj_conf_matrices(dmg_mask, pred_mask, labels_set, iou_threshold=0.5):
    """Returns a confusión matrix for one image"""
    IoU_df, gt_labels, pd_labels = evaluate_polygon_iou(
        dmg_mask, pred_mask, labels_set, iou_threshold)
    conf_mrtx = conf_mtrx_obj_by_cls(gt_labels, pd_labels, IoU_df, labels_set)
    multi_conf_mtrx = obj_multiclass_conf_mtrx(gt_labels, pd_labels, IoU_df, labels_set)
    return conf_mrtx, multi_conf_mtrx
