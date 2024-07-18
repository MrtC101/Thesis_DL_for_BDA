# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import random
import pandas as pd
import concurrent.futures
import torch
from tqdm import tqdm
from postprocessing.bbs.bounding_boxes import BoundingBox
from utils.loggers.console_logger import LoggerSingleton
from utils.metrics.common import Level
import matplotlib

from postprocessing.bbs.polygon_manager import get_buildings, get_instance_mask
matplotlib.use("TkAgg")


class MatrixComputer:

    # PIXEL LEVEL
    @staticmethod
    def patches_px_conf_mtrx(level, labels_set, gt_bld_mask, gt_dmg_mask, pd_bld_mask, pd_dmg_mask):
        """This method computes a confusión matrix for the hole batch of patches masks
        TODO:desc"""
        conf_mtrx_list = []
        for label in labels_set:
            if level == Level.PX_BLD:
                conf_mtrx = MatrixComputer.\
                    conf_mtrx_px_by_cls(
                        gt_bld_mask, gt_bld_mask, pd_bld_mask, label)
            elif level == Level.PX_DMG:
                conf_mtrx = MatrixComputer.\
                    conf_mtrx_px_by_cls(
                        gt_bld_mask, gt_dmg_mask, pd_dmg_mask, label)
            conf_mtrx_list.append(conf_mtrx)
        return pd.DataFrame(conf_mtrx_list)

    @staticmethod
    def tile_px_conf_mtrx(gt_bld_mask: torch.Tensor, gt_dmg_mask: torch.Tensor,
                          pd_dmg_mask: torch.Tensor, labels_set: int):
        """Computes the confusion matrix for the predicted mask.(Pixel Level)"""
        conf_mtrx_list = []
        for label in labels_set:
            conf = MatrixComputer.\
                conf_mtrx_px_by_cls(gt_bld_mask.unsqueeze(0), gt_dmg_mask.unsqueeze(0),
                                    pd_dmg_mask.unsqueeze(0), label)
            conf_mtrx_list.append(conf)
        return pd.DataFrame(conf_mtrx_list)

    @staticmethod
    def conf_mtrx_px_by_cls(gt_bld_mask, gt_mask, pd_mask, label):
        """Computes a confusión matrix for a batch of images with a one vs all approach.

            Args:
                gt_bld_mask : Ground truth building mask
                gt_mask : Ground truth mask for buildings or for damages
                pd_mask : Predicted mask for buildings or for damages

        """
        bld_mat = (gt_bld_mask == 1)
        gt_mat = (gt_mask == label) & bld_mat
        pd_mat = (pd_mask == label) & bld_mat
        axis = (0, 1, 2)
        tp = torch.sum(gt_mat & pd_mat, axis)
        fn = torch.sum(gt_mat & ~pd_mat, axis)
        fp = torch.sum(~gt_mat & pd_mat, axis)
        tn = torch.sum(~bld_mat & ~(pd_mask == label), axis)
        total_pixels = tp + fn + fp + tn
        return {'class': label,
                'true_pos': int(tp),
                'false_pos': int(fp),
                'false_neg': int(fn),
                'true_neg': int(tn),
                'total': int(total_pixels)}

    @staticmethod
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

    # OBJECT LEVEL

    @staticmethod
    def match_polygons(gt_buildings, gt_label_matrix, pd_label_matrix, th):
        """
            Match each ground truth polygon with the max IoU predicted polygon,
            without repetition and only if its IoU is higher than the given
            threshold.
        """
        pd_len = pd_label_matrix.unique().numel() - 1
        relation = torch.zeros(
            size=(len(gt_buildings), pd_len), dtype=torch.float)

        # Fill relation matrix
        for gid, (gt_poly, _) in enumerate(tqdm(gt_buildings)):
            gt_bb = BoundingBox.create(gt_poly)
            x1, y1, x2, y2 = gt_bb.get_components()
            candidates = pd_label_matrix[y1:y2, x1:x2].unique()
            best_pid, best_iou = -1, 0.0
            for pid in candidates:
                if pid > 0:
                    intersection = ((gt_label_matrix == gid + 1)
                                    & (pd_label_matrix == pid)).sum()
                    union = ((gt_label_matrix == gid + 1) |
                             (pd_label_matrix == pid)).sum()
                    iou = intersection / union if union > 0 else 0.0
                    if iou > best_iou and iou >= th:
                        best_pid = int(pid)
                        best_iou = iou
            pid, iou = best_pid, best_iou
            if pid > -1:
                relation[gid][pid-1] = iou

        building_list = []
        if len(gt_buildings) > 0:
            for pid in range(pd_len):
                gid = torch.argmax(relation[:, pid])
                iou = relation[gid][pid]
                relation[:, pid] = 0.0
                relation[gid][pid] = iou

        for gid in range(len(gt_buildings)):
            if pd_len > 0:
                pid = torch.argmax(relation[gid])
                iou = relation[gid][pid]
            else:
                iou = 0.0
            building_list.append({
                'gid': int(gid),
                'pid': int(pid) if iou >= th else -1,
                'iou': int(iou)
            })

        return pd.DataFrame(building_list)

    @staticmethod
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
        - IoU_df: DataFrame con los valores de IoU entre los polígonos coincidentes.
        """
        log = LoggerSingleton()

        gt_buildings = get_buildings(target_mask, labels_set)
        gt_label_matrix = get_instance_mask(gt_buildings)
        pd_buildings = get_buildings(pred_mask, labels_set)
        pd_label_matrix = get_instance_mask(pd_buildings)
        log.info('\n' +
                 f"{len(gt_buildings)} buildings found in ground truth mask" +
                 '\n' +
                 f"{len(pd_buildings)} buildings found in predicted mask")
        log.info("Matching labels for each building.")
        IoU_df = MatrixComputer.match_polygons(gt_buildings, gt_label_matrix,
                                               pd_label_matrix, iou_threshold)

        gt_labels = pd.Series([l for _, l in gt_buildings])
        pd_labels = pd.Series([l for _, l in pd_buildings])
        return gt_labels, pd_labels, IoU_df

    @staticmethod
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
            gt_labels, pd_labels, IoU_df = MatrixComputer.\
                evaluate_polygon_iou(target_mask, pred_mask, labels_set)
            cls_conf_mtrx = MatrixComputer.\
                conf_mtrx_obj_by_cls(gt_labels, pd_labels, IoU_df, labels_set)

            if conf_matrix is None:
                conf_matrix = cls_conf_mtrx.copy()
            else:
                for key in conf_matrix.keys():
                    conf_matrix[key] += cls_conf_mtrx[key]
        return conf_matrix

    @staticmethod
    def conf_mtrx_obj_by_cls(gt_labels, pd_labels, IoU_df, labels_set):
        conf_mrtx = []
        for c in labels_set:
            gt_s = gt_labels[gt_labels == c]
            pd_s = pd_labels[pd_labels == c]
            relation = IoU_df.loc[gt_s.index]
            tp = (relation["pid"] > -1).sum()
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
                'total': len(gt_s)+len(pd_s)
            })

        return pd.DataFrame(conf_mrtx)

    @staticmethod
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

        for gt in range(l_size):
            l_g = labels_set[gt]
            curr_gt = gt_labels[gt_labels == l_g]
            rel_df = IoU_df.loc[curr_gt.index]
            matched_ids = rel_df[rel_df["pid"] > -1]["pid"]
            curr_prd = pd_labels.loc[matched_ids]

            for prd in range(l_size):
                l_p = labels_set[prd]
                mat[gt][prd] = (curr_prd == l_p).sum()

            mat[gt][l_size] = (rel_df["pid"] == -1).sum()  # Undetected
            mat[gt][size - 1] = len(curr_gt)

            ghost = pd_labels.loc[~pd_labels.index.isin(
                matched_ids)][pd_labels == l_g]
            mat[l_size][gt] = len(ghost)  # Ghost predictions

        for prd in range(l_size):
            mat[size - 1][prd] = torch.sum(mat[:, prd])  # Total predicted

        # Ghost predictions
        mat[l_size][size - 1] = torch.sum(mat[l_size])
        # Unpredicted
        mat[size - 1][l_size] = torch.sum(mat[:, l_size])
        # Totales por fila y columna
        mat[size - 1][size -
                      1] = torch.sum(mat[:size, size - 1]) + torch.sum(mat[size - 1, :size])

        # Convertir a DataFrame de pandas
        mat_df = pd.DataFrame(mat.numpy(), columns=labels_set + ['Undetected', 'Total'],
                              index=labels_set + ['Ghost', 'Total'])
        return mat_df

    @staticmethod
    def tile_obj_conf_matrices(dmg_mask, pred_mask, labels_set, iou_threshold=0.5):
        """Returns a confusión matrix for one image"""
        gt_labels, pd_labels, IoU_df = MatrixComputer.\
            evaluate_polygon_iou(dmg_mask, pred_mask,
                                 labels_set, iou_threshold)
        conf_mrtx = MatrixComputer.\
            conf_mtrx_obj_by_cls(gt_labels, pd_labels, IoU_df, labels_set)
        multi_conf_mtrx = MatrixComputer.\
            obj_multiclass_conf_mtrx(gt_labels, pd_labels, IoU_df, labels_set)
        return conf_mrtx, multi_conf_mtrx

    # BINARY MASKS
    @staticmethod
    def compute_bin_matrices_px(bin_pred_tensor: torch.Tensor,
                                bin_true_tensor: torch.Tensor) -> torch.Tensor:
        axis = (1, 2)
        tp = torch.sum(bin_true_tensor & bin_pred_tensor, axis)
        fn = torch.sum(bin_true_tensor & ~bin_pred_tensor, axis)
        fp = torch.sum(~bin_true_tensor & bin_pred_tensor, axis)
        tn = torch.sum(~bin_true_tensor & ~bin_pred_tensor, axis)
        return torch.stack([tp, fp, fn, tn], axis=0).transpose(0, 1)

    @staticmethod
    def compute_bin_matrices_obj(bin_pred_tensor: torch.Tensor,
                                 bin_true_tensor: torch.Tensor) -> torch.Tensor:
        tp_list = []
        fp_list = []
        fn_list = []
        tn_list = []
        tot_list = []
        for i in range(0, bin_pred_tensor.shape[0]):
            matrx_dict = MatrixComputer.evaluate_polys(
                bin_true_tensor[i], bin_pred_tensor[i], [1])[0]
            tp_list.append(matrx_dict["true_pos"])
            fp_list.append(matrx_dict["false_pos"])
            fn_list.append(matrx_dict["false_neg"])
            tn_list.append(matrx_dict["true_neg"])
            tot_list.append(matrx_dict["total"])

        tp = torch.tensor(tp_list, dtype=torch.int32)
        fp = torch.tensor(fp_list, dtype=torch.int32)
        fn = torch.tensor(fn_list, dtype=torch.int32)
        tn = torch.tensor(tn_list, dtype=torch.int32)
        tot = torch.tensor(tot_list, dtype=torch.int32)

        return torch.stack([tp, fp, fn, tn, tot], axis=0)
