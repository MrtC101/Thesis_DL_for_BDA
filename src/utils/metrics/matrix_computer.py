# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import random
import pandas as pd
import concurrent.futures
import torch
from tqdm import tqdm
from postprocessing.bbs.bounding_boxes import BoundingBox
from utils.metrics.common import Level
import matplotlib

from postprocessing.bbs.polygon_manager import get_buildings, get_instance_mask
matplotlib.use("TkAgg")

class MatrixComputer:

    #PIXEL LEVEL
    @staticmethod
    def patches_px_conf_mtrx(level, labels_set, gt_bld_mask, gt_dmg_mask, pd_bld_mask, pd_dmg_mask):
        """This method computes a confusión matrix for the hole batch of patches masks
        TODO:desc"""
        conf_mtrx_list = []
        for label in labels_set:
            if level == Level.PX_BLD:
                conf_mtrx = MatrixComputer.\
                    conf_mtrx_px_by_cls(gt_bld_mask, gt_bld_mask, pd_bld_mask, label)
            elif level == Level.PX_DMG:
                conf_mtrx = MatrixComputer.\
                    conf_mtrx_px_by_cls(gt_bld_mask, gt_dmg_mask, pd_dmg_mask, label)
            conf_mtrx_list.append(conf_mtrx)
        return pd.DataFrame(conf_mtrx_list)

    @staticmethod
    def tile_px_conf_mtrx(gt_bld_mask : torch.Tensor, gt_dmg_mask : torch.Tensor,
                          pd_dmg_mask : torch.Tensor, labels_set : int):
        """Computes the confusion matrix for the predicted mask.(Pixel Level)"""
        conf_mtrx_list = []
        for label in labels_set:
            conf = MatrixComputer.\
                conf_mtrx_px_by_cls(gt_bld_mask.unsqueeze(0) , gt_dmg_mask.unsqueeze(0),
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
        axis  = (0,1,2)
        tp = torch.sum(gt_mat & pd_mat, axis)
        fn = torch.sum(gt_mat & ~pd_mat, axis)
        fp = torch.sum(~gt_mat & pd_mat, axis)
        tn = torch.sum(~bld_mat & ~(pd_mask == label), axis)
        total_pixels = torch.prod(torch.tensor(gt_mask.shape))
        return {'class': label,
                'true_pos': int(tp),
                'false_pos': int(fp),
                'false_neg': int(fn),
                'true_neg': int(tn),
                'total': int(total_pixels)}

    # OBJECT LEVEL
    @staticmethod
    def patches_obj_conf_mtrx(level, labels_set, n_masks, gt_bld_mask, gt_dmg_mask,
                               pd_bld_mask, pd_dmg_mask):
        """ Because computing a confusion matrix for the hole batch of patches is computationally
        expensive, we are randomly sampling n. The result is the sum of all matrices."""
        patch_ids = [random.randint(0, len(pd_bld_mask)-1) for _ in range(n_masks)]
        # iterates over the shard
        conf_matrix = None
        for i in patch_ids:
            if level == Level.OBJ_BLD:
                label_conf_matirx = MatrixComputer.\
                    evaluate_polys(pd_bld_mask[i], gt_bld_mask[i], labels_set)
            elif level == Level.OBJ_DMG:
                label_conf_matirx = MatrixComputer.\
                    evaluate_polys(pd_dmg_mask[i], gt_dmg_mask[i], labels_set)
            if(conf_matrix is None):
                conf_matrix = pd.DataFrame(label_conf_matirx)
            label_df = pd.DataFrame(label_conf_matirx)
            conf_matrix["true_pos"] += label_df["true_pos"] 
            conf_matrix["false_pos"] += label_df["false_pos"] 
            conf_matrix["false_neg"] += label_df["false_neg"] 
            conf_matrix["true_neg"] += label_df["true_neg"]
            conf_matrix["total"] += label_df["total"]
        return conf_matrix

    @staticmethod
    def tile_obj_conf_mtrx(dmg_mask, pred_mask, labels_set):
        """Returns a confusión matrix for one image"""
        conf_mrtx = MatrixComputer.evaluate_polys(dmg_mask, pred_mask, labels_set)
        return pd.DataFrame(conf_mrtx)         
    
    @staticmethod
    def compute_IoU(gt_buildings, gt_label_matrix, pd_buildings, pd_label_matrix):
        def compute_iou(candidates):
            best_iou, best_intersection_area, best_union_area =  0.0, 0, 0
            for pid in candidates:
                if pid > 0:
                    _, pd_label = pd_buildings[pid-1]
                    if gt_label != pd_label:
                        intersection = 0
                        union = (gt_label_matrix == gid).sum() + (pd_label_matrix == pid).sum()
                    else:
                        intersection = ((gt_label_matrix == gid) & (pd_label_matrix == pid)).sum()
                        union = ((gt_label_matrix == gid) | (pd_label_matrix == pid)).sum()
                    
                    iou = intersection / union if union > 0 else 0

                    if iou > best_iou:
                        best_iou = iou
                        best_intersection_area = intersection
                        best_union_area = union

            return best_iou, best_intersection_area, best_union_area

        building_list = []
        # Calculate metrics for each ground truth polygon
        for gid, (gt_poly, gt_label)  in enumerate(gt_buildings):
            gt_bb = BoundingBox.create(gt_poly)
            gx1, gy1, gx2, gy2 = gt_bb.get_components()
            candidates = pd_label_matrix[gx1:gx2,gy1:gy2].unique()
            if(len(candidates) > 1000):
               print(len(candidates))
            best_iou = 0
            best_intersection_area = 0
            best_union_area = 0
            chunk_size = 1000

            parts = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(compute_iou, part) for part in parts]            
                for future in concurrent.futures.as_completed(futures):
                    iou, intersection, union = future.result()
                    if iou > best_iou:
                        best_iou = iou
                        best_intersection_area = intersection
                        best_union_area = union
                        
            building_list.append({
                'id': gid,
                'polygon': gt_poly,
                'label': gt_label,
                'iou': best_iou,
                'intersection_area': best_intersection_area,
                'union_area': best_union_area
            })
        
        return pd.DataFrame(building_list,columns=['id',  'polygon',
                'label', 'iou', 'intersection_area', 'union_area'])
    
    @staticmethod
    def evaluate_polys(target_mask, pred_mask, labels_set, iou_threshold: float = 0.7):
        """ This method calculates the corresponding confusion matrix at object level.
        1- Obtain for target mask and pred mask
            1- All buildings from mask are obtained
            2- All clusters of pixels are obtained from mask
            3- All buildings with class by matching buildings and clusters
        2- Computes the IoU of each building from target_mask. This is done by brute force
          testing the IoU of each building from target with each from pred mask. Each building
          matches with the one with greater IoU score.
        3- Computes the confusion matrix for the matched polygons and classes 
        
        Args:
            target_mask: target dmg_mask or bld_mask
            pred_mask: predicted dmg_mask or bld_mask
            labels_set: which classes should be evaluated
            iou_threshold: Intersection over union threshold above which a predicted
                polygon is considered true positive

        Returns:
            pd.DataFrame : A confusion matrix for each label. 
        """
        # code
        gt_buildings = get_buildings(target_mask, labels_set)            
        gt_label_matrix = get_instance_mask(gt_buildings)
        pd_buildings = get_buildings(pred_mask, labels_set)
        pd_label_matrix = get_instance_mask(pd_buildings)
        IoU_df = MatrixComputer.compute_IoU(gt_buildings, gt_label_matrix,
                                             pd_buildings, pd_label_matrix)
        results = []
        for c in labels_set:
            df = IoU_df[IoU_df["label"]==c]
            tp = (df["iou"] >= iou_threshold).sum()
            fp = len(pd_buildings) - tp
            fn = len(gt_buildings) - tp
            tn = 0  
            # Calculating true negatives is typically not done in object detection
            
            results.append({
                'class': c, 
                'true_pos': tp, 
                'false_pos': fp, 
                'false_neg': fn,
                'true_neg': tn,
                'total': len(gt_buildings)+len(gt_buildings)
            })
    
        return results
    
    @staticmethod
    def compute_bin_matrices_px(bin_pred_tensor : torch.Tensor,
                                bin_true_tensor : torch.Tensor) -> torch.Tensor:
        axis  = (1, 2)
        tp = torch.sum(bin_true_tensor & bin_pred_tensor, axis)
        fn = torch.sum(bin_true_tensor & ~bin_pred_tensor, axis)
        fp = torch.sum(~bin_true_tensor & bin_pred_tensor, axis)
        tn = torch.sum(~bin_true_tensor & ~bin_pred_tensor, axis)
        size = bin_pred_tensor.shape
        tot = torch.ones(size[0],dtype=torch.int32) * size[1] * size[2]
        return torch.stack([tp, fp, fn, tn, tot], axis=0)

    @staticmethod
    def compute_bin_matrices_obj(bin_pred_tensor : torch.Tensor,
                                 bin_true_tensor : torch.Tensor) -> torch.Tensor:
        tp_list = []
        fp_list = []
        fn_list = []
        tn_list = []
        tot_list = []
        for i in range(0,bin_pred_tensor.shape[0]):
            matrx_dict = MatrixComputer.evaluate_polys(bin_true_tensor[i],bin_pred_tensor[i],[1])[0]
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

        return torch.stack([tp,fp,fn,tn,tot],axis=0)