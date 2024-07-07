# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import random
import pandas as pd
import torch
from utils.metrics.common import Level
import matplotlib

from postprocessing.polygon_manager import get_buildings
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
    def evaluate_polys(target_mask, pred_mask, labels_set, iou_threshold: float = 0.5):
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

        def compute_IoU(gt_buildings,pd_buildings):
            building_list = []
            # Calculate metrics for each ground truth polygon
            for id, gt_dict  in enumerate(gt_buildings):
                gt_poly = gt_dict["bld"]
                gt_label = gt_dict["label"]
                best_iou = 0
                best_intersection_area = 0
                best_union_area = 0

                for pd_dict in pd_buildings:
                    pd_poly = pd_dict["bld"]
                    pd_label = pd_dict["label"]
                    if gt_label != pd_label:
                        intersection = 0
                        union = gt_poly.area + pd_poly.area
                    else:
                        intersection = gt_poly.intersection(pd_poly).area
                        union = gt_poly.union(pd_poly).area
                    iou = intersection / union if union > 0 else 0

                    if iou > best_iou:
                        best_iou = iou
                        best_intersection_area = intersection
                        best_union_area = union

                building_list.append({
                    'id': id,
                    'polygon':gt_poly,
                    'label': gt_label,
                    'iou': best_iou,
                    'intersection_area': best_intersection_area,
                    'union_area': best_union_area
                })
            
            return pd.DataFrame(building_list,columns=['id',  'polygon',
                    'label', 'iou', 'intersection_area', 'union_area'])
        
        # code
        gt_buildings = get_buildings(target_mask, labels_set)            
        pd_buildings = get_buildings(pred_mask, labels_set)
        IoU_df = compute_IoU(gt_buildings, pd_buildings)
        results = []
        for c in labels_set:
            df = IoU_df[IoU_df["label"]==c]
            total_buildings = len(df)
            tp = (df["iou"] >= iou_threshold).sum()
            fn = (df["iou"] < iou_threshold).sum()
            fp = total_buildings - tp
            tn = 0  
            # Calculating true negatives is typically not done in object detection
            
            results.append({
                'class': c, 
                'true_pos': tp, 
                'false_pos': fp, 
                'false_neg': fn,
                'true_neg': tn,
                'total': total_buildings
            })
    
        return results