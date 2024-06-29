# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
from collections import OrderedDict, defaultdict
import random

import pandas as pd
import rasterio
import shapely
import rasterio.features
import numpy as np
import shapely.geometry
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import shape
from shapely.geometry import Polygon
from utils.metrics.common import Level
import matplotlib
matplotlib.use("TkAgg")

class MatrixComputer:

    #PIXEL LEVEL
    @staticmethod
    def patches_px_conf_mtrx(level, labels_set, pred_bld_mask, pred_dmg_mask, y_bld_mask, y_dmg_mask):
        """This method computes a confusión matrix for the hole batch of patches masks"""
        conf_mtrx_list = []
        for cls in labels_set:
            if level == Level.PX_BLD:
                conf_mtrx = MatrixComputer.conf_mtrx_px_by_cls(pred_bld_mask, pred_bld_mask,
                                                                  y_bld_mask.detach().clone(), cls)
            elif level == Level.PX_DMG:
                conf_mtrx = MatrixComputer.conf_mtrx_px_by_cls(pred_dmg_mask, y_dmg_mask,
                                                                  y_bld_mask.detach().clone(), cls)
            conf_mtrx_list.append(conf_mtrx)
        return pd.DataFrame(conf_mtrx_list)

    @staticmethod
    def tile_px_conf_mtrx(bld_mask , dmg_mask, pred_mask, dmg_labels):
        """Computes the confusion matrix for the predicted mask.(Pixel Level)"""
        conf_mtrx_list = []
        for cls in dmg_labels:
            conf = MatrixComputer.conf_mtrx_px_by_cls(bld_mask , dmg_mask, pred_mask, cls)
            conf_mtrx_list.append(conf)
        return pd.DataFrame(conf_mtrx_list)

    @staticmethod
    def conf_mtrx_px_by_cls(bld_mask, true_mask, pred_mask, cls):
        """Computes a confusión matrix for given class using only building pixels from bld mask"""
        # ONLY TEST PIXELS THAT ARE BUILDINGS
        cls_mat = pred_mask[(true_mask==cls) & (bld_mask == 1)]
        tp_mat = np.where(cls_mat == cls,1,0)
        fn_mat = np.where(cls_mat != cls,1,0)
        pred_mat = pred_mask[(true_mask!=cls) & (bld_mask == 1)]
        fp_mat = np.where(pred_mat != cls,1,0)
        tn_mat = np.where(pred_mat != cls,1,0)
        
        tp = tp_mat.sum()
        fn = fn_mat.sum()
        fp = fp_mat.sum()
        tn = tn_mat.sum()

        # compute total pixels
        total_pixels = tp+tn+fp+fn
        # The tp value for the ground class (0) it is supposed to be 0 if
        # all pixels from the binary building mask match with all
        # not 0 labeled pixels from the damage mask. This usually is not the case.
        return {'class': cls,
                'true_pos': tp,
                'true_neg': tn,
                'false_pos': fp,
                'false_neg': fn,
                'total': total_pixels}
    

    # OBJECT LEVEL
    @staticmethod
    def patches_obj_conf_mtrx(level, labels_set, n_masks, pred_bld_mask, pred_dmg_mask,
                                y_bld_mask, y_dmg_mask):
        """ Because computing a confusion matrix for the hole batch of patches is computationally
        expensive, we are randomly sampling n. The result is the average off all confusion matrices."""
        patch_ids = [random.randint(0, len(pred_bld_mask)-1) for _ in range(n_masks)]
        # iterates over the shard
        conf_matrices = []
        for i in patch_ids:
            if level == Level.OBJ_BLD:
                conf_matrix = MatrixComputer.\
                    get_buildings(pred_bld_mask[i], y_bld_mask[i], labels_set)
            elif level == Level.OBJ_DMG:
                conf_matrix = MatrixComputer.evaluate_polys(pred_dmg_mask[i], y_dmg_mask[i], labels_set)
            conf_matrices.append(conf_matrix)
        avg_confusion = pd.DataFrame(conf_matrices).mean()
        df_avg_confusion = pd.DataFrame([avg_confusion])
        return df_avg_confusion

    @staticmethod
    def tile_obj_conf_mtrx(dmg_mask, pred_mask, labels_set):
        """Returns a confusión matrix for one image"""
        conf_mrtx = MatrixComputer.evaluate_polys(dmg_mask, pred_mask, labels_set)
        return conf_mrtx
    
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
        # functions
        def get_polygons(region, mask):
            """Returns a list of tuples (clusters,pixel_value)"""
            polys = []
            connected_components = rasterio.features.shapes(region, mask=mask)
            for shape_geojson, pixel_val in connected_components:
                shape = shapely.geometry.shape(shape_geojson)
                assert isinstance(shape, Polygon)
                polys.append((shape, int(pixel_val)))
            return polys
        
        def assing_mayority_class(blds, clusters_with_cls):
            # ESTA PARTE ES SUPER INNEFICIENTE Y se puede mejorar. Hilos y euristicas
            """Assign a class to a building based on majority vote.
            (It is assigned the class from the cluster with more superposition)"""
            # Assign majority class to each predicted polygon
            poly_overlap_class = [(0.0,None)] * len(blds)
            bld_area_label = []
            # compears the area of one polygon with all the others
            for i, (building, _) in enumerate(blds):
                for cluster_list in clusters_with_cls:
                    for cluster, label in cluster_list:

                        if not building.is_valid:
                            building = building.buffer(0)
                        if not cluster.is_valid:#????
                            cluster = cluster.buffer(0)
                        
                        intersection_area = building.intersection(cluster).area
                        if intersection_area > poly_overlap_class[i][0]:
                            poly_overlap_class[i] = (intersection_area,label)
                bld_area_label.append(
                    (building,poly_overlap_class[i][0],poly_overlap_class[i][1])
                    )

            # Build the final list of buildings with class 
            return bld_area_label

        def get_buildings(mask,label_set):
            """Return a list of (polygon,class)"""
            mask = np.array(mask).astype(np.int16)
            binary_region = np.where(mask > 0, 1, 0).astype(np.int16)
            blds = get_polygons(binary_region, binary_region > 0) #GET ALL FINAL POLYGONS
            clusters_with_cls = [get_polygons(mask, mask == c) for c in label_set] #
            blds_with_cls = assing_mayority_class(blds, clusters_with_cls)
            return blds_with_cls

        def compute_IoU(gt_buildings,pd_buildings):
            building_list = []
            # Calculate metrics for each ground truth polygon
            for id, (gt_poly, _, gt_label) in enumerate(gt_buildings):
                best_iou = 0
                best_intersection_area = 0
                best_union_area = 0

                for pd_poly, _, pd_label in pd_buildings:
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
            
            return pd.DataFrame(building_list)
        
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
                'true_neg': tn,
                'false_pos': fp, 
                'false_neg': fn,
                'total': total_buildings
            })
    
        return results