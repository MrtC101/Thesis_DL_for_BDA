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
from shapely.geometry import Polygon

from utils.metrics.common import Level



class MatrixComputer:

    @staticmethod
    def conf_mtrx_for_px_level(level, labels_set, pred_bld_mask, pred_dmg_mask,
                               y_bld_mask, y_dmg_mask):
        """Confusion pixel level metrics for masks"""
        conf_mtrx_list = []
        for cls in labels_set:
            if level == Level.PX_BLD:
                conf_mtrx = MatrixComputer.conf_mtrx_px_bld_mask(
                    pred_bld_mask, y_bld_mask, cls)
            elif level == Level.PX_DMG:
                conf_mtrx = MatrixComputer.conf_mtrx_px_cls_mask(
                    pred_dmg_mask, y_dmg_mask, y_bld_mask, cls)
            conf_mtrx_list.append(conf_mtrx)
        return pd.DataFrame(conf_mtrx_list)

    @staticmethod
    def conf_mtrx_px_bld_mask(y_preds, y_true, cls):
        """Computes the confusion matrix for the predicted building mask.(Pixel Level)"""
        y_true_binary = y_true.detach().clone()
        y_preds_binary = y_preds.detach().clone()

        # compute confusion metric
        true_pos_cls = ((y_true_binary == y_preds_binary) &
                        (y_true_binary == 1)).float().sum().item()
        false_neg_cls = ((y_true_binary != y_preds_binary) &
                         (y_true_binary == 1)).float().sum().item()
        true_neg_cls = ((y_true_binary == y_preds_binary) &
                        (y_true_binary == 0)).float().sum().item()
        false_pos_cls = ((y_true_binary != y_preds_binary) &
                         (y_true_binary == 0)).float().sum().item()

        # compute total pixels
        total_pixels = 1
        for item in y_true_binary.size():
            total_pixels *= item
        return {'class': cls, 'true_pos': true_pos_cls,
                'true_neg': true_neg_cls, 'false_pos': false_pos_cls,
                'false_neg': false_neg_cls, 'total': total_pixels}

    @staticmethod
    def conf_mtrx_px_cls_mask(y_preds, y_dmg_mask, y_bld_mask, cls):
        """Computes the confusion matrix for predicted damage mask.(Pixel Level)"""
        # Convert any other class to 0
        y_true_binary = y_dmg_mask.detach().clone()
        y_true_binary[y_true_binary != cls] = -1
        y_true_binary[y_true_binary == cls] = 1
        y_true_binary[y_true_binary == -1] = 0

        y_preds_binary = y_preds.detach().clone()
        y_preds_binary[y_preds_binary != cls] = -1
        y_preds_binary[y_preds_binary == cls] = 1
        y_preds_binary[y_preds_binary == -1] = 0

        # compute confusion metric
        true_pos_cls = ((y_true_binary == y_preds_binary) & (
            y_true_binary == 1) & (y_bld_mask == 1)).float().sum().item()
        false_neg_cls = ((y_true_binary != y_preds_binary) & (
            y_true_binary == 1) & (y_bld_mask == 1)).float().sum().item()
        true_neg_cls = ((y_true_binary == y_preds_binary) & (
            y_true_binary == 0) & (y_bld_mask == 1)).float().sum().item()
        false_pos_cls = ((y_true_binary != y_preds_binary) & (
            y_true_binary == 0) & (y_bld_mask == 1)).float().sum().item()

        # compute total pixels
        total_pixels = y_bld_mask.float().sum().item()
        return {'class': cls, 'true_pos': true_pos_cls,
                'true_neg': true_neg_cls, 'false_pos': false_pos_cls,
                'false_neg': false_neg_cls, 'total': total_pixels}

    # Compute confusión Matrixes
    @staticmethod
    def conf_mtrx_for_obj_level(level, labels_set, n_masks, pred_bld_mask, pred_dmg_mask,
                                y_bld_mask, y_dmg_mask):
        """Creates a confusión matrix for the current predicted masks."""
        # Because computing a confusion matrix for each building of each image in the batch is
        # computationally expensive, we are randomly sampling n images from the batch and
        # calculating one matrix for each of those images. Then, we obtain the average matrix
        # from all the calculated matrices.
        patch_ids = [random.randint(0, len(pred_bld_mask)-1)
                     for _ in range(n_masks)]
        # iterates over the shard
        curr_conf_matrices = []
        for i in patch_ids:
            if level == Level.OBJ_BLD:
                pred_polys, true_polys = MatrixComputer.get_buildings(
                    pred_bld_mask[i], y_bld_mask[i], labels_set)
                allowed_classes = labels_set
            elif level == Level.OBJ_DMG:
                pred_polys, true_polys = MatrixComputer.get_buildings(
                    pred_dmg_mask[i], y_dmg_mask[i], labels_set)
                allowed_classes = labels_set
            results, list_preds, list_labels = MatrixComputer.evaluate_polys(
                pred_polys, true_polys, allowed_classes, 0.1)
            for label_class in results:
                if label_class != -1:
                    curr_conf_mtrx = MatrixComputer.conf_mtrx_obj_cls_mask(results, label_class)
                    conf_mtrx = OrderedDict({"img_idx": i})
                    conf_mtrx.update(curr_conf_mtrx)
                    curr_conf_matrices.append(conf_mtrx)
        return pd.DataFrame(curr_conf_matrices)

    @staticmethod
    def conf_mtrx_obj_cls_mask(results, cls):
        true_pos_cls = results[cls]['tp'] if 'tp' in results[cls].keys() else 0
        true_neg_cls = results[cls]['tn'] if 'tn' in results[cls].keys() else 0
        false_pos_cls = results[cls]['fp'] if 'fp' in results[cls].keys() else 0
        false_neg_cls = results[cls]['fn'] if 'fn' in results[cls].keys() else 0
        return {'class': cls, 'true_pos': true_pos_cls, 'true_neg': true_neg_cls,
                'false_pos': false_pos_cls, 'false_neg': false_neg_cls, 'total': results[-1]}

    @staticmethod
    def get_polygons_with_class(mask, labels):
        # tuples of (shapely polygon, damage_class_num)
        curr_labels =[i for i in labels if i in list(np.unique(mask))] 
        polygons_and_class = []
        for c in curr_labels and curr_labels:
            # default is 4-connected for connectivity

            shapes = rasterio.features.shapes(mask, mask=(mask == c))
            for shape_geojson, pixel_val in shapes:
                shape = shapely.geometry.shape(shape_geojson)
                assert isinstance(shape, Polygon)
                polygons_and_class.append((shape, int(pixel_val)))

        return polygons_and_class

    @staticmethod
    def get_buildings(pred_mask, true_mask, labels_set):
        """
        For each tile, polygonize the prediction and label mask.

        Args:

        Returns:
            pred_polygons_and_class: list of tuples of shapely Polygon representing
                the geometry of the prediction, and the predicted class
            label_polygons_and_class: list of tuples of shapely Polygon representing
                the ground truth geometry, and the class
        """
        pred_mask = np.array(pred_mask).astype(np.uint8)
        true_mask = np.array(true_mask).astype(np.uint8)
        true_polygons_and_class = MatrixComputer.get_polygons_with_class(true_mask, labels_set)

        # 1. Detect the connected components by all non-background classes to determine the
        # predicted building blobs first (if we do this per class, a building with some pixels
        # predicted to be in another class will result in more buildings than connected components)

        background_and_others_mask = np.where(pred_mask > 0, 1, 0).astype(np.int16)
        # all non-background classes become 1

        # default is 4-connected for connectivity
        # see https://www.mathworks.com/help/images/pixel-connectivity.html
        # specify the `mask` parameter, otherwise the background will be returned as a shape
        connected_components = rasterio.features.shapes(background_and_others_mask,
                                                        mask=pred_mask > 0)
        polygons = []
        for component_geojson, pixel_val in connected_components:
            # reference: https://shapely.readthedocs.io/en/stable/manual.html#python-geo-interface
            shape = shapely.geometry.shape(component_geojson)
            assert isinstance(shape, Polygon)
            if shape.area > 20:
                polygons.append(shape)

        # 2. The majority class for each building blob is assigned to be that building's
        # predicted class.
        polygons_and_class = MatrixComputer.get_polygons_with_class(pred_mask, labels_set)

        # we take the class of the shape with the maximum overlap with the building polygon to
        # be the class of the building - majority vote
        polygons_max_overlap = [0.0] * len(polygons)  # indexed by polygon_i
        polygons_max_overlap_class = [None] * len(polygons)
        # TODO : IS THIS A KIND OF INSTERSECTION OVER UNION?
        assert isinstance(polygons, list)  # need"): the order constant

        for polygon_i, polygon in enumerate(polygons):
            for shape, shape_class in polygons_and_class:
                if not shape.is_valid:
                    shape = shape.buffer(0)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                intersection_area = polygon.intersection(shape).area
                if intersection_area > polygons_max_overlap[polygon_i]:
                    polygons_max_overlap[polygon_i] = intersection_area
                    polygons_max_overlap_class[polygon_i] = shape_class

        pred_polygons_and_class = []  # include all classes
        for polygon_i, (max_overlap_area, clss) in \
                enumerate(zip(polygons_max_overlap, polygons_max_overlap_class)):
            pred_polygons_and_class.append((polygons[polygon_i], clss))
        return pred_polygons_and_class, true_polygons_and_class

    @staticmethod
    def evaluate_polys(pred_polygons_and_class: list, label_polygons_and_class: list,
                       allowed_classes, iou_threshold: float = 0.5):
        """
        Method
        - For each predicted polygon, we find the maximum value of IoU it has with any ground truth
        polygon within the tile. This ground truth polygon is its "match".
        - Using the threshold IoU specified (typically and by default 0.5), if a prediction has
        overlap above the threshold AND the correct class, it is considered a true positive.
        All other predictions, no matter what their IOU is with any gt, are false positives.
            - Note that it is possible for one ground truth polygon to be the match for
            multiple predictions, especially if the IoU threshold is low, but each prediction
            only has one matching ground truth polygon.
        - For ground truth polygon not matched by any predictions, it is a false negative.
        - Given the TP, FP, and FN counts for each class, we can calculate the precision and recall
        for each tile *for each class*.


        - To plot a confusion table, we output two lists, one for the predictions and one for the
        ground truth polygons (because the set of polygons to confuse over are not the same...)
        1. For the list of predictions, each item is associated with the ground truth class of
        the polygon that it matched, or a "false positive" attribute.
        2. For the list of ground truth polygons, each is associated with the predicted class of
        the polygon it matched, or a "false negative" attribute.

        Args:
            pred_polygons_and_class: list of tuples of shapely Polygon representing the geometry of
                the prediction, and the predicted class
            label_polygons_and_class: list of tuples of shapely Polygon representing the ground
                truth geometry, and the class
            allowed_classes: which classes should be evaluated
            iou_threshold: Intersection over union threshold above which a predicted
                polygon is considered true positive

        Returns:
            results: a dict of dicts, keyed by the class number, and points to a dict with counts of
                true positives "tp", false positives "fp", and false negatives "fn"
            list_preds: a list with one entry for each prediction. Each entry is of the form
                {'pred': 3, 'label': 3}. This information is for a confusion matrix based on the
                predicted polygons.
            list_labels: same as list_preds, while each entry corresponds to a ground truth polygon.
                The value for 'pred' is None if this polygon is a false negative.
        """

        # the matched label polygon's IoU with the pred polygon, and the label polygon's index
        pred_max_iou_w_label = [(0.0, None)] * len(pred_polygons_and_class)

        for i_pred, (pred_poly, pred_class) in enumerate(pred_polygons_and_class):

            # cannot skip pred_class if it's not in the allowed list, as the list above
            # relies on their indices

            for i_label, (label_poly, label_class) in enumerate(label_polygons_and_class):

                if not pred_poly.is_valid:
                    pred_poly = pred_poly.buffer(0)
                if not label_poly.is_valid:
                    label_poly = label_poly.buffer(0)

                intersection = pred_poly.intersection(label_poly)
                # they should not have zero area
                union = pred_poly.union(label_poly)
                iou = intersection.area / union.area

                if iou > pred_max_iou_w_label[i_pred][0]:
                    pred_max_iou_w_label[i_pred] = (iou, i_label)

        # class: {tp, fp, fn} counts
        results = defaultdict(lambda: defaultdict(int))
        results[-1] = len(pred_polygons_and_class)
        i_label_polygons_matched = set()
        list_preds = []
        list_labels = []

        for i_pred, (pred_poly, pred_class) in enumerate(pred_polygons_and_class):

            if pred_class not in allowed_classes:
                continue

            max_iou, matched_i_label = pred_max_iou_w_label[i_pred]

            item = {
                'pred': pred_class,
                'label': label_polygons_and_class[matched_i_label][1]
                if matched_i_label is not None else None
            }

            if matched_i_label is not None:
                list_labels.append(item)

            list_preds.append(item)

            if max_iou > iou_threshold and \
                    label_polygons_and_class[matched_i_label][1] == pred_class:
                # true positive
                i_label_polygons_matched.add(matched_i_label)
                results[pred_class]['tp'] += 1
                for cls in allowed_classes:
                    if cls != pred_class:
                        results[cls]['tn'] += 1
            else:
                # false positive - all other predictions
                # note that it is a FP for the prediction's class
                results[pred_class]['fp'] += 1
                # print(matched_i_label)
                # results[matched_i_label]['fn'] += 1  # note that it is a FP for the
                # prediction's class

        # calculate the number of false negatives - how many label polygons are not matched by
        # any predictions
        for i_label, (label_poly, label_class) in enumerate(label_polygons_and_class):

            if label_class not in allowed_classes:
                continue

            if i_label not in i_label_polygons_matched:
                results[label_class]['fn'] += 1
                list_labels.append({
                    'pred': None,
                    'label': label_class
                })

        return results, list_preds, list_labels
