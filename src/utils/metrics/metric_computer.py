# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
from collections import defaultdict
import os
import sys
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
import torch

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))


class MetricComputer:
    """Class for computing confusion matrices and evaluation metrics for
      classification tasks.

    Attributes:
        type (Level): The level of classification
        (e.g., building-level, damage-level, or both).
        labels_set (list): List of class labels.
        phase_context (dict): Dictionary containing context
        information for the current phase (e.g., 'train', 'val').
        static_context (dict): Dictionary containing static context
          information.
    """

    # Compute Metrics
    @staticmethod
    def compute_metrics(conf_mtrx_list: pd.Series, label_set : list):
        conf_matrices_df = pd.concat(list(conf_mtrx_list), axis=0, ignore_index=True)
        class_metrics = MetricComputer.compute_eval_metrics(conf_matrices_df, label_set)
        return class_metrics
    
    @staticmethod
    def compute_eval_metrics(conf_mtrx_df: pd.DataFrame, labels_set : list):
        """Computes the evaluation metrics for the current epoch
            metrics are:
            - Precision
            - Recall
            - F1-score
            - Accuracy
        """
        eval_results = []
        f1_harmonic_mean = 0
        for cls in labels_set:
            class_idx = (conf_mtrx_df['class'] == cls)
            tp = conf_mtrx_df.loc[class_idx, 'true_pos'].sum()
            fp = conf_mtrx_df.loc[class_idx, 'false_pos'].sum()
            fn = conf_mtrx_df.loc[class_idx, 'false_neg'].sum()
            tn = conf_mtrx_df.loc[class_idx, 'true_neg'].sum()
            tot = conf_mtrx_df.loc[class_idx, 'total'].sum()

            precision = tp / (tp + fp) if (tp > 0 or fp > 0) else 0

            recall = tp / (tp + fn) if (tp > 0 or fn > 0) else 0

            f1 = 2 * (precision * recall) / (precision + recall) \
                if (precision > 0 or recall > 0) else 0

            accuracy = (tp + tn) / (tot) if (tot > 0) else 0

            eval_results.append({'class': cls, 'precision': precision,
                                 'recall': recall, 'f1': f1,
                                 'accuracy': accuracy})
            # F1_harmonico
            f1_harmonic_mean += 1.0 / (f1) if f1 > 0.0 else 0.0
            

        f1_harmonic_mean = len(labels_set) / \
            f1_harmonic_mean if f1_harmonic_mean > 0.0 else 0.0

        metrics = pd.DataFrame(eval_results)
        metrics.insert(0, "f1_harmonic_mean", f1_harmonic_mean)
        return metrics

    @staticmethod    
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
    
    @staticmethod
    def _compute_ap(precision, recall):
        ap = 0.0
        for i in range(len(recall)):
            if i == 0:
                ap += precision[i] * recall[i]
            else:
                ap += precision[i] * (recall[i] - recall[i-1])
        return ap
    
    @staticmethod
    def compute_PR(conf_dict: dict[torch.Tensor]):
        pr_curves = {}
        conf = torch.stack([t for t in conf_dict.values()],dim=0)
        conf_list = torch.unbind(conf,dim=1)
        for i_label in range(conf.shape[1]):
            tp, fp, fn, tn = torch.unbind(conf_list[i_label],dim=1)
            precision = tp / (tp + fp)
            precision = torch.where(torch.isnan(precision), torch.tensor(1.0), precision)
            recall = tp / (tp + fn)
            recall = torch.where(torch.isnan(recall), torch.tensor(0.0), recall)
            base_precision = tp.max() / (tp.max() + tn.max())
            pr_curves[i_label] = (recall, precision, base_precision,
                                  MetricComputer._compute_ap(precision,recall)) 
        return pr_curves