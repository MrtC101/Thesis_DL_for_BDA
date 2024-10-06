# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import pandas as pd


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
    def compute_metrics(conf_mtrx_list: pd.Series, label_set: list):
        conf_matrices_df = pd.concat(list(conf_mtrx_list), axis=0, ignore_index=True)
        class_metrics = MetricComputer.compute_eval_metrics(conf_matrices_df, label_set)
        return class_metrics

    @staticmethod
    def compute_eval_metrics(conf_mtrx_df: pd.DataFrame, labels_set: list):
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
            tot = tp + fp + fn + tn

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

        f1_harmonic_mean = len(labels_set) / f1_harmonic_mean if f1_harmonic_mean > 0.0 else 0.0

        metrics = pd.DataFrame(eval_results)
        metrics.insert(0, "f1_harmonic_mean", f1_harmonic_mean)
        return metrics
