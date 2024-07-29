# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import os
from sklearn.model_selection import ParameterGrid
from utils.common.pathManager import FilePath


def create_params(params) -> list[dict]:
    """
    Create a list of configuration dictionaries for hyperparameter
      optimization.

    Args:
        configs (dict): Base configuration dictionary with default settings.

    Returns:
        list[dict]: List of configuration dictionaries with different
          hyperparameter combinations.
    """
    out_path = FilePath(os.environ["OUT_PATH"])
    train_weights = out_path.join("train_weights.json").read_json()
    weights = [1]  # w for label 0
    lab_w = [round(w) for w in train_weights.values()]
    lab_w.reverse()
    weights.extend(lab_w)  # weights for all classes

    configs = {**params['train'], **params['visual'],
               **params['preprocessing'], **params['weights']}
    configs["weights_dmg"] = weights
    param_combinations = list(ParameterGrid(params['hyperparameter']))
    configs = [(i, {**configs, **params})
               for i, params in enumerate(param_combinations)]
    out_path.join("conf_list.json").save_json(configs)
