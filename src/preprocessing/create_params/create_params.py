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
    dmg_w = [round(w,4) for w in train_weights["dmg"].values()]
    dmg_w.reverse()
    seg_w = [round(w,4) for w in train_weights["seg"].values()]
    seg_w.reverse()

    configs = {**params['train'], **params['visual'],
               **params['preprocessing'], **params['weights']}
    configs["weights_dmg"] = dmg_w
    configs["weights_seg"] = seg_w
    param_combinations = list(ParameterGrid(params['hyperparameter']))
    configs = [(i, {**configs, **params})
               for i, params in enumerate(param_combinations)]
    out_path.join("conf_list.json").save_json(configs)
