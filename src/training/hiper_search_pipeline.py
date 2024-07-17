from joblib import Parallel, delayed
from tqdm import tqdm
from utils.common.pathManager import FilePath
from training.cross_validation_pipeline import k_cross_validation
from utils.loggers.console_logger import LoggerSingleton
from sklearn.model_selection import ParameterGrid


def _create_params(configs: dict) -> list[dict]:
    """
    Create a list of configuration dictionaries for hyperparameter
      optimization.

    Args:
        configs (dict): Base configuration dictionary with default settings.

    Returns:
        list[dict]: List of configuration dictionaries with different
          hyperparameter combinations.
    """
    hyperparameter_config = {
        'init_learning_rate': [0.0005],
        'tot_epochs': [1],
        'batch_size': [25]
    }
    param_combinations = list(ParameterGrid(hyperparameter_config))
    return [{**configs, **params} for params in param_combinations]


def _start_k_fold(folds: int, index: int,
                  config: dict, paths: dict) -> tuple[int, float]:
    """
    Run k-fold cross-validation for a given configuration.

    Args:
        folds (int): Number of folds for cross-validation.
        index (int): Index of the current configuration.
        config (dict): Configuration dictionary.
        paths (dict): Dictionary of paths used in the process.

    Returns:
        tuple[int, float]: Index and score of the configuration.
    """
    paths['out_dir'] = FilePath(paths['out_dir']).join(f'config-{index}')
    config['configuration_num'] = index
    score = k_cross_validation(folds, config, paths)
    return index, score


def parameter_search(folds: int, configs: dict, paths_dict: dict) -> dict:
    """
    Perform hyperparameter search using k-fold cross-validation.

    Args:
        folds (int): Number of folds for cross-validation.
        configs (dict): Base configuration dictionary.
        paths_dict (dict): Dictionary of paths used in the process.

    Returns:
        dict: Best configuration dictionary.
    """
    log_out = FilePath(paths_dict['out_dir']).join(
        "hyperparameter_console_logs")
    log = LoggerSingleton("HYPERPARAMETER_SEARCH", log_out)
    param_list = _create_params(configs)
    log.info(f"Cantidad de configuraciones: {len(param_list)}.")
    results = [_start_k_fold(folds, i, config, paths_dict)
               for i, config in enumerate(tqdm(param_list))]
    # results = Parallel(n_jobs=-1)(delayed(_start_k_fold)
    #                              (folds, i, config, paths_dict)
    #                              for i, config in enumerate(tqdm(param_list)))
    best_index, best_acc = min(results, key=lambda x: x[1])

    log.info(f"Configuration number {best_index}" +
             f"with a validation loss of {best_acc:.4f}")
    best_params = param_list[best_index]
    return best_params
